r"""Train the CTAdipo landmark model: five craniocaudal planes from two body-masked projections.

This is the whole training path in one file, in two stages:

    extract   raw volumes + a reviewed label CSV  ->  one .npz of features per scan
    train     those .npz                          ->  fold0.pt .. fold4.pt + the out-of-fold report
    all       both, in order

WHAT THE MODEL IS FOR. The original pipeline read five z-planes -- lung apex, diaphragm, cranial and
caudal kidney, bladder -- plus the craniocaudal direction from tables a reviewer filled in scan by
scan. Those planes cut the ten sub-depots out of VAT and SUBQ, so every per-depot number CTAdipo
reports depends on them. This model predicts them, which is what removes the last human step.

WHY PROJECTIONS AND NOT THE VOLUME. Every label is a SCALAR z-plane, so the target is one number per
landmark. A coronal and a sagittal projection keep the anatomy those planes are defined on -- rib
cage, lung field, kidney silhouettes, pelvic brim -- while collapsing the axis that carries no label
at all. A volume of 89 M voxels at the median of this cohort, 151 M at the largest, becomes
8 x 256 x 128. Training is about 15 min per fold on one RTX 4080; at inference the five folds and
the mirror TTA are 0.3 s on CPU, and predict() end to end is about 9 s, dominated by the body mask
and the projections rather than by the network.

WHY A DISTRIBUTION AND NOT A REGRESSED NUMBER. The network emits a distribution over z per landmark
and the position is its expectation (soft-argmax). That buys sub-bin precision, a gradient far
better conditioned than regressing one scalar through a global pool, and -- the reason the app cares
-- a confidence, because the spread of that distribution is a usable "this scan is unusual" signal.
Measured out of fold, flagging on spread catches 73% of the >5 mm errors at the cost of flagging
10.9% of scans; 6.2% of scans miss by >5 mm on at least one plane and 2.0% by >10 mm.

MEASURED ACCURACY, out of fold on all 1,186 scans, absolute error in mm:

    landmark          median     p90
    lung_apex           0.51    1.54
    diaphragm           0.48    2.22
    kidney_cranial      0.66    2.07
    kidney_caudal       0.79    2.60
    bladder             0.84    2.70

Craniocaudal direction is not predicted separately -- it is whether lung_apex came out cranial of
bladder -- and agrees with the reviewed table on 1182/1186 = 99.66%.

THE LABEL CSV. One row per scan. `Animal` must equal the volume's file name without its extension,
because that is the only key joining the two. The five plane columns are in VOXELS along z, in the
volume's own index space:

    Animal,lung_apex_z,diaphragm_z,kidney_cranial_z,kidney_caudal_z,bladder_z
    mouse2072_12M,429,347,323,228,124

Extra columns are ignored. Rows whose `Animal` has no matching volume are skipped and counted.

WHAT IS NOT IN THIS FILE. The feature builder and the network live in `ctadipo/landmark_model.py`
and are imported, never copied -- see the comment at the import.

Reproducing the published model:

    python CTAdipo_train.py all --scans <dir> [<dir> ...] --labels <csv>

On the 1,186 expert-reviewed scans that produced the published model, extraction runs at roughly
30 scans/min on five reader threads and each of the five folds takes about 15 min on an RTX 4080.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# THE IMPORT IS THE POINT, not a convenience. The features are non-trivial -- body-masked
# projections, a hole-filled mask, per-view standardisation over body pixels only -- and if this
# trainer carried its own copy of them the two would drift, silently, and the published model would
# end up trained on something slightly different from what the app feeds it at inference. There is
# one implementation of the features and the network and both sides import it. Everything defined
# below this line is training-only machinery -- label joining, augmentation, the split, the loss --
# that inference never executes, which is why it lives here and not in the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctadipo.landmark_model import (  # noqa: E402
    ISO, LM, ZR, LandmarkNet, animal_mask, features, soft_argmax, standardise)

# The CSV column names, derived from LM rather than typed out again, so a rename in the package
# cannot leave this script silently reading five columns that are no longer the right ones.
LABEL_COLS = [k + "_z" for k in LM]

# Suffixes recognised when turning a volume's file name into an `Animal` key. `.nii.gz` is two
# extensions, which Path.stem gets wrong on its own -- it would key the scan as "mouse2072_12M.nii".
VOL_SUFFIXES = (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd")

# The trailing token that marks a repeat scan of one mouse rather than a different mouse. See
# split_by_animal() for why this matters more than it looks.
TIMEPOINTS = {"Baseline", "3M", "6M", "12M"}


# ------------------------------------------------------------------------------------- extract
def animal_key(p: Path) -> str:
    """File name minus its extension. This is the join key to the label CSV and nothing else."""
    n = p.name
    for s in VOL_SUFFIXES:
        if n.lower().endswith(s):
            return n[:-len(s)]
    return p.stem


def build_index(dirs, pattern):
    """Map animal key -> volume path. Earlier directories win, so a curated or re-reconstructed
    copy can shadow the archive simply by being listed first on the command line."""
    idx = {}
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print("  warning: %s does not exist, skipped" % d, flush=True)
            continue
        for p in sorted(d.glob(pattern)):
            idx.setdefault(animal_key(p), p)
    return idx


def any_volume_matched(rows, no_vol):
    """True if at least one label row found its volume. Separating this from `todo` is what lets
    cmd_extract tell a finished run apart from a join that matched nothing."""
    return len(no_vol) < len(rows)


def body_z_extent(A):
    """First and last z slice holding a real cross-section of the animal.

    Stored alongside the features but deliberately NOT used as a target. Targets are fractions of
    the VOLUME's z axis, not of the body's, because the volume is what the model sees: a fraction of
    it is directly readable from the image and exactly invertible at inference, whereas a
    body-relative coordinate would make every label depend on a mask that can fail. With head and
    hindlimbs in or out of the field of view from scan to scan, "fraction of the animal" is not a
    fixed anatomical quantity in this cohort anyway. The extent is kept only so a later audit can
    ask how much of the animal was actually in frame on a scan the model got wrong.
    """
    per = A.sum((1, 2))
    on = np.where(per > 0.0005 * A.shape[1] * A.shape[2])[0]
    return (int(on[0]), int(on[-1])) if on.size else (0, A.shape[0] - 1)


def cmd_extract(a):
    """Volumes -> one .npz of features per scan.

    EVERY PROJECTION IS TAKEN INSIDE THE ANIMAL MASK, AND THAT IS THE WHOLE TRICK. The first version
    projected over the raw ray and the two air channels came out as pure noise on rendered
    inspection: background air sits at -875 to -1000 HU and lung at -300 to -400, so the darkest
    voxel on any ray that leaves the animal is always background, and the lung field -- the one
    structure the thoracic landmarks are defined on -- is invisible. Masking first makes the minimum
    mean "darkest TISSUE on this ray", which is the lung.

    THE MASK MUST HAVE ITS HOLES FILLED, for the opposite reason. It is thresholded at -500 HU, so
    the lungs fall BELOW it and would be excluded as background, re-creating exactly the blindness
    the masking was meant to cure. animal_mask() in the package fills them, and it is the same mask
    the app's z-scoring uses, deliberately: largest 3-D connected component above -500 HU, filled.
    3-D connectivity is what excludes the sample holder, which can be the widest object in a single
    slice but is a separate object in 3-D.

    FOUR CHANNELS PER VIEW, none of them redundant:
        bone   max inside body    first rib, floating ribs, pelvic brim -- the skeletal edges
        air    min inside body    lung field and bowel gas, what the thoracic landmarks sit on
        bulk   mean inside body   soft-tissue mass; separates diaphragm from last rib
        thick  body depth in mm   silhouette and girth, pure shape, free to compute
    A bone-only set would lose the lungs; an air-only set would lose the ribs.

    NO SCANNER NORMALISATION HERE. The ~275 HU offset between lean 3M scans and obese Aging scans is
    itself informative, and the data is not HU-calibrated in any case -- the air peak runs from -773
    to -1424 HU across the two scanners that produced it. Per-scan standardisation happens at
    training time instead, which keeps the choice reversible without re-extracting 1,186 volumes.

    Work is threaded rather than multiprocessed: SimpleITK's reader and the per-slice NumPy
    reductions both release the GIL, and a thread pool avoids pickling a several-hundred-MB volume
    between processes. Scans whose .npz already exists are skipped, so an interrupted run resumes.
    """
    # Imported here rather than at module scope so that `train` runs in an environment with no
    # volume reader installed -- the features are already .npz by then.
    import pandas as pd
    import SimpleITK as sitk

    out = Path(a.feat_out)
    out.mkdir(parents=True, exist_ok=True)
    idx = build_index(a.scans, a.pattern)
    print("index: %d volumes matching %s" % (len(idx), a.pattern), flush=True)

    lab = pd.read_csv(a.labels, dtype={"Animal": str})
    missing = [c for c in ["Animal"] + LABEL_COLS if c not in lab.columns]
    if missing:
        raise SystemExit("label CSV %s is missing column(s): %s\nexpected: %s"
                         % (a.labels, ", ".join(missing), ", ".join(["Animal"] + LABEL_COLS)))

    rows = [r for _, r in lab.iterrows()]
    no_vol = [r.Animal for r in rows if r.Animal not in idx]
    todo = [r for r in rows if r.Animal in idx and not (out / (r.Animal + ".npz")).exists()]
    if a.limit:
        todo = todo[:a.limit]
    print("labels: %d rows, %d with no matching volume, %d already extracted, %d to do"
          % (len(rows), len(no_vol), len(rows) - len(no_vol) - len(todo), len(todo)), flush=True)
    if no_vol:
        print("  no volume for: %s%s" % (", ".join(no_vol[:5]), " ..." if len(no_vol) > 5 else ""),
              flush=True)
    # A join that matched NOTHING is a misconfiguration, not a finished run -- almost always a
    # mistyped --scans, the wrong --pattern, or an `Animal` column that is not the file name. It
    # has to be an error rather than a quiet exit 0, because under `all` a silent success here
    # would send `train` on to whatever happened to be sitting in --out already.
    if rows and not any_volume_matched(rows, no_vol):
        raise SystemExit(
            "no volume matched any of the %d label rows.\n"
            "  --scans   %s\n"
            "  --pattern %s\n"
            "  --labels  %s\n"
            "`Animal` must equal the volume's file name without its extension."
            % (len(rows), " ".join(str(d) for d in a.scans), a.pattern, a.labels))
    if not todo:
        print("nothing to extract (all %d matched scans are already done)"
              % (len(rows) - len(no_vol)), flush=True)
        return 0
    print("extract: %d scans, %d workers -> %s" % (len(todo), a.workers, out), flush=True)

    def work(r):
        t = time.time()
        vol = sitk.GetArrayFromImage(sitk.ReadImage(str(idx[r.Animal])))
        A = animal_mask(vol)
        f = features(vol, A)
        z0, z1 = body_z_extent(A)
        np.savez_compressed(out / (r.Animal + ".npz"), feat=f,
                            z=np.array([float(r[k]) for k in LABEL_COLS], np.float32),
                            shape=np.array(vol.shape, np.int32),
                            body=np.array([z0, z1], np.int32))
        return r.Animal, time.time() - t

    if a.workers <= 1:
        for r in todo:
            aid, dt = work(r)
            print("  %-22s %.1fs" % (aid, dt), flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor
        n, t0 = 0, time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            for aid, dt in ex.map(work, todo):
                n += 1
                if n % 25 == 0 or n == len(todo):
                    el = time.time() - t0
                    print("  %4d/%d  %.0fs, %.1f/min, eta %.0f min"
                          % (n, len(todo), el, n / el * 60,
                             (len(todo) - n) / (n / el) / 60), flush=True)
    print("done: %d npz in %s" % (len(list(out.glob("*.npz"))), out), flush=True)
    return 0


# ---------------------------------------------------------------------------------------- data
def group_of(aid):
    """Animal identity from a scan key. mouse1546_12M and ctrl_13_Baseline are repeat scans of one
    mouse, so both collapse to the mouse rather than staying two independent examples."""
    p = aid.split("_")
    return "_".join(p[:-1]) if len(p) > 1 and p[-1] in TIMEPOINTS else aid


def load_all(d):
    """Every .npz in a directory, standardised, with targets as fractions of the z axis.

    The `+ 0.5` is the bin-centre convention and it is not cosmetic. soft_argmax() returns
    (index + 0.5)/n, so a target of plain z/Z would sit half a bin away from anywhere the network
    can ever point, and every landmark would carry a constant bias of half a slice -- 0.075 mm,
    which is a sixth of the diaphragm's median error. predict() inverts it with the matching
    `frac * Z - 0.5`.

    Scans carrying a non-finite feature or label are dropped rather than repaired: a NaN here means
    the mask or the CSV is wrong for that scan, and substituting a number would quietly put a
    fabricated landmark into the training set.
    """
    X, Y, Z, A = [], [], [], []
    for f in sorted(Path(d).glob("*.npz")):
        dd = np.load(f)
        feat, z, shp = dd["feat"], dd["z"], dd["shape"]
        if not np.isfinite(z).all() or not np.isfinite(feat).all():
            print("  skipped %s (non-finite)" % f.stem, flush=True)
            continue
        X.append(standardise(feat))
        Y.append(((z + 0.5) / shp[0]).astype(np.float32))
        Z.append(int(shp[0]))
        A.append(f.stem)
    if not X:
        raise SystemExit("no usable .npz in %s -- run the `extract` stage first" % d)
    return np.stack(X), np.stack(Y), np.array(Z), np.array(A)


def augment(x, y, rng):
    """x (10, ZR, W) float32, y (5,) fractions of the z axis. Both are transformed together.

    THE CRANIOCAUDAL FLIP IS THE IMPORTANT ONE. Flipping the projections and the labels turns a
    head-up scan into a head-down one, which is exactly the `direction` the original pipeline took
    from a human-reviewed table. Only 171 of the 1,186 scans are `low` -- 14% -- far too few to
    learn the orientation from directly; the flip makes the model see both equally often and removes
    the need for the table entirely. It is also an exact symmetry of the training distribution,
    which is why predict() can average over the mirror at inference and pay nothing in correctness.

    The field-of-view crop teaches the model that the animal need not fill the frame -- crop size
    varies across this cohort and is confounded with timepoint -- and it always keeps all five
    labels inside, because a scan whose bladder has been cropped away has no bladder to learn from.

    The jitter and the noise are re-masked by the validity channels afterwards. Fill must stay
    EXACTLY zero: an additive offset applied to fill would turn the background into a second,
    scan-varying intensity, and the network could then read the frame geometry off it instead of
    the anatomy.
    """
    if rng.random() < 0.5:                              # craniocaudal flip == direction high/low
        x, y = x[:, ::-1].copy(), (1.0 - y).astype(np.float32)
    if rng.random() < 0.7:                              # field of view, all labels kept inside
        lo, hi = float(y.min()), float(y.max())
        a = rng.uniform(0.0, max(lo - 0.04, 0.0))
        b = rng.uniform(min(hi + 0.04, 1.0), 1.0)
        if b - a > 0.35:
            idx = np.clip(((np.arange(ZR) + 0.5) / ZR * (b - a) + a) * ZR - 0.5, 0, ZR - 1)
            i0 = np.floor(idx).astype(int)
            w = (idx - i0).astype(np.float32)
            i1 = np.minimum(i0 + 1, ZR - 1)
            x = x[:, i0] * (1 - w)[None, :, None] + x[:, i1] * w[None, :, None]
            y = ((y - a) / (b - a)).astype(np.float32)
    keep = x[8:10] > 0                                  # fill must stay exactly zero through jitter
    x = x.copy()
    x[:8] = x[:8] * rng.uniform(0.9, 1.1, (8, 1, 1)).astype(np.float32) \
        + rng.uniform(-0.1, 0.1, (8, 1, 1)).astype(np.float32)
    x[:8] += rng.normal(0, 0.02, x[:8].shape).astype(np.float32)
    x[:4] *= keep[0][None]
    x[4:8] *= keep[1][None]
    return x, y


def make_dataset(torch):
    """The Dataset class, built inside a function so that importing this module costs no torch and
    `extract` can run on a machine that has none."""

    class DS(torch.utils.data.Dataset):
        def __init__(self, X, Y, train, seed=0):
            self.X, self.Y, self.train = X, Y, train
            self.rng = np.random.default_rng(seed)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, i):
            x, y = self.X[i], self.Y[i].copy()
            if self.train:
                x, y = augment(x, y, self.rng)
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(y)

    return DS


def split_by_animal(A, folds, seed):
    """Assign folds to ANIMALS, then give every scan its animal's fold.

    THIS IS THE ONE SPLIT DECISION THAT CHANGES THE HEADLINE NUMBER. The longitudinal arm of the
    cohort scans the SAME MOUSE at Baseline, 6M and 12M, so the scans are not independent: on the
    published set of 1,186, group_of() above resolves them to 856 animals -- 637 scanned once, 108
    twice, 111 three times, so 549 of the 1,186 scans have a sibling. Under a per-scan split an
    animal's Baseline would sit in train while its 12M sat in validation -- two projections of one
    skeleton, differing mostly in fat -- and the model would be scored on a mouse whose rib
    spacing, kidney position and pelvis it had already memorised. The number that comes out of that
    is not wrong about the data; it is simply not an answer to the question the app asks, which is
    always about a mouse the model has never seen.

    The permutation is seeded so the published five folds are reproducible from this script.
    """
    g = np.array([group_of(x) for x in A])
    ug = np.unique(g)
    rng = np.random.default_rng(seed)
    fold_of_group = dict(zip(ug, rng.permutation(len(ug)) % folds))
    return np.array([fold_of_group[x] for x in g]), len(ug)


# --------------------------------------------------------------------------------------- train
def loss_fn(torch, logits, y, sigma_bins=3.0):
    """Cross-entropy to a Gaussian centred on the truth, plus 0.1 x L1 on the expectation.

    The heatmap term shapes the whole distribution, so the network learns where the evidence is; the
    L1 term is what the app actually reads out. Either alone is worse -- pure L1 lets a bimodal
    distribution average to the right answer for the wrong reason, which also destroys the spread as
    a confidence signal, and pure cross-entropy ignores sub-bin position, which is most of the
    0.5 mm.

    sigma is in BINS, so it is 3/256 of the volume's z axis and scales with the scan rather than
    imposing one fixed millimetre tolerance across animals of different length.
    """
    n = logits.shape[-1]
    pos = (torch.arange(n, device=logits.device, dtype=logits.dtype) + 0.5) / n
    g = torch.exp(-0.5 * ((pos[None, None] - y[..., None].to(logits.dtype)) * n / sigma_bins) ** 2)
    g = g / g.sum(-1, keepdim=True).clamp_min(1e-8)
    ce = -(g * logits.log_softmax(-1)).sum(-1).mean()
    zhat, _ = soft_argmax(logits)
    return ce + 0.1 * ((zhat - y).abs() * n).mean(), zhat


def run_fold(torch, X, Y, tr, va, args, dev, seed):
    """One fold: train on `tr`, return the trained net and its predictions on `va`.

    OneCycle rather than a plateau schedule because the run is short and fixed-length -- 120 epochs,
    no early stopping, no validation-driven choices of any kind. That is not laziness: every
    validation scan here is an out-of-fold scan whose prediction is reported as the headline
    accuracy, so touching it to pick a stopping point or a learning rate would contaminate exactly
    the number the app is sold on.

    AMP is enabled only under CUDA. The soft-argmax expectation accumulates in the autocast dtype,
    and on CPU half precision gains nothing while costing precision in the very quantity being
    reported. `torch.amp` is the current spelling of the old `torch.cuda.amp` entry points; same
    implementation, no deprecation warning.
    """
    DS = make_dataset(torch)
    torch.manual_seed(seed)
    dl = torch.utils.data.DataLoader(DS(X[tr], Y[tr], True, seed), batch_size=args.bs,
                                     shuffle=True, num_workers=0, drop_last=True)
    net = LandmarkNet().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr,
                                              total_steps=args.epochs * max(len(dl), 1))
    cuda = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    for ep in range(args.epochs):
        net.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev, non_blocking=True), yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=cuda):
                l, _ = loss_fn(torch, net(xb), yb)
            scaler.scale(l).backward()
            scaler.step(opt)
            scaler.update()
            sch.step()
            tot += float(l.item())
        if (ep + 1) % 30 == 0 or ep == args.epochs - 1:
            print("      ep %3d  loss %.4f" % (ep + 1, tot / max(len(dl), 1)), flush=True)
    net.eval()
    P = []
    with torch.no_grad():
        for i in range(0, len(va), 32):
            xb = torch.from_numpy(X[va[i:i + 32]]).to(dev)
            P.append(soft_argmax(net(xb))[0].float().cpu().numpy())
    return net, np.concatenate(P)


def cmd_train(a):
    """Features -> fold*.pt, oof.npz, oof_summary.json, and the out-of-fold table.

    NO FOLD IS HELD BACK AS A FINAL TEST, and that is the right call here rather than a corner cut.
    With five folds every one of the 1,186 scans is predicted by a model that never saw its animal,
    so the out-of-fold set IS the test set and it is 1,186 scans rather than a fifth of them --
    which matters when the tail is the interesting part (2.0% of scans miss by >10 mm; a 237-scan
    holdout would contain about five of them). The app then averages all five folds in probability
    space, which is a different estimator from any single fold and could not have been scored by a
    held-out split anyway.
    """
    import torch

    out = Path(a.model_out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device) if a.device else \
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s | loading %s ..." % (dev, a.features), flush=True)
    X, Y, Zn, A = load_all(a.features)
    print("loaded %d scans, X %s (%.2f GB)" % (len(X), X.shape, X.nbytes / 1e9), flush=True)

    fold, n_animals = split_by_animal(A, a.folds, a.seed)
    print("%d scans / %d animals / %d folds grouped by animal"
          % (len(A), n_animals, a.folds), flush=True)

    oof = np.zeros_like(Y)
    for k in range(a.folds):
        tr, va = np.where(fold != k)[0], np.where(fold == k)[0]
        print("  fold %d: train %d, val %d" % (k, len(tr), len(va)), flush=True)
        t0 = time.time()
        net, p = run_fold(torch, X, Y, tr, va, a, dev, seed=k)
        oof[va] = p
        torch.save(net.state_dict(), out / ("fold%d.pt" % k))
        e = np.abs(p - Y[va]) * Zn[va][:, None] * ISO
        print("    %.0fs | median mm: %s" % (time.time() - t0,
              "  ".join("%s %.2f" % (LM[j][:9], np.median(e[:, j])) for j in range(5))), flush=True)

    # Error in MILLIMETRES, not in bins or fractions. A fraction of z is a different distance on a
    # 516-slice scan than on a 540-slice one, and mm is the only unit in which the five landmarks --
    # and the reviewer's own median correction, which they are being compared against -- are
    # commensurable.
    err = np.abs(oof - Y) * Zn[:, None] * ISO
    print("\n=== out-of-fold error in mm (splits grouped by animal) ===")
    print("%-16s %8s %8s %8s %8s" % ("landmark", "median", "p90", "p95", "max"))
    for j, k in enumerate(LM):
        print("%-16s %8.2f %8.2f %8.2f %8.2f"
              % (k, np.median(err[:, j]), np.percentile(err[:, j], 90),
                 np.percentile(err[:, j], 95), err[:, j].max()))
    np.savez(out / "oof.npz", pred=oof, true=Y, Z=Zn, animal=A, fold=fold, err_mm=err)
    with open(out / "oof_summary.json", "w") as fh:
        json.dump({"landmarks": LM,
                   "median_mm": {k: float(np.median(err[:, j])) for j, k in enumerate(LM)},
                   "p95_mm": {k: float(np.percentile(err[:, j], 95)) for j, k in enumerate(LM)}},
                  fh, indent=2)
    print("\nsaved -> %s" % out)
    return 0


# ----------------------------------------------------------------------------------------- cli
LABEL_HELP = r"""
the label CSV
-------------
One row per scan, these columns (any others are ignored):

  Animal             the volume's file name without its extension -- the join key
  lung_apex_z
  diaphragm_z        the five planes, in VOXELS along z,
  kidney_cranial_z   in the volume's own index space
  kidney_caudal_z
  bladder_z

  Animal,lung_apex_z,diaphragm_z,kidney_cranial_z,kidney_caudal_z,bladder_z
  mouse2072_12M,429,347,323,228,124

Nothing in this script knows any path on the machine it was written on. Point --scans at the
directories holding the volumes and --labels at the CSV above.
"""


def _add_extract_args(p):
    p.add_argument("--scans", nargs="+", required=True, metavar="DIR",
                   help="directories of volumes; earlier ones win on a name clash")
    p.add_argument("--labels", required=True, metavar="CSV",
                   help="reviewed landmark table (see the notes at the end of this help)")
    p.add_argument("--out", dest="feat_out", default="traindata/landmarks", metavar="DIR",
                   help="where the .npz features go (default: %(default)s)")
    p.add_argument("--pattern", default="*.nii.gz",
                   help="glob matching the volumes (default: %(default)s)")
    p.add_argument("--limit", type=int, default=0,
                   help="stop after N scans; for a smoke test")
    p.add_argument("--workers", type=int, default=5,
                   help="reader threads (default: %(default)s)")


def _add_train_args(p, with_features=True):
    if with_features:
        p.add_argument("--features", default="traindata/landmarks", metavar="DIR",
                       help="directory of .npz from the extract stage (default: %(default)s)")
    p.add_argument("--model-out", dest="model_out", default="model", metavar="DIR",
                   help="where fold*.pt and the out-of-fold report go (default: %(default)s)")
    p.add_argument("--epochs", type=int, default=120, help="default: %(default)s")
    p.add_argument("--bs", type=int, default=16, help="batch size (default: %(default)s)")
    p.add_argument("--lr", type=float, default=3e-4, help="AdamW peak lr (default: %(default)s)")
    p.add_argument("--folds", type=int, default=5, help="default: %(default)s")
    p.add_argument("--seed", type=int, default=0,
                   help="seeds the animal->fold permutation; 0 reproduces the published split")
    p.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="CTAdipo_train.py",
        description="Train the CTAdipo landmark model: five craniocaudal planes from two "
                    "body-masked projections. Two stages -- `extract` turns volumes into .npz "
                    "features, `train` turns those into the five fold checkpoints the app loads.",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=LABEL_HELP)
    sub = ap.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("extract", help="volumes -> .npz features",
                        description=cmd_extract.__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=LABEL_HELP)
    _add_extract_args(pe)

    pt = sub.add_parser("train", help="features -> fold*.pt + the out-of-fold report",
                        description=cmd_train.__doc__,
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    _add_train_args(pt)

    # `all` takes no --features: train reads exactly what extract just wrote, which is the only
    # way the two stages cannot be pointed at different feature sets by accident.
    pa = sub.add_parser("all", help="extract, then train on what it produced",
                        description="Run extract, then train on its output directory.",
                        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=LABEL_HELP)
    _add_extract_args(pa)
    _add_train_args(pa, with_features=False)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "extract":
        return cmd_extract(a)
    if a.cmd == "train":
        return cmd_train(a)
    rc = cmd_extract(a)
    if rc:
        return rc
    a.features = a.feat_out
    return cmd_train(a)


if __name__ == "__main__":
    raise SystemExit(main())
