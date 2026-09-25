r"""One upload in, the full depot table out. Everything between is here.

This is the headless core; the Shiny app is a thin wrapper on it, which is deliberate -- the numbers
have to be testable without a browser, because the acceptance test for the whole project is that a
scan already in _doz_long_v3.csv comes back with the same volumes.

THE MASK DICTIONARY IS THE CONTRACT. cohort_derive.derive_masks defines the shape the depot rules
consume, and this function reproduces that shape exactly so depot_rules runs UNCHANGED against it.
That matters more than it looks: the ten depot definitions, the dorsoventral frame, the spine
midline and the conservation checks are all code that has already been validated page by page, and
re-implementing any of it to suit a new caller is how an app quietly stops agreeing with the paper.

WHAT CTAdipo SUPPLIES THAT THE ORIGINAL PIPELINE TOOK FROM A HUMAN TABLE:
    five landmark planes     the landmark model
    craniocaudal direction   implied by the landmarks (lung apex relative to bladder)
    dorsoventral axis        geometry, with the decision margin surfaced and a manual override
    lung mask                the same sweep the cohort used, pinned to the reviewed configuration
    hardware polygons        NOT supplied, and deliberately so

HARDWARE. The original derivation subtracts hand-drawn polygons keyed to scanner AND reconstruction
crop, because on that rig the breathing pad fuses to the animal and its dark interior is labelled
cavity, i.e. counted as visceral fat. Those polygons cannot transfer to another scanner and shipping
them would be worse than useless. CTAdipo does what generalises -- keep_largest3d, which removes
DETACHED hardware -- and shows the user the mask so fused hardware is visible rather than silent.

LUNG. Lung reads -300 to -400 HU, inside both fat bands, so unsubtracted lung is counted as fat.
Measured across the adiposity range, that is ~1.5% of fat in the fattest animals and ~16% in the
leanest, correlating with log total fat at r = -0.83. Skipping it is therefore NOT neutral: it
inflates lean animals specifically, along the exact axis these measurements are used to study. The
caller passes `lung_fn`; when it is absent that fact is reported, never passed over quietly.
"""
from __future__ import annotations

import gc
import os

import numpy as np

from . import preprocess as P
from . import landmark_model as LMM

def rss_gb():
    """Resident memory, or 0.0 where psutil is absent. Printed at every stage so a container log
    reports the real peak instead of leaving it to be inferred from the fact that it died."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        try:
            with open("/proc/self/statm") as fh:          # Linux fallback, no dependency
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
        except Exception:
            return 0.0


def trim_heap():
    """Hand freed memory back to the kernel, which glibc does not do on its own.

    THIS IS WHY THE DESKTOP MEASUREMENT DID NOT PREDICT THE WORKER. Dropping the segmenter frees
    1.07 GB from Python's heap, and on Windows that showed up in RSS immediately -- peak measured
    3.58 GB and the fix looked done. glibc instead keeps the freed arenas mapped for reuse, and the
    container limit is enforced on RSS, not on what Python believes it is using. So the same code
    that peaked at 3.58 GB locally entered the depot partition at 3.94 GB on the worker.

    malloc_trim(0) releases the free top of the heap. It exists only in glibc, so musl containers
    and Windows fall through the except and are no worse off than before.
    """
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        return True
    except Exception:
        return False


DENSITY_G_PER_ML = 0.9          # adipose tissue; volumes convert to mass by simple density
VOX_MM3 = P.ISO ** 3

# THE TEN that tile VAT and SUBQ exactly, and are what the paper reports.
VAT_DEPOTS = ["perigonadal", "mesenteric", "retroperitoneal", "thoracic_VAT"]
SUBQ_DEPOTS = ["anterior_SUBQ", "dorsolumbar_SUBQ", "inguinal_SUBQ",
               "gluteal_SUBQ", "hindlimb_SUBQ", "head_neck_SUBQ"]
PARTITION_DEPOTS = VAT_DEPOTS + SUBQ_DEPOTS

# rule_A_slabs ALSO returns depots that OVERLAP the ten and each other, so they must never be summed
# with them or shown beside them unmarked. Note this is depot_rules.OVERLAP, which is a different
# set from depot_rules.PROVISIONAL -- PROVISIONAL marks depots whose DEFINITION is not settled and
# includes `mesenteric` and `thoracic_VAT`, both of which are among the ten. The app reads
# PROVISIONAL off the rules module itself and flags those two rather than silently presenting them
# as settled; it is not this file's job to second-guess which list a depot belongs on.
OVERLAPPING = ["posterior_SUBQ_whole", "interscapular", "perirenal_proxy", "mediastinal",
               "lumbar_window_VAT", "lumbar_window_SUBQ", "preperitoneal"]


def build_masks(raw, pred, lung_fn=None):
    """The dictionary depot_rules consumes, built the way cohort_derive builds it.

    Returned keys and their meanings are cohort_derive.derive_masks's, not a new invention.
    """
    body, cav = P.compartments(pred)
    air_p1, flo, fhi = P.scanner_band(raw)
    fat = P.fat_mask(raw, body, flo, fhi)

    lung_reg = np.zeros_like(body)
    lung_mL, lung_floor, lung_used = 0.0, float("nan"), "no"
    if lung_fn is not None:
        lung_mL, lung_reg, lung_floor, _ = lung_fn(raw, P.keep_largest3d(body))
        lung_used = "yes"
    # What was actually taken OUT OF THE FAT, which is not the same as the lung region's volume:
    # only about 63% of the lung region falls inside the fat band, so reporting lung_mL as "fat
    # removed" overstates it by half as much again.
    removed = fat & lung_reg
    lung_fat_mL = float(removed.sum()) * VOX_MM3 / 1000.0
    fat = fat & ~lung_reg

    wall_mm3, edge_on_wall = P.wall_quality(pred, body, cav)
    return dict(pred=pred, raw=raw, body=body, cav=cav, cav_raw=(pred == P.CAVITY), fat=fat,
                vat=fat & cav, sat=fat & ~cav, lung_reg=lung_reg,
                air_p1=air_p1, flo=flo, fhi=fhi,
                lung_mL=lung_mL, lung_fat_mL=lung_fat_mL,
                lung_floor=lung_floor, lung_used=lung_used,
                wall_mm3=wall_mm3, edge_on_wall=edge_on_wall)


def analyse(raw, segmenter, landmark_nets, *, lung_fn=None, device="cpu",
            dorsal_is_low=None, depot_rules=None, aid="upload", progress=None,
            on_segmented=None):
    """Full measurement of one volume already on the 0.15 mm isotropic grid.

    `segmenter(z_scored_volume) -> label volume` keeps the nnU-Net call injectable, so the pipeline
    is testable against stored predictions without a GPU or a 410 MB checkpoint.
    """
    def say(msg):
        mem = rss_gb()
        print("[ctadipo] %-38s rss %.2f GB" % (msg, mem), flush=True)
        if progress:
            progress(msg)

    say("finding the animal")
    A = P.animal_mask(raw)

    say("normalising")
    z, mean, sd = P.normalise(raw, A)

    say("segmenting body, wall and cavity")
    pred = segmenter(z)
    # The network is finished with. It is ~1.07 GB of weights, and the single most
    # allocation-heavy step -- the depot partition, which builds seventeen full-resolution boolean
    # volumes at once -- is still to come. Measured on a 540x529x529 scan the partition alone takes
    # the process from 3.5 GB to 7.1 GB, so on an 8 GB worker the weights are the difference
    # between finishing and being killed. The caller decides how to release them, because it owns
    # the cache; the cost is reloading from local disk on the next scan, which is seconds.
    if on_segmented is not None:
        # Drop OUR reference before asking the caller to drop theirs. Python frees an object only
        # when the last name bound to it goes, so clearing a cache while this parameter still holds
        # the predictor frees nothing at all -- which is exactly what the first attempt at this did.
        segmenter = None
        on_segmented()
        gc.collect()
    # The z-scored copy is 604 MB on a normal scan and is dead the instant the network has run.
    # Holding it through the fat band, the lung sweep and the depot partition -- the three stages
    # that allocate most -- is what pushes an 8 GB worker over the edge.
    del z
    gc.collect()

    say("measuring fat")
    M = build_masks(raw, pred, lung_fn=lung_fn)
    # M holds it from here on; the local was a second reference across the whole partition.
    del pred

    say("placing landmarks")
    L = LMM.predict(raw, A, landmark_nets, device=device)
    # The animal mask is finished with. depot_rules takes its animal from keep_largest3d(body),
    # not from this, so it would otherwise sit through the partition for nothing.
    del A
    gc.collect()

    out = {
        "landmarks": L, "normalisation": {"mean": mean, "sd": sd},
        "scanner": {"air_p1": M["air_p1"], "fat_band": (M["flo"], M["fhi"]),
                    "scale": "wide" if M["air_p1"] >= P.SCALE_AIR else "narrow"},
        "quality": quality_flags(M, L),
        "totals": totals(M),
    }

    if depot_rules is not None:
        # frame() reads only body, cav, raw, vat and sat (it discards pred itself). Everything else
        # in the mask set is finished with by now, and each one is a full-resolution boolean volume
        # -- about 450 MB together on a normal scan, freed immediately before the single most
        # allocation-heavy step in the pipeline.
        for dead in ("fat", "cav_raw", "lung_reg"):
            M.pop(dead, None)
        gc.collect()
        say("partitioning into depots")
        # The partition can legitimately fail -- an abdomen-only scan, or landmarks that come out
        # in the wrong order -- and when it does, the totals, the scanner band and the wall QC
        # above are all still correct. Letting the exception escape would throw those away too and
        # hand the user a traceback after a seven-minute wait, so it is caught HERE and reported.
        try:
            out["depots"] = partition(M, L, depot_rules, aid=aid,
                                      dorsal_is_low=dorsal_is_low, say=say)
        except Exception as e:
            out["depots_error"] = str(e)
    return out


def totals(M):
    """Whole-compartment volumes and masses. VAT and SUBQ are fat inside and outside the cavity."""
    def mL(m):
        return float(m.sum()) * VOX_MM3 / 1000.0
    v, s = mL(M["vat"]), mL(M["sat"])
    return {"VAT_mL": v, "SUBQ_mL": s, "TotalFat_mL": v + s,
            "VAT_g": v * DENSITY_G_PER_ML, "SUBQ_g": s * DENSITY_G_PER_ML,
            "TotalFat_g": (v + s) * DENSITY_G_PER_ML,
            "lung_fat_removed_mL": M["lung_fat_mL"],
            "lung_region_mL": M["lung_mL"]}


PARTITION_BYTES_PER_VOXEL = 24.0
"""Peak bytes per voxel of the depot partition, measured not assumed.

rule_A_slabs builds seventeen full-resolution boolean volumes and a distance transform in a single
call. On a 540x529x529 scan (151.1 M voxels) that took the process from 3.5 GB to 7.1 GB.
"""


def _budget_check(shape0, shape1, say=None):
    """Refuse a partition that cannot fit, rather than let the container SIGKILL the session.

    The crop normally removes most of the field -- across forty cohort scans it kept a median of
    22%, worst case 51% -- but that is a property of a mouse lying in a wide bore, not a guarantee.
    A scan that arrives already trimmed tight to the animal cannot be reduced at all, and on an
    8 GB worker the partition then runs the process into the ceiling. What the user sees when that
    happens is the browser disconnecting after half an hour with no error and no result.

    Raising here is caught by analyse(), which keeps the totals, the scanner band, the landmarks
    and the QC flags and reports only the depots as unavailable. A partial answer that explains
    itself beats a dead session.
    """
    n0, n1 = int(np.prod(shape0)), int(np.prod(shape1))
    kept = 100.0 * n1 / max(n0, 1)
    need = n1 * PARTITION_BYTES_PER_VOXEL / 1e9
    now = rss_gb()
    try:
        budget = float(os.environ.get("CTADIPO_MEM_BUDGET_GB") or 6.1)
    except ValueError:
        budget = 16.0
    if say:
        say("cropped %s to %s, keeping %.0f%%" % (_sh(shape0), _sh(shape1), kept))
        say("partition needs ~%.1f GB on top of %.1f GB (budget %.1f)" % (need, now, budget))
    # rss_gb() returns 0.0 where it cannot measure; refusing on that would be refusing blind.
    if now and now + need > budget:
        raise MemoryError(
            "the depot partition needs about %.1f GB on top of the %.1f GB already in use, over "
            "the %.1f GB budget. This scan cropped to %s, %.0f%% of %s -- a volume that arrives "
            "already trimmed to the animal cannot be cropped further. Total fat, VAT, SUBQ, the "
            "landmarks and the QC checks are unaffected and are reported above."
            % (need, now, budget, _sh(shape1), kept, _sh(shape0)))


def _sh(s):
    return "x".join(str(int(x)) for x in s)


def crop_to_animal(M, lm, pad=2):
    """Trim every volume to the animal's bounding box before the depot partition.

    THIS IS THE DIFFERENCE BETWEEN FINISHING AND BEING OOM-KILLED. rule_A_slabs builds seventeen
    full-resolution boolean volumes and a distance transform in one call; measured on a 540x529x529
    scan that takes the process from 3.5 GB to 7.1 GB, and the container log for the first failed
    run on an 8 GB worker says exactly "oom (out of memory)".

    Almost all of that is air. A lean mouse in a wide bore fills about a TENTH of the field --
    540x529x529 crops to 530x159x182 on one real scan -- so the partition's cost falls by ~90%, and
    it runs about ten times faster too (31 s -> 3 s).

    It cannot change a result: every depot is a subset of the animal by construction, and the frame
    is built from relative quantities -- craniocaudal position from the mouse's own z extent, the
    dorsoventral axis from the spine centroid. That is an argument rather than evidence, so it was
    checked: on a real scan all ten depot volumes matched the full-volume partition to 0.00e+00,
    every one.

    IT MUST COPY, AND IT MUST EMPTY THE DICT IT WAS GIVEN. The first version of this function
    did `v[sl]`, which is basic indexing and therefore a VIEW: every "cropped" array kept its
    full-resolution base alive, so the partition iterated over a tenth of the data while the
    process still held all of it. The depot volumes matched and the step ran ten times faster,
    which is precisely why the bug survived review -- correctness and speed both looked right and
    neither one measures bytes. The worker was OOM-killed anyway.

    Popping each key as it is copied means only one full-resolution array is alive at a time, and
    because the dict is the caller's own object it empties analyse()'s M as well; rebinding a
    local would have left a second reference pinning all of it.

    Returns (cropped masks, landmarks shifted into the crop, the z offset). THE MASKS PASSED IN
    ARE CONSUMED.
    """
    mouse = P.keep_largest3d(M["body"])
    # Three boolean reductions rather than argwhere, which would build an int64 index array of
    # about 178 MB for a 7 M-voxel mouse -- at the moment the process is nearest its ceiling.
    lo, hi = [], []
    for ax in range(3):
        nz = np.where(mouse.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        if nz.size == 0:
            return M, lm, 0
        lo.append(max(int(nz[0]) - pad, 0))
        hi.append(min(int(nz[-1]) + 1 + pad, mouse.shape[ax]))
    del mouse
    gc.collect()
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    Mc = {}
    for k in list(M.keys()):
        v = M.pop(k)
        Mc[k] = v[sl].copy() if isinstance(v, np.ndarray) and v.ndim == 3 else v
        del v
    gc.collect()
    z0 = lo[0]
    lmc = {k: (None if v is None else v - z0) for k, v in lm.items()}
    return Mc, lmc, z0


def partition(M, L, DR, aid="upload", dorsal_is_low=None, say=None):
    """The ten depots, via rule_A_slabs against the model's landmarks."""
    lm = {slot: L["z"][name] for slot, name in LMM.DEPOT_ALIAS.items()}
    lm["_kidney_cranial"] = L["z"]["kidney_cranial"]
    # frame() pops pred itself, so carrying it through the crop copies a full int volume for
    # nothing. Dropping it here frees the original a step earlier as well.
    M.pop("pred", None)
    shape0 = M["body"].shape
    M, lm, _z0 = crop_to_animal(M, lm)
    gc.collect()
    trim_heap()
    _budget_check(shape0, M["body"].shape, say)
    F = DR.frame(aid, lm, L["direction"], masks=M, dorsal_is_low=dorsal_is_low)
    d = DR.rule_A_slabs(F)
    v = DR.volumes(d)                                    # mm^3
    ten = {k: v[k] / 1000.0 for k in PARTITION_DEPOTS if k in v}
    return {"mL": ten,
            "g": {k: x * DENSITY_G_PER_ML for k, x in ten.items()},
            "overlapping_mL": {k: v[k] / 1000.0 for k in OVERLAPPING if k in v},
            # Only the ten. The seven overlap masks are never meshed and never read -- their
            # volumes are already in overlapping_mL above -- and the frame is read by nobody at
            # all. Both used to be carried out of here and held through the mesh loop.
            "masks": {k: d[k] for k in PARTITION_DEPOTS if k in d},
            # The animal itself, CROPPED, for the 3-D view only. Ten depots floating in empty space
            # give the viewer nothing to orient against -- no head, no dorsal side, no way to see
            # that VAT sits inside the cavity and SUBQ outside it. This is display-only and is
            # never counted; the app meshes it and drops it immediately.
            "body": M["body"],
            "conservation": DR.check_partition(F, d)}


def quality_flags(M, L):
    """The things that go wrong silently, surfaced rather than hidden.

    Each of these has a documented history of producing a plausible, wrong number. The app shows
    them next to the result; none of them rejects a scan on its own.
    """
    f = {}
    f["wall_mm3"] = M["wall_mm3"]
    f["edge_on_wall"] = M["edge_on_wall"]
    f["cavity_ok"] = M["wall_mm3"] >= P.WALL_MM3_OK
    # VAT and SUBQ are separated by ONE boundary. Where the model has not found the abdominal wall
    # the cavity is unanchored, and the VAT/SUBQ split stops meaning anything while both totals
    # still look entirely reasonable. The 70 mm3 threshold separates the reviewer's own verdicts
    # with no overlap, and it is reproduced here EXACTLY -- `cavity_ok` is the locked criterion and
    # nothing else may be folded into it.
    #
    # This is advisory and deliberately separate. The same review recorded 45-65% of the cavity edge
    # against wall on good scans versus 6-17% on bad ones, but never made that a gate. A lean scan
    # can clear 70 mm3 on wall volume and still sit at 0.29 here, which is worth showing the user --
    # as a second opinion, not as a second threshold that would quietly change what passes.
    # LOOSENED from 0.45. The review recorded 45-65% on good scans and 6-17% on bad ones; 0.45
    # therefore flagged the entire gap between the two populations, so a scan at 0.36 -- nowhere
    # near the bad range -- raised a warning next to a result whose conservation was exact. The
    # threshold belongs just above the BAD range, so it fires for the failure it was built to
    # catch and stays quiet across the gap.
    f["edge_on_wall_low"] = M["edge_on_wall"] < 0.20

    f["direction"] = L["direction"]
    f["direction_margin_mm"] = L["direction_margin_mm"]
    f["direction_uncertain"] = L["direction_uncertain"]
    # A direction flip is the most dangerous failure in the whole chain precisely because it is
    # invisible: it swaps anterior with gluteal and retroperitoneal with perigonadal while
    # conservation still holds exactly and every total is unchanged.

    f["landmarks_ordered"] = L["ordered"]
    # The depot rules require bladder < kidney_caudal < diaphragm < lung_apex along the body. Out of
    # order means the landmarks and the implied direction disagree, and the partition would be
    # guessing which is wrong.

    f["landmark_spread_mm"] = L["spread_mm"]
    # LOOSENED from 3.0 mm. Measured p90 error per plane is 1.54-2.70 mm and a miss is defined
    # elsewhere as >5 mm, so 3.0 sat inside the normal range and flagged three of five planes on a
    # scan that partitioned cleanly. 6 mm is clear of both, so this now means "the model is lost",
    # not "this scan is unfamiliar" -- which, for an app whose whole purpose is scans from other
    # scanners, is the ordinary case.
    #
    # THE COST IS REAL: at 3.0 mm this caught 73% of the >5 mm errors while flagging 10.9% of
    # scans. At 6 mm it catches fewer. That is the deliberate trade -- a panel that cries wolf is
    # one the user stops reading.
    f["landmark_uncertain"] = {k: v > 6.0 for k, v in L["spread_mm"].items()}
    # The model predicts a distribution over z, so a wide one is a genuine "this scan does not look
    # like the training data" signal rather than a confident answer from an unfamiliar image.

    f["lung_subtracted"] = M["lung_used"] == "yes"
    fat_mL = float(M["fat"].sum()) * VOX_MM3 / 1000.0
    f["lung_share_of_fat"] = M["lung_fat_mL"] / max(fat_mL + M["lung_fat_mL"], 1e-9)

    # Almost no fat inside the band. The bands were derived on UNCALIBRATED microCT; a genuinely
    # HU-calibrated scan puts adipose at -190 to -30 HU, which barely overlaps either band, so the
    # app would otherwise report ~0 mL with a clean panel. Also catches an all-zero volume, which
    # passes every other check because 0 > -500 is true everywhere.
    body_mL = float(M["body"].sum()) * VOX_MM3 / 1000.0
    f["fat_fraction_of_body"] = fat_mL / max(body_mL, 1e-9)
    f["band_suspect"] = f["fat_fraction_of_body"] < 0.005
    return f
