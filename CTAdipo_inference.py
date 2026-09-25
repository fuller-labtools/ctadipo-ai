#!/usr/bin/env python
r"""CTAdipo from the command line: scans in, one row of depot volumes each out.

This is the headless equivalent of the Shiny app, and deliberately a THIN one. Every number it
prints comes from `ctadipo.pipeline.analyse`, which is the same function the app calls; the only
things this file owns are argument parsing, batching, and the shape of the CSV. Nothing in the
measurement is re-implemented here, because a second implementation is how a command-line tool
quietly stops agreeing with the app and with the paper -- and the depot rules, the fat band, the
lung subtraction and the landmark features have each already been validated scan by scan.

WHY A CLI AT ALL. The app measures one upload at a time behind a 2 GB upload limit and a single
measurement slot, which is right for a browser and wrong for a cohort. A study directory holds
hundreds of scans that are already on disk; this runs them in one pass, reuses one loaded model
across all of them, and writes a tidy table that goes straight into R or pandas.

WHAT IT MEASURES (all of it locked, see ctadipo/pipeline.py and ctadipo/preprocess.py for why):
    fat        the scan's own band -- [-491,-118] HU when air_p1 < -1050, else [-490,-200] --
               applied to a 3x3x3 median-filtered volume. Against 709 hand-placed fat points the
               median filter keeps 645/709 (91.0%) versus 576/709 (81.2%) on the raw volume.
    VAT/SUBQ   fat inside / outside the nnU-Net cavity.
    ten depots slabs at five model-placed landmark planes. Against expert landmarks on 60 scans
               with identical masks the median per-depot difference is 1.2% (perigonadal) to 8.1%
               (inguinal); total fat, VAT and SUBQ are unchanged at 0.00%, because landmarks only
               redistribute fat between depots.
    mass       volume x 0.9 g/cm3.

TWO THINGS THAT MUST BE RIGHT BEFORE THE NUMBERS MEAN ANYTHING, and neither is silently assumed.
VOXEL SIZE is read from the file where the format carries it and otherwise must be given
with --voxel; a wrong value produces an entirely plausible table that is wrong by the cube of the
error. CRANIOCAUDAL DIRECTION is not asked for -- it is implied by the landmarks (lung apex
relative to bladder), which agrees with the reviewed table on 1182/1186 scans -- but its margin is
written to every row, because a flip swaps anterior with gluteal and retroperitoneal with
perigonadal while every total stays exactly correct.

USAGE
    python CTAdipo_inference.py scan.nii.gz --out depots.csv
    python CTAdipo_inference.py "scans/*.nii.gz" --out depots.csv --fast
    python CTAdipo_inference.py dicom_folder/ --out depots.csv
    python CTAdipo_inference.py stack.tif --voxel 0.5 0.1 0.1 --out depots.csv   # no spacing in
                                                                                 # the file

MODELS. The 410 MB segmentation checkpoint is downloaded from Zenodo on first use and cached
(sha256-verified; see ctadipo/models.py). Point CTADIPO_NNUNET at a folder that already holds it
to skip the download. The 3.5 MB-per-fold landmark models ship with the repository.

RUNTIME. The network dominates, and everything around it is a fixed cost that does not shrink with
--fast. Timed stage by stage on one 540x529x529 scan, 32 threads, mirroring off: 166 s in total, of
which segmentation 105 s, the fat band and lung sweep 22 s, the depot partition 24 s, reading and
normalising 8 s, landmarks 2 s. The same scan with mirroring on takes about 13 min: the 8x falls
entirely on that 105 s. Scale by voxels for another scan, and expect fewer cores to move the 105 s
and very little else -- a quoted segmentation time is not a per-scan time.
--fast disables the eight-fold mirroring the published numbers were computed with, so it is for
triage, not for the paper.

VALIDATION. Run at the published configuration on a 540x529x529 whole-body 12-month scan that is
already in the reference cohort table, this script returns total fat 1.0643 mL, VAT 0.2502 mL,
SUBQ 0.8141 mL against that table's 1.0641 / 0.2502 / 0.8139 -- +0.02% on total fat and 0.00% on
VAT -- with wall_mm3 (128.2) and edge_on_wall (0.292) identical to three figures, which is the
segmentation itself agreeing rather than two errors cancelling. Re-running with --threads 16
instead of the machine's 32 gave identical numbers, so the result is not thread-order dependent.
The same scan measured with --fast returns 1.0837 mL, +1.8%, because mirroring moves the body-mask
rim and a lean animal's fat is mostly rim.

    Which table. The comparison is against the CURRENT derivation, the one whose fat band is
    applied to the 3x3x3 median-filtered volume. A superseded derivation of the same cohort, made
    before that change, reads 1.1125 mL for this scan -- 4.4% higher. The two agree on the cavity
    volume for all 1186 scans and use the same band, so the difference is the median filter alone;
    across the cohort it moves total fat by a median of -1.6%, -27% in the leanest quartile (grain
    inside the band, removed) and +11% in the fattest (real fat, recovered). If a number from this
    script is ~4% under a remembered one, check which derivation the remembered one came from.

Part of CTAdipo -- https://github.com/fuller-labtools/ctadipo-ai
"""
from __future__ import annotations

import argparse
import csv
import glob as globmod
import os
import sys
import time
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

import numpy as np                                                            # noqa: E402

from ctadipo import io_any, pipeline, landmark_model as LMM                   # noqa: E402

__version__ = "1.0.0"

# ------------------------------------------------------------------ configuration
# The landmark folds ship in the repository; the nnU-Net checkpoint does not (410 MB). Both
# locations are overridable by environment variable, exactly as in app.py, so the CLI and the app
# can be pointed at the same cache and neither downloads twice.
LANDMARK_DIR = Path(os.environ.get("CTADIPO_LANDMARKS", APP_DIR / "model"))
NNUNET_RESULTS = os.environ.get("CTADIPO_NNUNET") or None

DENSITY = pipeline.DENSITY_G_PER_ML          # 0.9 g/cm3, adipose tissue
ISO = 0.15                                   # mm; the grid the whole pipeline is defined on

# Memory ceiling, in the units app.py measured: about 40 bytes of working set per output voxel
# through the whole chain, dominated by the depot partition. The REASON for a ceiling is app.py's
# too -- ndi.zoom allocates the whole output up front, so a clinical CT at 3 mm slices asks for a
# terabyte-scale array and the process dies with nothing in the log, and refusing early while
# naming the voxel size is the difference between a bad argument and a dead batch. The BUDGET,
# though, is deliberately not the app's: a 6.1 GB web worker is what the hosted app has, and
# "run it locally, where memory is not capped" is what the app tells a user with a big scan to do,
# which is this tool. 16 GB of working set is a workstation's answer; lower it with
# CTADIPO_MEM_BUDGET_GB (the same variable app.py reads) on a smaller machine.
BYTES_PER_VOXEL = 40
MEM_BUDGET_GB = float(os.environ.get("CTADIPO_MEM_BUDGET_GB", "16"))
MAX_OUTPUT_VOXELS = int(MEM_BUDGET_GB * 1e9 / BYTES_PER_VOXEL)   # 400 M voxels at the default
MIN_Z_AFTER_RESAMPLE = 128                   # fewer slices cannot carry five landmark planes
VOX_MIN, VOX_MAX = 0.005, 1.0                # mm

VOLUME_EXT = (".nii", ".nii.gz", ".tif", ".tiff", ".npy", ".npz", ".h5", ".hdf5", ".he5", ".zip")

# Column order is fixed and a failed scan still writes a full row of blanks. A batch table whose
# columns depend on which scans happened to succeed cannot be rbind-ed with the next batch's.
QUALITY_COLUMNS = [
    "wall_mm3", "cavity_ok", "edge_on_wall",
    "direction", "direction_margin_mm", "direction_uncertain",
    "landmarks_ordered", "landmark_spread_max_mm",
    "lung_subtracted", "lung_fat_removed_mL", "lung_share_of_fat",
    "air_p1_HU", "fat_lo_HU", "fat_hi_HU", "fat_fraction_of_body", "band_suspect",
    "partition_conserved",
]
COLUMNS = (
    ["scan", "path", "status", "error",
     "shape_z", "shape_y", "shape_x",
     "voxel_z_mm", "voxel_y_mm", "voxel_x_mm", "spacing_source", "hu_calibrated",
     "TotalFat_mL", "VAT_mL", "SUBQ_mL", "TotalFat_g", "VAT_g", "SUBQ_g"]
    + ["%s_mL" % d for d in pipeline.PARTITION_DEPOTS]
    + ["%s_g" % d for d in pipeline.PARTITION_DEPOTS]
    + QUALITY_COLUMNS
    + ["mirroring", "seconds", "ctadipo_version"]
)


# ------------------------------------------------------------------ inputs
def expand_inputs(patterns):
    """Turn the command line into a list of SCANS, where one scan may be a folder of slices.

    The ambiguity worth naming: a directory is a DICOM series (io_any sniffs it as one), but a
    directory of NIfTI files is a batch of scans, and treating it as a series would hand the DICOM
    reader a folder with no DICOM in it and report "no DICOM series found" for a format the user
    never mentioned. So a directory holding recognised volume files and no DICOM is expanded into
    its files; anything else is passed through whole and io_any decides.

    Globs are expanded here rather than by the shell because this is a Windows-first tool and
    neither cmd.exe nor PowerShell expands a quoted wildcard the way a POSIX shell does.

    The same scan named twice -- a directory and a glob that overlaps it, which is an easy thing
    to type -- is measured ONCE. A batch table with duplicate rows does not fail, it quietly
    double-weights those animals in whatever is joined to it next.
    """
    out = []
    for pat in patterns:
        if any(c in pat for c in "*?["):
            hits = sorted(globmod.glob(pat, recursive=True))
            if not hits:
                raise SystemExit("no files match %s" % pat)
            out.extend(Path(h) for h in hits)
            continue
        p = Path(pat)
        if not p.exists():
            raise SystemExit("no such file or folder: %s" % p)
        if p.is_dir():
            files = sorted(c for c in p.iterdir() if c.is_file())
            vols = [c for c in files if c.name.lower().endswith(VOLUME_EXT)]
            dicom = [c for c in files if _looks_dicom(c)]
            out.extend(vols if (vols and not dicom) else [p])
        else:
            out.append(p)
    seen, unique = set(), []
    for p in out:
        key = os.path.normcase(str(p.resolve()))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


def _looks_dicom(p: Path) -> bool:
    """DICOM by magic at byte 128, because DICOM slices are routinely written with no extension."""
    if p.name.lower().endswith((".dcm", ".dicom", ".ima")):
        return True
    try:
        with open(p, "rb") as f:
            return f.read(132)[128:132] == b"DICM"
    except Exception:
        return False


def resolve_spacing(sp, meta, voxel, name, warn):
    """The spacing to resample with: the file's if it has one, otherwise --voxel, never a guess.

    --voxel does NOT override a file that carries its own spacing, which is io_any's rule and the
    safer default -- but a user who passes one and is ignored deserves to be told, so a
    disagreement is reported rather than absorbed.
    """
    if sp is not None:
        if voxel is not None and not np.allclose(np.asarray(sp, float), voxel, rtol=1e-3):
            warn("--voxel %s ignored: %s carries its own voxel size %s (%s)"
                 % (_fmt_sp(voxel), name, _fmt_sp(sp), meta.get("spacing_source")))
        return tuple(float(x) for x in sp)
    if voxel is None:
        raise ValueError(
            "this file carries no voxel size (%s), so it must be given with "
            "--voxel z y x (mm). It cannot be guessed: everything downstream is defined on a "
            "0.15 mm grid and a wrong value is wrong by the cube of the error."
            % meta.get("kind", "unknown format"))
    return tuple(float(x) for x in voxel)


def _fmt_sp(sp):
    return "%.4f x %.4f x %.4f mm" % tuple(sp)


# ------------------------------------------------------------------ models
def load_landmark_nets(device="cpu"):
    """The five landmark folds, once for the whole batch.

    Averaged in PROBABILITY space by LMM.predict, not by averaging five scalar answers, so two
    folds that disagree about which of two candidate planes is right give a visibly bimodal,
    low-confidence result instead of a confident one halfway between them. Out of fold on 1186
    scans, grouped by animal, the median absolute error runs from 0.48 mm (diaphragm) to 0.84 mm
    (bladder); the per-plane table is in the README.
    """
    if not list(LANDMARK_DIR.glob("fold*.pt")):
        raise SystemExit(
            "no landmark model in %s (expected fold*.pt). These are 3.5 MB each and ship with the "
            "repository, so this means the checkout is incomplete; set CTADIPO_LANDMARKS to point "
            "at them." % LANDMARK_DIR)
    return LMM.load(LANDMARK_DIR, device=device)


def load_segmenter(mirroring=True, device="cpu", threads=None, progress=print):
    """The nnU-Net segmenter, loaded ONCE and reused by every scan in the batch.

    Re-initialising the predictor per scan would re-read 410 MB of weights each time; over a
    cohort that is the dominant cost. That is the right trade on a workstation and the wrong one on
    a small machine: the weights are ~1.07 GB resident while the depot partition, the heaviest step,
    is still to come, so on roughly 8 GB of RAM pass `on_segmented` to pipeline.analyse (it drops
    the predictor before the partition) and reload here per scan. This tool does not, on purpose.
    The configuration is not a choice -- 3d_lowres, ResEnc-M
    plans, nnUNetTrainer_100epochs, fold 0, checkpoint_final.pth -- because it is what produced the
    locked cohort values this tool has to reproduce.
    """
    from ctadipo import segment, models
    root = models.ensure_checkpoint(progress=lambda f, m: progress("  " + m),
                                    results_root=NNUNET_RESULTS)
    return segment.load_segmenter(root.parent.parent, device=device,
                                  mirroring=mirroring, threads=threads)


def load_lung_fn():
    """(lung_fn, note). The lung sweep pinned to the configuration the cohort was reviewed under.

    Lung reads -300 to -400 HU, which is inside the fat band, so unsubtracted lung is COUNTED AS
    FAT: measured at ~16% of a lean animal's total fat against ~1.5% of an obese one's, correlating
    with log total fat at r = -0.83. Skipping it is therefore not a neutral saving, it is a bias
    along the adiposity axis itself -- so its absence is returned as a note and written to the CSV
    rather than passed over.
    """
    try:
        from ctadipo.vendor.lung import lung_air
    except Exception as e:
        return None, "lung module not importable (%s); lung will NOT be removed from fat" % e

    def fn(raw, body):
        return lung_air(raw, body, select="peak", coarse=True)
    return fn, ""


def load_depot_rules():
    """The vendored depot definitions, imported rather than reimplemented.

    These are the ten definitions validated scan by scan, together with the dorsoventral frame, the
    spine midline and the conservation check. CTADIPO_DEPOT_RULES overrides with a research-tree
    copy for development. A failure is returned, not swallowed: silently degrading to totals-only
    would hand back a table whose depot columns are simply absent and look like a scan that has no
    fat there.
    """
    override = os.environ.get("CTADIPO_DEPOT_RULES", "")
    if override and Path(override).exists():
        sys.path.insert(0, str(Path(override).parent))
        try:
            import depot_rules
            return depot_rules, ""
        except Exception as e:
            return None, "CTADIPO_DEPOT_RULES set but not importable: %s" % e
    try:
        from ctadipo.vendor import depot_rules
        return depot_rules, ""
    except Exception as e:
        return None, "the vendored depot rules failed to import: %s" % e


# ------------------------------------------------------------------ one scan
def measure_one(path, nets, segmenter, depot_rules, lung_fn, device="cpu",
                voxel=None, dorsal_is_low=None, say=print):
    """Measure one volume, returning what row_from needs. Raises if it cannot be measured.

    The masks are dropped before returning, on purpose. rule_A_slabs hands back 17 full-resolution
    boolean volumes and the frame still holds raw/body/cav/vat/sat (analyse drops fat, cav_raw and
    lung_reg before building it) -- about 4 GB retained on a normal scan, which in a batch is not a
    leak that shows up later but the second scan dying.
    """
    p = Path(path)
    name = p.name if p.is_file() else p.name or str(p)
    aid = p.name[:-7] if p.name.lower().endswith(".nii.gz") else p.stem

    vol, sp, meta = io_any.read_any(p, original_name=(name if p.is_file() else None))
    spacing = resolve_spacing(sp, meta, voxel, name, warn=lambda m: say("  WARNING: " + m))
    for v in spacing:
        if not (VOX_MIN <= v <= VOX_MAX):
            raise ValueError("voxel size %.4f mm is outside the accepted range %.3f-%.1f mm"
                             % (v, VOX_MIN, VOX_MAX))
    say("  read %s as %s, %d x %d x %d voxels at %s (%s)"
        % (name, meta.get("kind"), *meta["shape"], _fmt_sp(spacing), meta.get("spacing_source")))

    say("  resampling to the %.2f mm grid" % ISO)
    raw = io_any.to_isotropic(vol, spacing, iso=ISO, max_voxels=MAX_OUTPUT_VOXELS)
    n_vox = int(np.prod(raw.shape))
    if n_vox > MAX_OUTPUT_VOXELS:
        # to_isotropic's own guard covers the RESAMPLE, and it returns the array untouched when the
        # file is already on the 0.15 mm grid -- so an already-isotropic oversized volume reaches
        # here having been checked by nothing. app.py repeats the count for the same reason.
        raise ValueError(
            "this scan is %s = %.0f million voxels at %.2f mm, which needs roughly %.1f GB to "
            "measure against a %.1f GB budget, so it would be killed part-way through rather than "
            "finishing. Crop it to the torso, or raise CTADIPO_MEM_BUDGET_GB if the machine has "
            "the memory."
            % ("x".join(str(int(x)) for x in raw.shape), n_vox / 1e6, ISO,
               n_vox * BYTES_PER_VOXEL / 1e9, MEM_BUDGET_GB))
    if raw.shape[0] < MIN_Z_AFTER_RESAMPLE:
        raise ValueError(
            "after resampling to %.2f mm this volume is only %d slices (%.1f mm), too short to "
            "carry the five anatomical planes the depots are defined on. Check the voxel size."
            % (ISO, raw.shape[0], raw.shape[0] * ISO))

    out = pipeline.analyse(raw, segmenter, nets, lung_fn=lung_fn, device=device,
                           dorsal_is_low=dorsal_is_low, depot_rules=depot_rules, aid=aid,
                           progress=lambda m: say("  " + m))
    if "depots" in out:
        out["depots"].pop("frame", None)
        out["depots"].pop("masks", None)
    return out, meta, spacing, raw.shape


def row_from(out, meta, spacing, shape, name, path, mirroring, seconds):
    """One tidy row. Every quality flag that has ever produced a plausible wrong number is in it.

    `cavity_ok` is the locked criterion and nothing else is folded into it: VAT and SUBQ are
    separated by ONE boundary, and against the reviewer's verdicts the abdominal-wall volume
    separates good from bad with no overlap (72-1174 mm3 against 28-67 mm3), so 70 mm3 is the gate.
    `direction_margin_mm` is there because the four direction errors in 1186 scans had an
    apex-to-bladder separation of 0.4-25.1 mm against a 1st percentile of 29.4 mm when the call was
    right -- the separation IS the confidence.
    """
    r = blank_row(name, path)
    t, q = out["totals"], out["quality"]
    sc = out["scanner"]
    d = out.get("depots")

    r.update(status="ok", error=_oneline(out.get("depots_error", "")),
             shape_z=shape[0], shape_y=shape[1], shape_x=shape[2],
             voxel_z_mm=round(spacing[0], 5), voxel_y_mm=round(spacing[1], 5),
             voxel_x_mm=round(spacing[2], 5),
             spacing_source=meta.get("spacing_source", ""),
             hu_calibrated=bool(meta.get("hu_calibrated", False)),
             mirroring=bool(mirroring), seconds=round(seconds, 1),
             ctadipo_version=__version__)

    for k in ("TotalFat_mL", "VAT_mL", "SUBQ_mL", "TotalFat_g", "VAT_g", "SUBQ_g"):
        r[k] = round(float(t[k]), 4)
    if d is not None:
        for k in pipeline.PARTITION_DEPOTS:
            if k in d["mL"]:
                r["%s_mL" % k] = round(float(d["mL"][k]), 4)
                r["%s_g" % k] = round(float(d["g"][k]), 4)

    r.update(wall_mm3=round(float(q["wall_mm3"]), 1),
             cavity_ok=bool(q["cavity_ok"]),
             edge_on_wall=round(float(q["edge_on_wall"]), 3),
             direction=q["direction"],
             direction_margin_mm=round(float(q["direction_margin_mm"]), 2),
             direction_uncertain=bool(q["direction_uncertain"]),
             landmarks_ordered=bool(q["landmarks_ordered"]),
             landmark_spread_max_mm=round(float(max(q["landmark_spread_mm"].values())), 2),
             lung_subtracted=bool(q["lung_subtracted"]),
             lung_fat_removed_mL=round(float(t["lung_fat_removed_mL"]), 4),
             lung_share_of_fat=round(float(q["lung_share_of_fat"]), 4),
             air_p1_HU=round(float(sc["air_p1"]), 1),
             fat_lo_HU=sc["fat_band"][0], fat_hi_HU=sc["fat_band"][1],
             fat_fraction_of_body=round(float(q["fat_fraction_of_body"]), 4),
             band_suspect=bool(q["band_suspect"]),
             partition_conserved=_conserved(d))
    return r


def _conserved(d):
    """Did the ten depots tile VAT and SUBQ exactly? The partition's own correctness test.

    check_partition counts voxels that no depot claimed, voxels claimed outside the compartment,
    and voxels claimed twice. Anything other than zero on all three means the table's depot columns
    do not add up to its VAT and SUBQ columns, which is worth one boolean in every row.
    """
    if d is None:
        return ""
    c = d.get("conservation") or {}
    try:
        return all(int(v[k]) == 0 for v in c.values() for k in ("missing", "extra", "overlap"))
    except Exception:
        return ""


def blank_row(name, path):
    return dict({k: "" for k in COLUMNS}, scan=name, path=str(path), status="failed")


def _oneline(msg):
    """Errors onto one line. SimpleITK's messages carry newlines, and csv quotes them faithfully --
    which is correct and still makes a table nobody can read in a terminal or a text editor."""
    return " ".join(str(msg).split())


# ------------------------------------------------------------------ output
def print_summary(row):
    """The per-scan block on stdout. The flags sit next to the numbers, not in a footnote."""
    if row["status"] != "ok":
        print("  FAILED: %s" % row["error"])
        return
    tot = row["TotalFat_mL"] or 0.0
    pct = (lambda v: 100.0 * v / tot if tot else 0.0)
    print("  total fat  %8.3f mL  %8.3f g" % (row["TotalFat_mL"], row["TotalFat_g"]))
    print("  VAT        %8.3f mL  %8.3f g   (%.1f%% of fat)"
          % (row["VAT_mL"], row["VAT_g"], pct(row["VAT_mL"])))
    print("  SUBQ       %8.3f mL  %8.3f g   (%.1f%% of fat)"
          % (row["SUBQ_mL"], row["SUBQ_g"], pct(row["SUBQ_mL"])))
    have = [d for d in pipeline.PARTITION_DEPOTS if row["%s_mL" % d] != ""]
    if have:
        print("  depots (mL):")
        for d in have:
            print("    %-18s %7.3f   (%4.1f%%)" % (d, row["%s_mL" % d], pct(row["%s_mL" % d])))
    elif row["error"]:
        print("  depots: not partitioned -- %s" % row["error"])
    flags = [
        "wall %.0f mm3 %s" % (row["wall_mm3"], "ok" if row["cavity_ok"] else "LOW (VAT/SUBQ split untrustworthy)"),
        "direction %s (%.0f mm margin)%s" % (row["direction"], row["direction_margin_mm"],
                                             " UNCERTAIN" if row["direction_uncertain"] else ""),
        "landmarks %s" % ("ordered" if row["landmarks_ordered"] else "OUT OF ORDER"),
        "lung %s" % ("-%.3f mL (%.1f%% of fat)" % (row["lung_fat_removed_mL"],
                                                   100 * row["lung_share_of_fat"])
                     if row["lung_subtracted"] else "NOT SUBTRACTED"),
    ]
    if row["band_suspect"]:
        flags.append("BAND SUSPECT (almost no fat in band -- is this HU-calibrated?)")
    if row["partition_conserved"] is False:
        flags.append("PARTITION DOES NOT CONSERVE")
    print("  flags: " + " | ".join(flags))
    print("  %.0f s" % row["seconds"])


def write_csv(rows, out_path):
    """Rewritten after every scan, so a batch interrupted at hour three still has hours one and two.

    A 400-scan run is many hours of CPU; buffering the table until the end means a power cut, a
    full disk or a Ctrl-C throws all of it away. Rewriting a few hundred rows costs milliseconds.
    """
    tmp = Path(str(out_path) + ".part")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    os.replace(tmp, out_path)


# ------------------------------------------------------------------ CLI
def build_parser():
    ap = argparse.ArgumentParser(
        prog="CTAdipo_inference.py",
        description="Measure adipose depots in microCT volumes. Headless CTAdipo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Inputs may be files, folders of DICOM slices, or quoted globs.\n"
               "Models: the 410 MB nnU-Net checkpoint is fetched from Zenodo on first use and\n"
               "cached; set CTADIPO_NNUNET to a folder that already holds it to skip that.")
    ap.add_argument("inputs", nargs="+",
                    help="scan file(s), a DICOM folder, or a glob such as \"scans/*.nii.gz\"")
    ap.add_argument("--out", default="ctadipo_depots.csv", help="output CSV (default: %(default)s)")
    ap.add_argument("--voxel", nargs=3, type=float, metavar=("Z", "Y", "X"),
                    help="voxel size in mm, needed only for formats that carry none (.npy, most "
                         "TIFF). Ignored where the file supplies its own.")
    ap.add_argument("--fast", action="store_true",
                    help="disable eight-fold mirroring: roughly 4x faster end to end (about 3 min "
                         "against 13 on a 540x529x529 scan) and NOT the setting the published "
                         "numbers were computed with. Measured on one lean animal it moved total "
                         "fat by +1.8%%, so it is for triage, not for the paper")
    ap.add_argument("--no-lung", action="store_true",
                    help="skip lung subtraction. Lung sits inside the fat band, so this inflates "
                         "lean animals specifically (~16%% of their fat against ~1.5%% of an obese "
                         "animal's); use only for triage")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads for inference. Unset, the order is the cgroup quota where "
                         "there is one, then the process affinity, then os.cpu_count() -- so on "
                         "Windows or a bare host it takes every core, which is worth capping by "
                         "hand if anything else is running")
    ap.add_argument("--device", default="cpu", help="torch device, cpu or cuda (default: cpu)")
    ap.add_argument("--dorsal", choices=("auto", "low", "high"), default="auto",
                    help="override the dorsoventral verdict if the automatic one is wrong "
                         "(default: auto, which refuses rather than guessing when unsure)")
    ap.add_argument("--version", action="version", version="CTAdipo %s" % __version__)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.threads:
        # segment.load_segmenter reads this too, and sets torch's thread count from it. Note that
        # os.cpu_count() is the HOST's core count, not this process's allowance, which is why the
        # default is computed rather than assumed.
        os.environ["CTADIPO_THREADS"] = str(max(1, args.threads))

    scans = expand_inputs(args.inputs)
    print("CTAdipo %s -- %d scan%s, %s, mirroring %s"
          % (__version__, len(scans), "" if len(scans) == 1 else "s",
             args.device, "off (--fast)" if args.fast else "on"))

    depot_rules, dr_note = load_depot_rules()
    if dr_note:
        print("WARNING: %s\n         totals will be reported, depots will not." % dr_note)
    lung_fn, lung_note = ((None, "lung subtraction is OFF (--no-lung): lung sits inside the fat "
                                 "band, so it will be counted as fat")
                          if args.no_lung else load_lung_fn())
    if lung_note:
        print("WARNING: %s" % lung_note)

    # Models are loaded BEFORE the first scan is read: a missing checkpoint or an incomplete
    # checkout is a batch-wide failure, and finding that out after the last scan of an overnight
    # run rather than in the first ten seconds is the difference between a retry and a lost night.
    print("loading models")
    nets = load_landmark_nets(device=args.device)
    segmenter = load_segmenter(mirroring=not args.fast, device=args.device,
                               threads=args.threads, progress=print)

    dorsal_is_low = None if args.dorsal == "auto" else (args.dorsal == "low")
    rows, failures = [], []
    t_batch = time.time()
    for i, p in enumerate(scans, 1):
        name = p.name
        print("\n[%d/%d] %s" % (i, len(scans), p))
        t0 = time.time()
        try:
            out, meta, spacing, shape = measure_one(
                p, nets, segmenter, depot_rules, lung_fn, device=args.device,
                voxel=args.voxel, dorsal_is_low=dorsal_is_low, say=print)
            row = row_from(out, meta, spacing, shape, name, p,
                           mirroring=not args.fast, seconds=time.time() - t0)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            # One bad scan must not end the batch -- a cohort always holds an abdomen-only scan or
            # a truncated export, and the other 399 are still worth measuring. The message goes in
            # the row so the CSV itself says which scans are missing and why; the traceback goes to
            # stderr for anything that needs debugging.
            row = blank_row(name, p)
            row["error"] = _oneline("%s: %s" % (type(e).__name__, e))
            row["seconds"] = round(time.time() - t0, 1)
            # A failed row still says WHICH RUN failed it. Without these two the settings columns
            # are blank on exactly the rows someone re-runs, and a concatenated table cannot tell a
            # scan that failed under --fast from one that failed under the published configuration.
            row["mirroring"] = bool(not args.fast)
            row["ctadipo_version"] = __version__
            failures.append((str(p), row["error"]))
            traceback.print_exc(file=sys.stderr)
        rows.append(row)
        print_summary(row)
        write_csv(rows, args.out)

    ok = sum(1 for r in rows if r["status"] == "ok")
    print("\n%d/%d measured in %.1f min -> %s"
          % (ok, len(rows), (time.time() - t_batch) / 60.0, Path(args.out).resolve()))
    if failures:
        print("\n%d scan%s failed:" % (len(failures), "" if len(failures) == 1 else "s"))
        for path, err in failures:
            print("  %s\n    %s" % (path, err))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
