r"""nnU-Net inference, in process, on CPU.

THE CONFIGURATION IS NOT A CHOICE -- it is whatever produced the locked cohort, because CTAdipo has
to reproduce _doz_long_v3.csv on scans that are already derived. Production ran:

    nnUNetv2_predict -d 112 -c 3d_lowres -f 0 -p nnUNetResEncUNetMPlans
                     -tr nnUNetTrainer_100epochs -chk checkpoint_final.pth

with no --disable_tta, so eight-fold mirroring was ON, and with a SINGLE fold. Both matter:

  ONE FOLD means the app ships one 410 MB checkpoint rather than five. Averaging five folds would
  be a better model and would NOT reproduce the cohort, so it is not an option here.

  MIRRORING costs 8x. The 3d_lowres configuration takes a 540x529x529 input down to 204x199x199 at
  0.398 mm, which is 27 patches; on eight CPU threads that is roughly 50 s without mirroring and
  about 7 min with it. `mirroring` is exposed so the difference can be MEASURED against the locked
  volumes rather than assumed to be negligible -- but it defaults to on, because on is what the
  numbers in the paper were computed with.

THE INPUT MUST ALREADY BE Z-SCORED. dataset.json declares "noNorm", so nnU-Net applies no
normalisation of its own; preprocess.normalise is the normalisation. Passing a raw volume here
produces a confident, wrong segmentation.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DATASET = "Dataset112_MouseAdiposeWallHU"
TRAINER = "nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__3d_lowres"
CHECKPOINT = "checkpoint_final.pth"
FOLD = 0
ISO = 0.15


def model_folder(results_root):
    return Path(results_root) / DATASET / TRAINER


def _usable_cpus() -> int:
    """How many cores this process may actually use, not how many the machine has.

    os.cpu_count() reports the HOST's cores. Inside a cgroup-limited container -- which is what both
    deployment targets are -- that is wildly optimistic: it would start 8 or 16 OMP threads on a plan
    allocating one or two, and the resulting context thrashing plus per-thread allocator arenas makes
    it slower than running single-threaded, not faster. The cgroup quota is the truth when it exists;
    sched_getaffinity is the truth on Linux when it does not; cpu_count is the last resort.
    """
    env = os.environ.get("CTADIPO_THREADS")
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    for quota_f, period_f in (("/sys/fs/cgroup/cpu.max", None),
                              ("/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
                               "/sys/fs/cgroup/cpu/cpu.cfs_period_us")):
        try:
            if period_f is None:                       # cgroup v2: "<quota> <period>" or "max ..."
                q, p = Path(quota_f).read_text().split()
                if q == "max":
                    continue
                q, p = int(q), int(p)
            else:                                      # cgroup v1
                q, p = int(Path(quota_f).read_text()), int(Path(period_f).read_text())
            if q > 0 and p > 0:
                return max(1, int(q // p))
        except Exception:
            continue
    try:
        return max(1, len(os.sched_getaffinity(0)))    # Linux only
    except AttributeError:
        return max(1, os.cpu_count() or 2)


def load_segmenter(results_root, device="cpu", mirroring=True, threads=None, step=0.5):
    """Return `segmenter(z_scored_volume) -> uint8 label volume`.

    Returning a closure rather than a predictor object keeps the pipeline injectable: the tests
    swap in a function that reads a stored prediction, so the whole chain can be checked without a
    checkpoint or a GPU.
    """
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    auto = threads is None
    if auto:
        threads = _usable_cpus()
    usable = max(1, int(threads))
    # RESERVE ONE CORE FOR THE EVENT LOOP.
    #
    # nnU-Net saturates every thread it is given, for twelve minutes on this worker, and Shiny's
    # asyncio loop runs in the SAME process. With no core left to schedule it the server cannot
    # deliver a single output: the page freezes for the whole run -- no progress bar, no step text,
    # not even the Run button re-rendering. That is what "I see no bar" was.
    #
    # Only when the count was auto-detected: an explicit threads= is the caller's decision. Set
    # CTADIPO_RESERVE_CORE=0 to take the core back and accept a frozen page.
    reserve = os.environ.get("CTADIPO_RESERVE_CORE", "1") not in ("0", "false", "no")
    if auto and reserve and usable >= 2:
        threads = usable - 1
    else:
        threads = usable
    print("[ctadipo] cpus=%d torch_threads=%d%s"
          % (usable, threads, "  (one reserved for the UI)" if threads < usable else ""),
          flush=True)
    torch.set_num_threads(max(1, int(threads)))
    dev = torch.device(device)
    p = nnUNetPredictor(tile_step_size=step, use_gaussian=True, use_mirroring=bool(mirroring),
                        perform_everything_on_device=(dev.type == "cuda"), device=dev,
                        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    folder = model_folder(results_root)
    if not folder.exists():
        raise FileNotFoundError(
            "no trained model at %s -- the checkpoint is ~410 MB and is not bundled with the app; "
            "see DEPLOY.md for where it is fetched from" % folder)
    p.initialize_from_trained_model_folder(str(folder), use_folds=(FOLD,), checkpoint_name=CHECKPOINT)

    def segmenter(z_scored):
        # nnU-Net takes (channels, Z, Y, X) with a matching spacing, and returns (Z, Y, X).
        arr = np.ascontiguousarray(z_scored[None].astype(np.float32))
        seg = p.predict_single_npy_array(arr, {"spacing": (ISO, ISO, ISO)}, None, None, False)
        return np.asarray(seg).astype(np.uint8)

    return segmenter


def stored_segmenter(pred_dir):
    """A segmenter that reads an already-computed prediction, keyed by animal id.

    This is what the regression tests use. It removes the network from the comparison entirely, so
    a difference against the locked cohort can only come from the steps CTAdipo actually changed.
    """
    import SimpleITK as sitk

    def make(aid):
        def segmenter(_z):
            p = Path(pred_dir) / ("%s.nii.gz" % aid)
            if not p.exists():
                raise FileNotFoundError(p)
            return sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.uint8)
        return segmenter
    return make
