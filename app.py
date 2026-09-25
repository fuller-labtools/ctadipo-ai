# app.py
# CTAdipo - Shiny for Python app that measures adipose depots in mice from a microCT volume.
# One upload in, total VAT/SUBQ and ten sub-depots out, in mL and in grams, with 3-D renders.
#
# Everything the original pipeline took from a human-reviewed table is supplied by the app:
# the five landmark planes and the craniocaudal direction come from the landmark model, the
# dorsoventral axis from geometry with a manual override, and the lung from the sweep. The
# measurement itself is the locked depot code, vendored and called unchanged.
#
# Run:
#   pip install -r requirements.txt
#   shiny run --reload app.py

from __future__ import annotations

import asyncio
import gc
import os
import sys
import threading
import traceback
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import App, ui, render, reactive, req

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

from ctadipo import io_any, pipeline, landmark_model as LMM, render as R3  # noqa: E402

# -----------------------------
# Configuration
# -----------------------------
MODEL_DIR = Path(os.environ.get("CTADIPO_LANDMARKS", APP_DIR / "model"))
# Set CTADIPO_NNUNET only to point at a folder that ALREADY holds the checkpoint. Left unset,
# ctadipo.models assembles the model folder in a writable cache and fetches the checkpoint on
# first use -- which is the deployment path, since no host here offers a persistent volume.
NNUNET_RESULTS_ENV = os.environ.get("CTADIPO_NNUNET") or None
DENSITY = pipeline.DENSITY_G_PER_ML
CSV_NAME = "ctadipo_depots.csv"

# Server-side limits. The HTML min/max on a numeric input is advisory only -- the browser sends
# whatever it likes over the websocket -- so every one of these is enforced here as well.
MAX_UPLOAD_MB = 2048
VOX_MIN, VOX_MAX = 0.005, 1.0
# MEASURED, not guessed. Profiling one 540x529x529 scan (151 M voxels) through the whole chain on
# this hardware gave a peak of 6.37 GB, of which 0.4 GB is the interpreter and the imports -- so the
# work costs about 40 bytes per voxel, dominated by the depot partition, which alone adds 3.3 GB
# because rule_A_slabs builds seventeen full-resolution boolean volumes and a distance transform at
# once. On an 8 GB worker, reserving 1.5 GB for Python, Shiny and the websocket leaves ~6.1 GB of
# work, i.e. about 155 M voxels. The container log for the first failed run says "oom (out of
# memory)", so this ceiling is the difference between a clear refusal and a killed worker.
BYTES_PER_VOXEL = 40
MEM_BUDGET_GB = float(os.environ.get("CTADIPO_MEM_BUDGET_GB", "6.1"))
MAX_OUTPUT_VOXELS = int(MEM_BUDGET_GB * 1e9 / BYTES_PER_VOXEL)
MIN_Z_AFTER_RESAMPLE = 128               # fewer slices than this cannot carry five landmark planes

DEPOT_LABEL = {
    "perigonadal": "Perigonadal", "mesenteric": "Mesenteric / omental",
    "retroperitoneal": "Retroperitoneal", "thoracic_VAT": "Thoracic",
    "anterior_SUBQ": "Anterior (interscapular)", "dorsolumbar_SUBQ": "Dorsolumbar",
    "inguinal_SUBQ": "Inguinal", "gluteal_SUBQ": "Gluteal",
    "hindlimb_SUBQ": "Hindlimb", "head_neck_SUBQ": "Head / neck",
}
DEPOT_COLOUR = {
    "perigonadal": "#ed2624", "mesenteric": "#f5a623", "retroperitoneal": "#a53694",
    "thoracic_VAT": "#ef97a1", "anterior_SUBQ": "#3d51a3", "dorsolumbar_SUBQ": "#7dd1e4",
    "inguinal_SUBQ": "#64bc46", "gluteal_SUBQ": "#0aa84f", "hindlimb_SUBQ": "#808080",
    "head_neck_SUBQ": "#feda12",
}


@lru_cache(maxsize=1)
def _depot_rules():
    """The vendored depot definitions.

    Vendored, not reimplemented: these are the ten depot definitions validated scan by scan, so the
    app imports them. CTADIPO_DEPOT_RULES overrides with a research-tree copy for development.
    Returns (module, error_message) so a failure can be SHOWN rather than silently degrading to
    totals only -- which is what a bare `return None` used to do.
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


@lru_cache(maxsize=1)
def _landmark_nets():
    """Loaded once per process, not once per click: five folds re-read per run is pure waste."""
    return LMM.load(MODEL_DIR, device="cpu")


_SEG_LOCK = threading.Lock()
_SEG_CACHE: dict = {}


def _segmenter(mirroring: bool, progress=None):
    """The segmenter, loaded at most ONCE per process, fetching the checkpoint if it is absent.

    maxsize=2 on an lru_cache was wrong twice over. It would hold TWO fully materialised predictors
    -- the mirroring and non-mirroring variants -- which is roughly 800 MB of weights on a host
    whose default instance is 1 GB. And lru_cache does not serialise concurrent misses: two sessions
    arriving together both miss, both load, and briefly hold two copies before one is discarded. A
    single slot behind an explicit lock fixes both; flipping the mirroring switch re-loads, which is
    rare and cheap next to running out of memory.
    """
    with _SEG_LOCK:
        if _SEG_CACHE.get("mirroring") == mirroring and "seg" in _SEG_CACHE:
            return _SEG_CACHE["seg"]
        from ctadipo import segment, models
        root = models.ensure_checkpoint(progress=progress, results_root=NNUNET_RESULTS_ENV)
        _SEG_CACHE.clear()
        seg = segment.load_segmenter(root.parent.parent, device="cpu", mirroring=mirroring)
        _SEG_CACHE.update(seg=seg, mirroring=mirroring)
        return seg


# Only one measurement may run per process. See _measure_blocking for why this lives here
# rather than in the platform's connection cap.
_MEASURE_SLOT = threading.Semaphore(1)

_PREFETCH_STARTED = threading.Event()


def _prefetch_model():
    """Start fetching the checkpoint as soon as a scan is uploaded, not when Run is clicked.

    Measured against the published record, the 410 MB download takes about 7 minutes -- Zenodo
    serves it at roughly 1 MB/s. Doing that after the Run click adds those 7 minutes to the first
    measurement a container ever performs. The user spends at least half a minute after uploading,
    checking the voxel size and the orientation, and on a cold container that time is otherwise
    spent idle.

    Safe to call repeatedly and safe to race with the Run click: ensure_checkpoint returns
    immediately once the file is present, and its lock means a click arriving mid-download waits for
    this thread rather than starting a second one. Failures are swallowed here on purpose -- this is
    an optimisation, and the real attempt during the run reports errors properly.
    """
    if _PREFETCH_STARTED.is_set():
        return
    _PREFETCH_STARTED.set()

    def work():
        try:
            from ctadipo import models
            models.ensure_checkpoint(results_root=NNUNET_RESULTS_ENV)
        except Exception:
            _PREFETCH_STARTED.clear()          # let the run retry, and report properly if it fails

    threading.Thread(target=work, daemon=True).start()


def _release_segmenter():
    """Drop the nnU-Net weights once the network has run.

    Measured on a 540x529x529 scan, the depot partition takes the process from 3.5 GB to 7.1 GB
    because rule_A_slabs builds seventeen full-resolution boolean volumes at once. The container log
    confirmed the consequence: "oom (out of memory)". The weights are ~1.07 GB and are dead by that
    point, so holding them is the difference between finishing and being killed. The cost is a
    reload from the local cache on the next scan -- seconds, against a run that otherwise dies.
    """
    with _SEG_LOCK:
        _SEG_CACHE.clear()
    gc.collect()


def _lung_fn():
    """The lung sweep, pinned to the configuration the cohort was reviewed under.

    Optional at import time so the app still starts without the research tree, but its ABSENCE is
    reported rather than passed over: unsubtracted lung sits inside the fat band and is ~16% of a
    lean animal's fat against ~1.5% of an obese one's, so silently skipping it biases results along
    the adiposity axis itself.
    """
    try:
        from ctadipo.vendor.lung import lung_air
    except Exception:
        try:
            from lung_candidate2 import lung_air
        except Exception:
            return None, "lung module not available - lung will NOT be removed from fat"

    def fn(raw, body):
        return lung_air(raw, body, select="peak", coarse=True)
    return fn, ""


def models_status():
    """(blocking_problems, notice). Run is disabled ONLY for problems a click cannot fix.

    The previous version disabled Run whenever the segmentation checkpoint was absent -- which is
    precisely when the first-use download needs to happen, so the fetch could never be triggered by
    anything. A missing checkpoint with a URL configured is not a blocker, it is a download; the
    button stays live and the user is told the first run will take longer.

    It also tested only that a DIRECTORY existed, so a half-finished download counted as ready.
    models.is_ready checks the checkpoint file and its size instead.
    """
    from ctadipo import models
    blocking, notice = [], ""
    if not list(MODEL_DIR.glob("fold*.pt")):
        blocking.append("landmark model missing (no fold*.pt in %s). It is 18 MB and ships in the "
                        "bundle, so this means the deployment is incomplete." % MODEL_DIR)
    if not models.is_ready(NNUNET_RESULTS_ENV):
        if models.CKPT_URL:
            notice = ("The segmentation model (410 MB) will be downloaded on the first run and "
                      "cached, so the first measurement will take a few minutes longer.")
        else:
            blocking.append(
                "segmentation model not installed and no download URL configured. Set "
                "CTADIPO_CKPT_URL to the direct link for checkpoint_final.pth, or point "
                "CTADIPO_NNUNET at a folder that already holds it.")
    return blocking, notice


# -----------------------------
# Styling - matched to the DEXAdipo app
# -----------------------------
custom_css = ui.tags.style("""
.btn-wide { width: 100%; }
.btn-lg   { font-size: 1.1rem; padding: 0.8rem 1rem; }
.shiny-input-container { margin-bottom: 0.6rem; }

#brand_logo { text-align: left; }
#brand_logo img {
  height: 220px !important;
  width: auto !important;
  max-width: 100% !important;
  display: inline-block;
}
#pipeline_diagram { text-align: left; }
#pipeline_diagram img {
  height: 320px !important;
  width: auto !important;
  max-width: 100% !important;
  object-fit: contain;
  display: inline-block;
}
.flag-ok   { color: #0aa84f; font-weight: 600; }
.flag-warn { color: #F78800; font-weight: 600; }
.flag-bad  { color: #ed2624; font-weight: 600; }
.scan-meta { font-size: 0.92rem; color: #444; }
.scan-meta code { color: #003761; }
.prov { color: #F78800; font-weight: 600; }
""")

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.output_image("brand_logo", width="100%", height="auto"),
        ui.hr(),

        ui.h5("1) Upload a scan"),
        ui.input_file(
            "scan", "Choose a volume, or all slices of a DICOM series", multiple=True,
            accept=[".dcm", ".dicom", ".ima", ".nii", ".gz", ".tif", ".tiff",
                    ".npy", ".npz", ".h5", ".hdf5", ".zip"],
        ),
        ui.output_ui("upload_summary"),

        ui.h5("2) Voxel size (mm)"),
        ui.output_ui("spacing_ui"),
        ui.hr(),

        ui.h5("3) Orientation"),
        ui.input_select("dorsal", "Which way is the animal's back?",
                        {"auto": "Detect automatically", "low": "Back toward low Y",
                         "high": "Back toward high Y"}, selected="auto"),
        ui.hr(),

        ui.h5("4) Run"),
        ui.input_switch("fast", "Fast mode (skip mirroring)", True),
        ui.tags.small("On by default. Mirroring is ~8x slower for a median difference of "
                      "-0.02% in total fat; turn it on to reproduce the published "
                      "configuration exactly.", class_="scan-meta"),
        ui.output_ui("run_button"),
        ui.download_button("dl_csv", "Download CSV", class_="btn-info btn-lg btn-wide"),
        ui.hr(),
        open="open",
    ),

    custom_css,

    ui.h4(ui.tags.strong("Welcome to CTAdipo, a tool to measure adipose tissue depots in mice "
                         "from microCT volumes")),
    ui.output_image("pipeline_diagram", width="100%", height="auto"),

    ui.h4("Instructions for use"),
    ui.markdown(
        """
1. **Upload one microCT volume** of a mouse — accepted formats: NIfTI (.nii/.nii.gz), TIFF stack, NumPy array (.npy/.npz), HDF5 (.h5), or a DICOM series (select every slice, or upload a .zip of them). A whole-body scan gives all ten depots; a partial scan gives the totals and whichever depots it covers.
2. **Check the voxel size.** It is read from the file where the format carries it — DICOM, NIfTI, ImageJ TIFF — and must be entered by hand for NumPy and plain TIFF, which carry none. This is the setting that matters most: every volume scales with its cube, so a wrong value gives plausible numbers that are badly wrong. If the scan is anisotropic the three axes are shown separately.
3. Leave **Orientation** on automatic unless the 3-D view comes back with the back and belly swapped. The verdict is read off the scan's own geometry and the app says so where it is unsure. Every scan it was developed on shared one mounting orientation, so on a differently mounted animal check the 3-D view and use the override.
4. Click **Measure adipose depots**. The scan is segmented into body, abdominal wall and cavity, fat is taken by attenuation, lung is removed, and five anatomical planes divide the fat into named depots. Expect about **1 minute** in fast mode, or **7 minutes** with mirroring on, which is the setting the published numbers used.
5. Read the results below: total **VAT** and **SUBQ**, then the **ten sub-depots** in mL and in grams (at 0.9 g/cm³). Rotate the 3-D view to check the segmentation looks like the animal, and use **Download CSV** to export.
        """
    ),

    ui.hr(),
    ui.h4("Scan quality"),
    ui.output_ui("quality_panel"),

    ui.h4("3-D view"),
    ui.output_ui("render_panel"),

    ui.hr(),
    ui.h4("Results"),
    ui.output_ui("totals_panel"),
    ui.output_data_frame("depot_table"),
    ui.output_ui("run_log"),
    title=None,
)


# -----------------------------
# Server
# -----------------------------
# render.download is deprecated in shiny 1.8; the server log warns on every worker start.
_download_dec_base = getattr(render, "download_button", None) or render.download


def server(input, output, session):
    loaded = reactive.value(None)       # (volume, spacing, meta)
    result = reactive.value(None)
    logmsg = reactive.value("")
    ticks = reactive.value(0)
    steps: list[str] = []               # appended from the worker thread; list.append is atomic

    @render.image
    def brand_logo():
        p = APP_DIR / "CTadipo_LOGO.png"
        if not p.exists():
            return None                 # a dict with src=None raises inside the renderer
        return {"src": str(p), "alt": "CTAdipo logo", "delete_file": False}

    @render.image
    def pipeline_diagram():
        p = APP_DIR / "ctadipo_pipeline.png"
        if not p.exists():
            return None
        return {"src": str(p), "alt": "CTAdipo pipeline", "delete_file": False}

    @render.ui
    def run_button():
        blocking, notice = models_status()
        if blocking:
            return ui.TagList(
                ui.tags.div(ui.tags.strong("Cannot measure: models not installed."), ui.tags.br(),
                            ui.HTML("<br>".join(blocking)), class_="scan-meta flag-bad mb-2"),
                ui.input_action_button("run", "Measure adipose depots",
                                       class_="btn-secondary btn-lg btn-wide", disabled=True))
        return ui.TagList(
            ui.tags.div(ui.tags.small(notice), class_="scan-meta flag-warn mb-2")
            if notice else ui.TagList(),
            ui.input_action_button("run", "Measure adipose depots",
                                   class_="btn-primary btn-lg btn-wide"))

    # ---- upload -------------------------------------------------------------------------------
    @reactive.effect
    @reactive.event(input.scan)
    def _read_upload():
        f = input.scan()
        if not f:
            return
        total_mb = sum(x.get("size", 0) for x in f) / 1e6
        if total_mb > MAX_UPLOAD_MB:
            loaded.set(None)
            logmsg.set("That upload is %.0f MB; the limit is %d MB." % (total_mb, MAX_UPLOAD_MB))
            return
        try:
            if len(f) > 1:
                # A DICOM series arrives as many files. Shiny stores each under an index-numbered
                # name, so they are gathered into one directory, keeping the ORIGINAL file names --
                # the series reader sorts on geometry, but the names must at least be distinct.
                import shutil
                d = Path(f[0]["datapath"]).parent / "_series"
                d.mkdir(exist_ok=True)
                for x in f:
                    shutil.copyfile(x["datapath"], d / Path(x["name"]).name)
                path, name = d, f[0]["name"]
            else:
                path, name = Path(f[0]["datapath"]), f[0]["name"]

            vol, sp, meta = io_any.read_any(path, original_name=name)
            loaded.set((vol, sp, meta))
            result.set(None)
            logmsg.set("")
            _prefetch_model()
        except Exception as e:
            loaded.set(None)
            logmsg.set("Could not read that upload.\n\n%s" % e)

    @render.ui
    def upload_summary():
        L = loaded()
        if L is None:
            return ui.TagList()
        vol, sp, meta = L
        sp_txt = "not in the file" if sp is None else \
            "%.4f x %.4f x %.4f mm (%s)" % (sp[0], sp[1], sp[2], meta.get("spacing_source"))
        extra = ""
        if meta.get("hu_calibrated"):
            extra = ("<br><span class='flag-warn'>This file declares HU calibration. "
                     "The fat bands were derived on uncalibrated microCT.</span>")
        return ui.tags.div(
            ui.tags.small(
                ui.HTML("Read as <code>%s</code>, %d x %d x %d voxels.<br>Voxel size: %s%s"
                        % (meta.get("kind"), *meta["shape"], sp_txt, extra))),
            class_="scan-meta mb-2")

    @render.ui
    def spacing_ui():
        """Three boxes, always, seeded from the file.

        A single box was wrong in a way nothing downstream could catch: it took sp[0], the SLICE
        spacing, and applied it to all three axes. On a 0.5 x 0.1 x 0.1 mm DICOM that scales the
        in-plane axes by five and every reported volume by twenty-five, with a clean quality panel.
        """
        L = loaded()
        sp = L[1] if L else None
        src = (L[2].get("spacing_source") if L else "none")
        if sp is None:
            note = ui.tags.div(ui.tags.strong("This format carries no voxel size."),
                               " Enter it before running — nothing downstream can detect a wrong "
                               "value.", class_="scan-meta flag-warn mb-2")
            vals = (None, None, None)
        else:
            aniso = max(sp) / max(min(sp), 1e-9) > 1.01
            note = ui.tags.div(
                ui.HTML("Read from the file (<code>%s</code>)%s" %
                        (src, ". <span class='flag-warn'>This scan is anisotropic.</span>"
                         if aniso else ".")), class_="scan-meta mb-2")
            vals = tuple(round(float(x), 5) for x in sp)
        return ui.TagList(
            note,
            ui.input_numeric("vox_z", "z / slice (mm)", value=vals[0], min=VOX_MIN, max=VOX_MAX,
                             step=0.001),
            ui.input_numeric("vox_y", "y (mm)", value=vals[1], min=VOX_MIN, max=VOX_MAX, step=0.001),
            ui.input_numeric("vox_x", "x (mm)", value=vals[2], min=VOX_MIN, max=VOX_MAX, step=0.001),
        )

    def _spacing_now():
        """The spacing to resample with, validated. Raises with a readable message."""
        vals = []
        for k in ("vox_z", "vox_y", "vox_x"):
            v = input[k]() if k in input else None
            if v is None or not np.isfinite(float(v)):
                raise ValueError("Enter the voxel size for all three axes before running. "
                                 "This file does not carry one, and it cannot be guessed.")
            v = float(v)
            if not (VOX_MIN <= v <= VOX_MAX):
                raise ValueError("Voxel size %.4f mm is outside the accepted range %.3f-%.1f mm."
                                 % (v, VOX_MIN, VOX_MAX))
            vals.append(v)
        return tuple(vals)

    # ---- the measurement, off the event loop ---------------------------------------------------
    def _measure_blocking(raw_vol, spacing, fast, dorsal, aid):
        """Runs in a worker thread. Touches no Shiny API; only appends to `steps`."""
        def say(msg):
            steps.append(msg)

        # ONE MEASUREMENT AT A TIME PER PROCESS. A single scan peaks at 5-6 GB, so two running
        # together exhaust an 8 GB instance and the worker is OOM-killed, taking both users' work
        # with it and logging nothing. The alternative -- capping concurrent CONNECTIONS -- also
        # turns away everyone who merely wants to READ the page, so the limit belongs here, where it
        # can tell looking apart from measuring. A queued run waits its turn rather than failing.
        if not _MEASURE_SLOT.acquire(blocking=False):
            say("another scan is being measured on this server; waiting for it to finish")
            _MEASURE_SLOT.acquire()
        try:
            say("resampling to the 0.15 mm grid")
            raw = io_any.to_isotropic(raw_vol, spacing, iso=0.15,
                                      max_voxels=MAX_OUTPUT_VOXELS)
            n_vox = int(np.prod(raw.shape))
            if n_vox > MAX_OUTPUT_VOXELS:
                raise ValueError(
                    "This scan is %s = %.0f million voxels at 0.15 mm. Measuring it would need "
                    "about %.1f GB, and this server has a %.1f GB working budget, so it would be "
                    "killed part-way through rather than finishing. Options: crop the scan to the "
                    "torso, or run CTAdipo locally where memory is not capped (see the README)."
                    % ("x".join(str(int(x)) for x in raw.shape), n_vox / 1e6,
                       n_vox * BYTES_PER_VOXEL / 1e9, MEM_BUDGET_GB))
            if raw.shape[0] < MIN_Z_AFTER_RESAMPLE:
                raise ValueError(
                    "After resampling to 0.15 mm this volume is only %d slices (%.1f mm). That is "
                    "too short to carry the five anatomical planes the depots are defined on. "
                    "Check the voxel size." % (raw.shape[0], raw.shape[0] * 0.15))

            say("loading models")
            nets = _landmark_nets()
            # The checkpoint download, if this container has not fetched it yet, happens HERE --
            # inside the worker thread, reporting through the same channel as everything else.
            # Held in a dict, not a local: the weights are released mid-run, and a local name
            # would keep them alive no matter what the cache does.
            seg_holder = {"seg": _segmenter(not fast, progress=lambda frac, msg: say(msg))}
            lung, lung_note = _lung_fn()

            def _drop_segmenter():
                _release_segmenter()      # the process-wide cache
                seg_holder.clear()        # and this run's own reference
                gc.collect()

            out = pipeline.analyse(raw, seg_holder["seg"], nets, lung_fn=lung, device="cpu",
                                   dorsal_is_low=dorsal, depot_rules=_depot_rules()[0],
                                   aid=aid, progress=say,
                                   on_segmented=_drop_segmenter)
            out["_spacing"] = spacing
            out["_lung_note"] = lung_note

            # BUILD THE MESHES HERE, THEN THROW THE MASKS AWAY.
            #
            # rule_A_slabs returns seventeen full-resolution boolean volumes -- about 2.6 GB on a
            # normal scan -- and the session used to keep ten of them for its whole life purely so
            # the 3-D view could be drawn later. A decimated mesh is a few hundred kilobytes and is
            # all the drawing needs. Doing it here also releases the masks while this run is still
            # the only thing in memory, rather than leaving them resident until the next upload,
            # which is what made a SECOND scan fail on an 8 GB worker.
            say("building the 3-D view")
            meshes = {}
            for k, m in list(out.get("depots", {}).get("masks", {}).items()):
                if k not in pipeline.PARTITION_DEPOTS:
                    continue
                built = R3.mesh(np.asarray(m), step=3, smooth=1.2)
                if built is None:
                    continue
                v, f, note = R3.decimate(*built, target_faces=20000)
                meshes[k] = (v, f, note)
            if "depots" in out:
                out["depots"].pop("masks", None)
                out["depots"].pop("frame", None)
            out["meshes"] = meshes
            gc.collect()
            say("done")
            return out
        finally:
            _MEASURE_SLOT.release()

    @reactive.extended_task
    async def _measure(raw_vol, spacing, fast, dorsal, aid):
        """Off the asyncio loop.

        A synchronous handler here would block the single event loop for the whole 1-7 minute run,
        freezing every other session on the worker -- no uploads, no updates, stalled heartbeats.
        PyTorch and SciPy release the GIL inside their kernels, so a thread genuinely offloads.
        """
        return await asyncio.to_thread(_measure_blocking, raw_vol, spacing, fast, dorsal, aid)

    @reactive.effect
    @reactive.event(input.run)
    def _run():
        L = loaded()
        if L is None:
            logmsg.set("Upload a scan first.")
            return
        vol, sp, meta = L
        try:
            spacing = _spacing_now()
        except ValueError as e:
            logmsg.set(str(e))
            return
        steps.clear()
        logmsg.set("")
        result.set(None)
        _measure(vol, spacing, bool(input.fast()),
                 None if input.dorsal() == "auto" else (input.dorsal() == "low"),
                 Path(meta.get("name", "upload")).stem or "upload")

    @reactive.effect
    def _collect():
        st = _measure.status()
        if st == "running":
            reactive.invalidate_later(1.0)
            ticks.set(ticks() + 1)
            return
        if st == "success":
            try:
                out = _measure.result()
            except Exception:
                return
            # Free what the session does not need. rule_A_slabs returns 17 full-resolution boolean
            # volumes and `frame` still holds raw/body/cav/fat/vat/sat: about 4 GB retained per
            # session on a normal scan, which is how a worker dies on its second upload.
            result.set(out)      # the worker already built the meshes and freed every mask
        elif st == "error":
            try:
                _measure.result()
            except Exception as e:
                # The message, not the traceback: a traceback rendered into the page leaks absolute
                # server paths to anyone on the internet.
                logmsg.set(str(e))
                traceback.print_exc()

    # ---- output -------------------------------------------------------------------------------
    @render.ui
    def quality_panel():
        if _measure.status() == "running":
            ticks()
            done = list(steps)
            return ui.tags.div(
                ui.tags.strong("Measuring… "), done[-1] if done else "starting",
                ui.tags.br(), ui.tags.small("step %d" % len(done)), class_="scan-meta")
        r = result()
        if r is None:
            return ui.tags.p("No scan measured yet.", class_="scan-meta")
        q, sc = r["quality"], r["scanner"]

        def flag(ok, good, bad, warn=False):
            return ui.tags.span(good if ok else bad,
                                class_="flag-ok" if ok else ("flag-warn" if warn else "flag-bad"))

        rows = [
            ui.tags.li(
                ui.HTML("Abdominal wall: <b>%.0f mm&sup3;</b>, %.0f%% of the cavity edge lies "
                        "against it — " % (q["wall_mm3"], 100 * q["edge_on_wall"])),
                flag(q["cavity_ok"], "the VAT/SUBQ split is anchored",
                     "the wall is too thin to anchor the cavity, so the VAT/SUBQ SPLIT is "
                     "untrustworthy on this scan (the total is not)", warn=True),
                (ui.tags.span(" Wall volume passes, but only %.0f%% of the cavity edge lies "
                              "against it, against 45-65%% on scans an expert judged good. "
                              "Worth checking the 3-D view."
                              % (100 * q["edge_on_wall"]), class_="flag-warn")
                 if q["cavity_ok"] and q.get("edge_on_wall_low") else "")),
            ui.tags.li("Landmarks ", flag(q["landmarks_ordered"], "in anatomical order",
                                          "OUT OF ORDER — sub-depots cannot be trusted here")),
            ui.tags.li(
                "Head-to-tail direction read as ", ui.tags.strong(q["direction"]),
                " (lung apex %.0f mm from bladder) — " % q["direction_margin_mm"],
                flag(not q["direction_uncertain"], "confident",
                     "TOO CLOSE TO CALL. If the 3-D view has head and tail the wrong way round, "
                     "the depots are mirrored — anterior swapped with gluteal — and the totals "
                     "will look perfectly normal.", warn=True)),
        ]
        wide = [k for k, v in q["landmark_uncertain"].items() if v]
        rows.append(ui.tags.li("Landmark confidence — ",
                               flag(not wide, "all five placed confidently",
                                    "uncertain: %s. This scan may not resemble the training data."
                                    % ", ".join(wide), warn=True)))
        rows.append(ui.tags.li(
            "Lung ", flag(q["lung_subtracted"],
                          "removed — %.2f mL of fat (%.1f%% of fat before removal)"
                          % (r["totals"]["lung_fat_removed_mL"], 100 * q["lung_share_of_fat"]),
                          "NOT removed. Lung sits inside the fat band, which inflates lean "
                          "animals most (~16%% of a lean mouse's fat). %s"
                          % r.get("_lung_note", ""), warn=True)))
        if q.get("band_suspect"):
            rows.append(ui.tags.li(ui.tags.span(
                "Almost no fat was found in the band. If this scan is HU-calibrated, the bands "
                "here were derived on uncalibrated microCT and will under-count badly.",
                class_="flag-bad")))
        rows.append(ui.tags.li(
            ui.HTML("Scanner scale: <code>air p1 = %.0f HU</code>, fat band "
                    "<code>[%.0f, %.0f]</code> (%s field)"
                    % (sc["air_p1"], sc["fat_band"][0], sc["fat_band"][1], sc["scale"]))))
        if r.get("depots_error"):
            rows.append(ui.tags.li(ui.tags.span(
                "Sub-depots unavailable: %s" % r["depots_error"], class_="flag-bad")))
        return ui.tags.ul(*rows, class_="scan-meta")

    @render.ui
    def totals_panel():
        r = result()
        if r is None:
            return ui.TagList()
        t = r["totals"]
        return ui.tags.div(
            ui.HTML("<b>Total fat</b> %.2f mL (%.2f g) &nbsp;·&nbsp; "
                    "<b>VAT</b> %.2f mL (%.2f g) &nbsp;·&nbsp; "
                    "<b>SUBQ</b> %.2f mL (%.2f g)"
                    % (t["TotalFat_mL"], t["TotalFat_g"], t["VAT_mL"], t["VAT_g"],
                       t["SUBQ_mL"], t["SUBQ_g"])),
            class_="mb-2")

    def _table():
        r = result()
        if r is None or "depots" not in r:
            return None
        DR = _depot_rules()[0]
        prov = getattr(DR, "PROVISIONAL", set()) if DR else set()
        mL, g = r["depots"]["mL"], r["depots"]["g"]
        tot = max(sum(mL.values()), 1e-9)
        return pd.DataFrame(
            [{"Depot": DEPOT_LABEL.get(k, k) + (" *" if k in prov else ""),
              "Compartment": "Visceral" if k in pipeline.VAT_DEPOTS else "Subcutaneous",
              "Volume (mL)": round(mL[k], 3),
              "Mass (g)": round(g[k], 3),
              "% of total fat": round(100 * mL[k] / tot, 1)}
             for k in pipeline.PARTITION_DEPOTS if k in mL])

    @render.data_frame
    def depot_table():
        df = _table()
        if df is None:
            return render.DataGrid(pd.DataFrame({"": ["No sub-depots for this scan."]}))
        return render.DataGrid(df, width="100%")

    @render.ui
    def render_panel():
        r = result()
        if r is None or not r.get("meshes"):
            return ui.tags.p("The 3-D view appears after a scan is measured.", class_="scan-meta")
        try:
            import plotly.io as pio
            traces, dropped = [], []
            # The meshes were built in the worker thread and the masks freed there; drawing now
            # only assembles traces from a few hundred kilobytes of vertices.
            for k in pipeline.PARTITION_DEPOTS:
                got = r.get("meshes", {}).get(k)
                if not got:
                    continue
                v, f, note = got
                if note:
                    dropped.append(note)
                traces.append(R3.plotly_mesh(v, f, DEPOT_COLOUR.get(k, "#888"),
                                             DEPOT_LABEL.get(k, k)))
            if not traces:
                return ui.tags.p("Nothing large enough to draw.", class_="scan-meta")
            fig = R3.scene(traces)
            html = pio.to_html(fig, full_html=False, include_plotlyjs="cdn",
                               default_height="620px")
            warn = ui.tags.p(ui.tags.small(dropped[0]), class_="scan-meta flag-warn") \
                if dropped else ui.TagList()
            return ui.TagList(warn, ui.HTML(html))
        except Exception as e:
            return ui.tags.p("Could not build the 3-D view: %s" % e, class_="scan-meta")

    @render.ui
    def run_log():
        m = logmsg()
        return ui.TagList() if not m else ui.tags.pre(m, class_="scan-meta")

    @_download_dec_base(filename=CSV_NAME)
    def dl_csv():
        r = result()
        df = _table()
        req(r is not None)
        t = r["totals"]
        head = pd.DataFrame([
            {"Depot": "TOTAL FAT", "Compartment": "", "Volume (mL)": round(t["TotalFat_mL"], 3),
             "Mass (g)": round(t["TotalFat_g"], 3), "% of total fat": 100.0},
            {"Depot": "VAT (all visceral)", "Compartment": "Visceral",
             "Volume (mL)": round(t["VAT_mL"], 3), "Mass (g)": round(t["VAT_g"], 3),
             "% of total fat": round(100 * t["VAT_mL"] / max(t["TotalFat_mL"], 1e-9), 1)},
            {"Depot": "SUBQ (all subcutaneous)", "Compartment": "Subcutaneous",
             "Volume (mL)": round(t["SUBQ_mL"], 3), "Mass (g)": round(t["SUBQ_g"], 3),
             "% of total fat": round(100 * t["SUBQ_mL"] / max(t["TotalFat_mL"], 1e-9), 1)},
        ])
        out = pd.concat([head, df], ignore_index=True) if df is not None else head
        out["density_g_per_mL"] = DENSITY
        out["voxel_z_mm"], out["voxel_y_mm"], out["voxel_x_mm"] = r.get("_spacing", (None,) * 3)
        out["lung_fat_removed_mL"] = round(t["lung_fat_removed_mL"], 3)
        out["lung_subtracted"] = r["quality"]["lung_subtracted"]
        out["wall_mm3"] = round(r["quality"]["wall_mm3"], 1)
        out["cavity_ok"] = r["quality"]["cavity_ok"]
        yield out.to_csv(index=False)


app = App(app_ui, server)
