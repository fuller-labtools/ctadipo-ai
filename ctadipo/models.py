r"""Fetch and cache the nnU-Net checkpoint, which is too large to live in the deployment bundle.

WHAT IS AND IS NOT DOWNLOADED. The trained-model folder is three files: dataset.json (293 bytes),
plans.json (17 KB) and fold_0/checkpoint_final.pth. The first two ship in the bundle; only the
checkpoint is fetched, so the hosting side is a single file and there is no archive to unpack.

THE CHECKPOINT IS THE INFERENCE-ONLY ONE, 410 MB rather than 819 MB. nnU-Net saves optimizer and
grad-scaler state beside the weights so training can resume; the predictor never reads them.
Stripping them halves the download and was verified to produce bit-identical labels against the
predictions the published numbers came from -- "probably fine" would not be good enough for a file
every reported volume depends on.

WHY IT IS VERIFIED BY HASH. This runs on a server, downloads several hundred megabytes over a
network we do not control, and hands the result to a model whose output nobody eyeballs. A truncated
or corrupted download that still loads would produce a confident, wrong segmentation. The expected
SHA-256 is pinned, the file is written to a temporary name and only renamed into place once it
matches, so an interrupted download can never be mistaken for a complete one.

WHY THERE IS A LOCK. Several sessions can hit a cold container at once. Without a lock each one
starts its own 410 MB download into the same path, which is slow at best and a corrupted file at
worst. The first process to create the lock downloads; the others wait for the finished file.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen, Request

DATASET = "Dataset112_MouseAdiposeWallHU"
TRAINER = "nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__3d_lowres"
CKPT_NAME = "checkpoint_final.pth"

# The published model. This is baked in rather than left to an environment variable because
# shinyapps.io has no way to set one -- that is a Posit Connect feature -- so an env-var-only
# default would deploy an app that can never fetch its own model. CTADIPO_CKPT_URL still overrides,
# for local testing and for anyone pointing at their own copy.
#
# It must be the DIRECT download form (.../records/<id>/files/<name>?download=1). A record landing
# page returns HTML, which would download happily and then fail the hash check.
ZENODO_RECORD = "22947300"
CKPT_URL = os.environ.get(
    "CTADIPO_CKPT_URL",
    "https://zenodo.org/records/%s/files/checkpoint_final.pth?download=1" % ZENODO_RECORD)
CKPT_SHA256 = os.environ.get(
    "CTADIPO_CKPT_SHA256",
    "87c814c3e771ea0d65cf036c61155f4d301a0db17d252346490d921dcdafe80a")
# Exact size of the published file, used only as a progress-bar fallback when the server sends
# no Content-Length. Measured from the artefact, not derived from a rounded "409.7 MB".
CKPT_BYTES = int(os.environ.get("CTADIPO_CKPT_BYTES", "409655822"))

DOWNLOAD_TIMEOUT = 60
LOCK_WAIT = 1800


class ModelUnavailable(RuntimeError):
    pass


def cache_root() -> Path:
    """Where the checkpoint is kept. Must be writable and should survive between sessions.

    An app directory is not a safe default: some hosts mount it read-only, and on the ones that do
    not the download would be lost on every redeploy anyway. The system temp directory is writable
    everywhere and persists for the life of the container, which is the right lifetime.
    """
    p = os.environ.get("CTADIPO_CACHE")
    return Path(p) if p else Path(tempfile.gettempdir()) / "ctadipo-models"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def bundled_model_dir() -> Path:
    """The two small JSONs that ship with the app."""
    return Path(__file__).resolve().parent.parent / "nnunet" / DATASET / TRAINER


def model_folder(results_root=None) -> Path:
    """The trainer folder nnU-Net is pointed at. Assembled in the cache, not in the bundle."""
    if results_root is not None:
        return Path(results_root) / DATASET / TRAINER
    return cache_root() / DATASET / TRAINER


def is_ready(results_root=None) -> bool:
    f = model_folder(results_root)
    ck = f / "fold_0" / CKPT_NAME
    return (f / "plans.json").exists() and ck.exists() and ck.stat().st_size > 1_000_000


def ensure_checkpoint(progress=None, url: str | None = None, sha256: str | None = None,
                      results_root=None) -> Path:
    """Return the ready trainer folder, downloading the checkpoint once if needed.

    `progress(fraction, message)` is called during the download so a UI can show something during
    what may be several minutes on a cold container.
    """
    def say(frac, msg):
        if progress:
            try:
                progress(frac, msg)
            except Exception:
                pass

    dest = model_folder(results_root)
    ck = dest / "fold_0" / CKPT_NAME
    if is_ready(results_root):
        return dest

    # the small files come from the bundle; only the checkpoint is ever fetched
    src = bundled_model_dir()
    (dest / "fold_0").mkdir(parents=True, exist_ok=True)
    for name in ("dataset.json", "plans.json"):
        if not (dest / name).exists():
            if not (src / name).exists():
                raise ModelUnavailable(
                    "%s is missing from the app bundle. It is a few kilobytes and must be "
                    "deployed with the code; only the checkpoint is downloaded." % name)
            shutil.copyfile(src / name, dest / name)

    if ck.exists():
        return dest

    u = url or CKPT_URL
    if not u:
        raise ModelUnavailable(
            "No checkpoint URL is configured. Set CTADIPO_CKPT_URL to the direct download link "
            "for %s (410 MB). On Zenodo that is the .../records/<id>/files/<name>?download=1 form, "
            "not the record page." % CKPT_NAME)

    lock = dest / "fold_0" / ".downloading"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        # someone else is fetching it; wait for them rather than starting a second download
        say(0.0, "another session is downloading the model, waiting")
        t0 = time.time()
        while time.time() - t0 < LOCK_WAIT:
            if ck.exists():
                return dest
            if not lock.exists():
                break
            time.sleep(2.0)
        if not ck.exists():
            raise ModelUnavailable("timed out waiting for another session's model download")
        return dest

    tmp = ck.with_suffix(".part")
    try:
        say(0.0, "downloading the segmentation model (410 MB, first run only)")
        req = Request(u, headers={"User-Agent": "CTAdipo"})
        with urlopen(req, timeout=DOWNLOAD_TIMEOUT) as r, open(tmp, "wb") as fh:
            total = int(r.headers.get("Content-Length") or CKPT_BYTES or 0)
            done = 0
            while True:
                b = r.read(1 << 20)
                if not b:
                    break
                fh.write(b)
                done += len(b)
                if total:
                    say(min(done / total, 1.0),
                        "downloading the segmentation model: %.0f / %.0f MB"
                        % (done / 1e6, total / 1e6))
        want = sha256 or CKPT_SHA256
        if want:
            say(1.0, "verifying the download")
            got = sha256_of(tmp)
            if got != want:
                tmp.unlink(missing_ok=True)
                raise ModelUnavailable(
                    "the downloaded checkpoint does not match its expected SHA-256 "
                    "(got %s, expected %s). The file is corrupt or the URL is wrong; it has been "
                    "discarded rather than used." % (got[:16], want[:16]))
        os.replace(tmp, ck)          # atomic: the real name never exists half-written
        say(1.0, "model ready")
        return dest
    except ModelUnavailable:
        raise
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise ModelUnavailable("could not download the segmentation model: %s" % e)
    finally:
        lock.unlink(missing_ok=True)
