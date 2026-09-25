r"""Upload the CTAdipo models to Zenodo and print the direct download URLs the app needs.

RUN THIS YOURSELF. It reads the token from the ZENODO_TOKEN environment variable in your own shell
and never writes it anywhere. Nothing else in this repository reads that variable.

    Windows PowerShell:
        $env:ZENODO_TOKEN = "<your new token>"
        python upload_to_zenodo.py --title "CTAdipo models" --dry-run
        python upload_to_zenodo.py --title "CTAdipo models"

    bash:
        export ZENODO_TOKEN="<your new token>"
        python upload_to_zenodo.py --title "CTAdipo models"

WHAT IT UPLOADS
    checkpoint_final.pth   410 MB  the nnU-Net segmentation model, optimizer state stripped
                                   (verified bit-identical to the full 819 MB checkpoint)
    ctadipo_landmarks.zip   18 MB  the five landmark folds

It creates a DRAFT deposition and stops. It does NOT publish, because publishing on Zenodo is
irreversible -- the DOI is minted and the files can never be changed or removed afterwards. Check
the draft in the browser, then press Publish there yourself.

--dry-run does everything except talk to Zenodo, so you can confirm what would be uploaded first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAINER_REL = (Path("Dataset112_MouseAdiposeWallHU")
               / "nnUNetTrainer_100epochs__nnUNetResEncUNetMPlans__3d_lowres")
# The checkpoint deliberately lives OUTSIDE the app tree. rsconnect-python uploads the whole
# directory and has no working ignore mechanism (.rscignore is an R-only feature), so a 390 MB file
# inside it would be pushed on every deploy -- which is the opposite of the point of hosting it.
CKPT = HERE.parent / "ctadipo_local_model" / TRAINER_REL / "fold_0" / "checkpoint_final.pth"
if not CKPT.exists():                                   # fall back to an in-tree copy if present
    CKPT = HERE / "nnunet" / TRAINER_REL / "fold_0" / "checkpoint_final.pth"
LANDMARKS = HERE / "model"
BASE = "https://zenodo.org/api"


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_landmark_zip() -> Path:
    """Build the archive in a TEMP directory, never in the app tree.

    Writing it beside app.py adds 17 MB to a 19 MB deployment bundle, and rsconnect-python uploads
    whatever is in the directory with no working ignore mechanism -- so a helper script for
    publishing would have quietly doubled every deploy.
    """
    folds = sorted(LANDMARKS.glob("fold*.pt"))
    if not folds:
        raise SystemExit("no fold*.pt in %s" % LANDMARKS)
    out = Path(tempfile.mkdtemp(prefix="ctadipo-upload-")) / "ctadipo_landmarks.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in folds:
            z.write(f, f.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="CTAdipo models")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sandbox", action="store_true",
                    help="use sandbox.zenodo.org, which needs its own separate token")
    a = ap.parse_args()

    base = "https://sandbox.zenodo.org/api" if a.sandbox else BASE
    if not CKPT.exists():
        raise SystemExit("checkpoint not found at %s" % CKPT)
    zip_path = build_landmark_zip()

    files = [CKPT, zip_path]
    print("Files to upload:")
    for f in files:
        print("  %-26s %8.1f MB" % (f.name, f.stat().st_size / 1e6))
    print("\nSHA-256 (pin these in the app):")
    digests = {f.name: sha256_of(f) for f in files}
    for k, v in digests.items():
        print("  %-26s %s" % (k, v))

    if a.dry_run:
        print("\n--dry-run: nothing sent. Re-run without it to create the draft.")
        return 0

    token = os.environ.get("ZENODO_TOKEN", "").strip()
    if not token:
        raise SystemExit(
            "ZENODO_TOKEN is not set in this shell. Set it, then re-run.\n"
            "  PowerShell:  $env:ZENODO_TOKEN = \"<token>\"\n"
            "  bash:        export ZENODO_TOKEN=\"<token>\"")

    import requests
    s = requests.Session()
    s.params = {"access_token": token}

    r = s.post("%s/deposit/depositions" % base, json={})
    if r.status_code == 401:
        raise SystemExit("Zenodo rejected the token (401). Check it is a personal access token "
                         "with the deposit:write and deposit:actions scopes, and that it is for "
                         "%s rather than the other site." % base)
    r.raise_for_status()
    dep = r.json()
    dep_id, bucket = dep["id"], dep["links"]["bucket"]
    print("\ncreated draft deposition %s" % dep_id)

    for f in files:
        size = f.stat().st_size
        print("  uploading %s (%.1f MB)..." % (f.name, size / 1e6), flush=True)
        with open(f, "rb") as fh:
            up = s.put("%s/%s" % (bucket, f.name), data=fh,
                       headers={"Content-Type": "application/octet-stream"})
        up.raise_for_status()
        print("    done")

    meta = {"metadata": {
        "title": a.title,
        "upload_type": "software",
        "description": (
            "Trained models for CTAdipo, a tool that measures adipose tissue depots in mice from "
            "microCT volumes.<br><br>"
            "<b>checkpoint_final.pth</b> - nnU-Net (3d_lowres, ResEnc-M) segmenting outer body, "
            "abdominal wall and abdominal cavity on a 0.15 mm isotropic grid. Fold 0 only, which is "
            "the configuration the published measurements used. Optimizer and grad-scaler state "
            "have been removed; the remaining weights were verified to produce bit-identical "
            "labels to the full training checkpoint.<br><br>"
            "<b>ctadipo_landmarks.zip</b> - five cross-validation folds of the landmark model, "
            "which places five craniocaudal planes (lung apex, diaphragm, cranial and caudal "
            "kidney, bladder) used to divide fat into named depots.<br><br>"
            "SHA-256: " + "; ".join("%s = %s" % (k, v) for k, v in digests.items())),
        "access_right": "open",
        "license": "cc-by-4.0",
    }}
    r = s.put("%s/deposit/depositions/%s" % (base, dep_id), data=json.dumps(meta),
              headers={"Content-Type": "application/json"})
    r.raise_for_status()

    host = base.replace("/api", "")
    print("\nDRAFT CREATED - not published. Review and publish it yourself at:")
    print("  %s/uploads/%s" % (host, dep_id))
    print("\nOnce published, the direct download URLs will be:")
    for f in files:
        print("  %s/records/%s/files/%s?download=1" % (host, dep_id, f.name))
    print("\nThen set, for the app:")
    print("  CTADIPO_CKPT_URL     = %s/records/%s/files/%s?download=1"
          % (host, dep_id, CKPT.name))
    print("  CTADIPO_CKPT_SHA256  = %s" % digests[CKPT.name])
    return 0


if __name__ == "__main__":
    sys.exit(main())
