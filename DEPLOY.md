# Deploying CTAdipo

**Live:** https://fullerlabtools.shinyapps.io/ctadipo/ (app 18007440, account `fullerlabtools`)
**Models:** Zenodo record [22947300](https://zenodo.org/records/22947300) - the checkpoint URL is
baked into `ctadipo/models.py`, because shinyapps.io has no way to set an environment variable.

Redeploy after any change with:

```
rsconnect deploy shiny . -n fullerlabtools --title CTAdipo
```

Delete `__pycache__` first - a deploy from a different Python leaves .pyc files in the tree and
rsconnect uploads whatever it finds.


Every fact below was checked on this machine or against Posit's documentation. Where something
could not be verified from here, it says so.

---

## 1. Publish the models to Zenodo

Two files, both built for you already:

| file | size | what it is |
|---|---|---|
| `checkpoint_final.pth` | **409.7 MB** | nnU-Net segmentation model, fold 0, inference-only |
| `ctadipo_landmarks.zip` | 17.1 MB | the five landmark folds (also in the bundle; published so they are citable) |

The checkpoint is **half the size of the trained one** — 819 MB on disk carries 409 MB of optimizer
and grad-scaler state that exists only to resume training. Removing it was verified to produce
**bit-identical labels** through the same code path, on scans at both ends of the adiposity range.
That is not an assumption: the first test of it appeared to fail, because it compared the stripped
checkpoint run in-process against a prediction made months earlier by the CLI, which differs in two
ways at once. The controlled comparison — both checkpoints, same process, same scan — agrees to
every voxel.

```bash
cd <your clone of this repo>; $env:ZENODO_TOKEN = "<your token>"; conda run -n torch-microCT --no-capture-output python upload_to_zenodo.py --title "CTAdipo models"
```

It creates a **draft and stops**. Zenodo publishing is irreversible — the DOI is minted and files can
never be changed or removed — so review it in the browser and press Publish yourself. Run it with
`--dry-run` first if you want to see what it would send.

The token is read from `ZENODO_TOKEN` in your own shell and is never written anywhere.

This has been done: record **22947300**, published, and the direct URL is pinned as the default
in `ctadipo/models.py` alongside its SHA-256. It was verified by running the real `ensure_checkpoint`
against the live URL - 410 MB fetched, hash matched, nnU-Net loaded from the result.

Zenodo served it at about **1.0 MB/s**, so a cold container spends ~7 minutes fetching the model
before it can measure anything. The app therefore starts that download when a scan is **uploaded**,
not when Run is clicked, so it overlaps with the user checking their voxel size.

### Why the checkpoint is fetched rather than bundled

Neither shinyapps.io nor Connect Cloud offers a persistent writable volume that survives a redeploy,
and a 410 MB binary cannot live in a git repository, which rules out the Connect Cloud git route for
it. So `ctadipo/models.py` downloads it on first use into a writable cache and keeps it for the life
of the container. It verifies the SHA-256 before use, writes to a temporary name and renames
atomically so an interrupted download can never be mistaken for a complete one, and takes a lock so
two simultaneous sessions do not both pull 410 MB into the same path.

`dataset.json` and `plans.json` are 17 KB between them and ship in the bundle.

---

## 2. Deploy

### The account is already authorised on this machine

`%APPDATA%\rsconnect-python\servers.json` already holds two registered accounts, **`fullerlab`** and
**`fullerlabtools`**, from the DEXAdipo deploy (live at `fullerlabtools.shinyapps.io/dexadipo-ai/`).
So `rsconnect add` — the only step that involves a token — **does not need running again**, and no
command below contains a credential.

If it ever does need redoing: shinyapps.io dashboard → avatar → Tokens → Show, then copy the
`rsconnect add` command it displays and run it in your own terminal.

### Install the deploy tool and push

```bash
pip install rsconnect-python
```

```bash
cd <your clone of this repo>; rsconnect deploy shiny . -n fullerlabtools --title CTAdipo --entrypoint app:app
```

**`.rscignore` does nothing** — it is an R-only feature that rsconnect-python ignores entirely. That
is why the training data and the checkpoint were physically moved out of the app tree rather than
excluded by a file. If you ever need exclusions, they are repeated quoted `-x` globs
(`-x 'data/**' -x '*.tif'`), not an ignore file.

### What actually uploads

**19 MB.** The bundle is the code, the two logos, the pipeline diagram, `requirements.txt`, the five
landmark folds (18 MB of the 19), and the two nnU-Net JSONs.

Deliberately **outside** the tree, in `../ctadipo_traindata/` and `../ctadipo_local_model/`:

- `landmarks/` — 616 MB, 1,186 `.npz` training files. Not used by the app, and their filenames are
  cohort animal identifiers, so deploying them would put the cohort roster and per-animal derived
  features on a third-party host.
- `checkpoint_final.pth` — 390 MB, fetched at runtime instead.

### Settings that matter

| setting | value | why |
|---|---|---|
| Python | **3.11** (`.python-version`) | floor 3.10 is set by scipy, nnunetv2 *and* shiny; 3.13 has no wheels for several pins; shinyapps.io tops out at 3.12 |
| Instance size | **larger than the default** | the default is 1 GB RAM and one scan peaks at 5–6 GB |
| Timeout | raise the idle/startup timeouts | the first run includes a 410 MB download; a normal run is ~7 min with mirroring |

---

## 3. Resources

| | shinyapps.io Basic | Connect Cloud |
|---|---|---|
| RAM | 8 GB (default instance is 1 GB — change it) | 32 GB |
| CPU | 2 | 8 |
| route | `rsconnect deploy` push | git-backed |

A 540×529×529 volume at 0.15 mm is 151 M voxels; `3d_lowres` works at 0.398 mm, which is 27 patches.
Peak for one scan is **5–6 GB**, not the 2.56 GB an earlier version of this file quoted — that figure
measured the nnU-Net step alone and ignored the raw volume, its z-scored copy, nnU-Net's own copy,
the median filter output and the mask set.

Three guards keep it in bounds, all enforced in code:

- the session drops depot masks and the frame once the render is built (they were ~4 GB retained per
  session, which killed the worker on a *second* upload rather than the first),
- `to_isotropic` refuses any resample past 400 M output voxels, naming the voxel size — without it a
  clinical CT at 3 mm slices asks for 2.3 TiB, and inside a memory-limited container that allocation
  *succeeds* and the process is OOM-killed while the pages fault in, silently,
- models load once per process behind a lock, not once per click.

`torch.set_num_threads` reads the **cgroup quota**, not `os.cpu_count()`, which reports the host's
cores inside a container and would start 8–16 threads on a 2-core plan.

### Mirroring

The app defaults to mirroring **on**, which is what the published numbers used. Turning it off is
about 8× faster and was measured to change total fat by a median of **−0.02%** (worst scan 2.7%,
n = 6), so Fast mode is defensible — but the default should not silently diverge from the paper.

---

## 4. Known limitations

- **It has never executed on Linux.** Everything here was developed and tested on Windows. The
  shipped code was audited for Linux-specific hazards and came back clean — no absolute paths
  reachable, no subprocess or conda calls, no `matplotlib` config-dir problem, the zip-slip guard
  correct under `os.sep = '/'`, and nnU-Net 2.8.0 needing no `nnUNet_*` environment variables for
  inference — but an audit is not an execution. Watch the first deploy's build log.
- **Hardware polygons are not shipped.** The original derivation subtracts hand-drawn polygons keyed
  to scanner *and* reconstruction crop, because on that rig a breathing pad fuses to the animal and
  its dark interior is labelled cavity — counted as visceral fat, inflating VAT by 8–38% on affected
  scans. They cannot transfer to another scanner. CTAdipo does what generalises (`keep_largest3d`,
  which removes *detached* hardware) and shows the mask instead of pretending the problem is solved.
- **Dorsoventral orientation is not validated for a differently-mounted animal.** All 1,186 training
  scans share one orientation, so there are no examples of the other and no agreement rate can be
  quoted for one. The axis is decided from geometry, the decision margin is surfaced, and the sidebar
  has an override.
- **The app subtracts lung on every scan**, whereas the source cohort skipped it on 584 reviewer-
  flagged scans. That is deliberate — skipping is not neutral, it inflates lean animals by ~16% of
  their fat against ~1.5% for obese ones — but it means the app and `_doz_long_v3.csv` will
  legitimately disagree on those scans.
- **pyvista cannot be used.** It needs a graphics context and neither host has one. Do not add it to
  `requirements.txt`; it will install and then fail at runtime.
