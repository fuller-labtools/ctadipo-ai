r"""Read any microCT volume the app is likely to be handed, and say honestly what is known about it.

FORMATS: a DICOM series (a folder, a zip, or a pile of loose slices), NIfTI (.nii/.nii.gz), a TIFF
stack, a NumPy array (.npy/.npz), and HDF5 (.h5/.hdf5).

TWO PROPERTIES DECIDE WHETHER A SCAN CAN BE MEASURED AT ALL, and neither is guaranteed to be in the
file:

  SPACING. Everything downstream is defined on a 0.15 mm isotropic grid -- the fat band, the median
  filter, the landmark positions, the depot slabs. A volume with unknown or wrong spacing produces
  numbers that look entirely plausible and are wrong by the cube of the error. So spacing is never
  guessed: it is read from the file where the format carries it, and otherwise returned as None for
  the caller to ask about. A NumPy array carries no spacing at all and never will.

  INTENSITY SCALE. The pipeline's fat band is in HU. DICOM carries RescaleSlope/RescaleIntercept and
  is converted here. Nothing else does, so for the other formats the values are passed through
  unchanged and `hu_calibrated` is False -- which matters, because this project's own data is not
  HU-calibrated either: the air peak ranges from -773 to -1424 HU across scanners, and the
  derivation picks the fat band from the scan's own air percentile rather than trusting the numbers.
  Reporting the scale honestly is what lets that logic run; silently assuming HU would not.

Returns (volume as int16-ish float32 (Z, Y, X), spacing (z, y, x) mm or None, meta dict).
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

NIFTI_EXT = (".nii", ".nii.gz")
TIFF_EXT = (".tif", ".tiff")
NUMPY_EXT = (".npy", ".npz")
HDF5_EXT = (".h5", ".hdf5", ".he5")
DICOM_EXT = (".dcm", ".dicom", ".ima")

# Smallest volume that can meaningfully be partitioned into depots. Also the guard that stops a
# 2-D radiograph, which reads as (1, Y, X), from being measured as if it were a scan.
MIN_DIM = 32


class ReadError(RuntimeError):
    pass


# --------------------------------------------------------------------------------- dispatch
def sniff(path: Path, original_name=None) -> str:
    """What kind of input this is, by structure rather than by extension alone.

    Extension is unreliable here: DICOM slices are routinely written with no extension at all, a
    'series' may arrive as a folder, a zip or a single multi-frame file, and an upload handler may
    have renamed the file on the way in -- hence original_name, which wins when given.
    """
    p = Path(path)
    name = str(original_name or p.name).lower()
    if p.is_dir():
        return "dicom_dir"
    if name.endswith(".zip"):
        return "zip"
    if name.endswith(NIFTI_EXT):
        return "nifti"
    if name.endswith(TIFF_EXT):
        return "tiff"
    if name.endswith(NUMPY_EXT):
        return "numpy"
    if name.endswith(HDF5_EXT):
        return "hdf5"
    if name.endswith(DICOM_EXT):
        return "dicom_file"
    with open(p, "rb") as f:                      # DICOM magic at byte 128, the common no-extension case
        if f.read(132)[128:132] == b"DICM":
            return "dicom_file"
    raise ReadError("unrecognised file type: %s" % p.name)


def read_any(path, dataset: Optional[str] = None, spacing_hint=None, original_name=None):
    """Main entry. spacing_hint (z,y,x) mm is used ONLY where the file carries no spacing.

    `original_name` matters more than it looks. Shiny stores an upload as `<index><suffix>` where
    the suffix is Path(name).suffix, so `mouse.nii.gz` lands on disk as `0.gz` -- which sniffs as
    nothing and raises, for a format the app advertises. The caller passes the name the user
    actually uploaded, and that is what the type is decided from.
    """
    p = Path(path)
    kind = sniff(p, original_name)
    # Only a FILE can be relinked under its true extension. A DICOM series arrives as a DIRECTORY,
    # and both os.link and shutil.copyfile raise on one -- so calling this unconditionally crashed
    # every multi-slice DICOM upload, which is the input the instructions push hardest.
    if p.is_file():
        p = _with_real_suffix(p, original_name)
    if kind == "zip":
        p, kind = _unzip(p)
    fn = {"dicom_dir": _read_dicom_series, "dicom_file": _read_dicom_series,
          "nifti": _read_nifti, "tiff": _read_tiff,
          "numpy": _read_numpy, "hdf5": _read_hdf5}[kind]
    vol, sp, meta = fn(p, dataset) if kind == "hdf5" else fn(p)
    vol = np.ascontiguousarray(vol.astype(np.float32))
    if vol.ndim != 3:
        raise ReadError("expected a 3-D volume, got shape %s" % (vol.shape,))
    # A 2-D image read through a 3-D reader arrives as (1, Y, X) and passes ndim == 3. A DEXA
    # radiograph does exactly this, and it would flow all the way to a depot table of confident
    # nonsense. Measuring fat needs a volume, so demand one explicitly.
    if min(vol.shape) < MIN_DIM:
        raise ReadError(
            "this is a %s image, not a CT volume: shape %s has a dimension under %d voxels. "
            "CTAdipo measures volumes; a single projection or a handful of slices cannot be "
            "partitioned into depots." % ("2-D" if min(vol.shape) == 1 else "very thin",
                                          vol.shape, MIN_DIM))
    if sp is None and spacing_hint is not None:
        sp = tuple(float(x) for x in spacing_hint)
        meta["spacing_source"] = "user"
    meta.update(kind=kind, shape=tuple(int(x) for x in vol.shape),
                name=str(original_name or Path(path).name),
                intensity_p1=float(np.percentile(vol, 1)),
                intensity_p99=float(np.percentile(vol, 99)))
    return vol, sp, meta


MAX_UNZIP_BYTES = 4 * 1024 ** 3
MAX_UNZIP_MEMBERS = 50_000
MAX_COMPRESSION_RATIO = 200


def _junk(name: str) -> bool:
    """macOS Finder's Compress adds a __MACOSX/._name fork to EVERY archive.

    That single extra file used to force the whole archive down the DICOM-series path -- so a Mac
    user zipping one NIfTI got "no DICOM series found", naming a format they never mentioned.
    """
    parts = Path(name).parts
    return (any(x in ("__MACOSX", ".DS_Store") for x in parts)
            or Path(name).name.startswith("._")
            or Path(name).name.startswith("."))


def _with_real_suffix(p: Path, original_name):
    """Present the file under its TRUE extension, linking rather than copying where possible.

    Sniffing the type from the uploaded name is not sufficient on its own, because several readers
    decide for themselves from the path: ITK's NIfTI reader validates the filename and refuses
    anything that is not .nii/.nii.gz/.hdr/.img, whatever ImageIO is named explicitly. Shiny stores
    an upload as `<index><suffix>` keeping only Path(name).suffix, so `mouse.nii.gz` arrives as
    `0.gz` and every NIfTI upload failed.

    A hard link costs nothing and leaves the original untouched; a copy is the fallback for
    filesystems that will not link.
    """
    if not original_name:
        return p
    name = Path(str(original_name)).name
    low = name.lower()
    suffix = ".nii.gz" if low.endswith(".nii.gz") else Path(name).suffix
    if not suffix or p.name.lower().endswith(suffix.lower()):
        return p
    target = p.parent / ("ctadipo_input" + suffix)
    if target.exists():
        return target
    try:
        os.link(p, target)
    except Exception:
        shutil.copyfile(p, target)
    return target


def _unzip(p: Path):
    """Extract a zip safely. This runs on files uploaded by strangers over the internet.

    ZipFile.extractall() honours whatever paths the archive claims, so an entry named
    `../../something` writes outside the destination -- the "zip slip" bug. Every member is
    therefore resolved against the target and anything escaping it is skipped, along with symlinks.

    Size is capped as well: DEFLATE reaches about 1000:1 on zeros, so a 20 MB upload can write
    20 GB into the container's disk. Both the total and the per-member ratio are checked.
    """
    out = p.with_suffix("")
    out.mkdir(exist_ok=True)
    root = out.resolve()
    written = 0
    with zipfile.ZipFile(p) as z:
        members = [i for i in z.infolist() if not i.is_dir() and not _junk(i.filename)]
        if len(members) > MAX_UNZIP_MEMBERS:
            raise ReadError("this archive holds %d files; the limit is %d"
                            % (len(members), MAX_UNZIP_MEMBERS))
        total = sum(i.file_size for i in members)
        if total > MAX_UNZIP_BYTES:
            raise ReadError("this archive expands to %.1f GB; the limit is %.0f GB"
                            % (total / 1e9, MAX_UNZIP_BYTES / 1e9))
        for info in members:
            if (info.external_attr >> 28) == 0xA:            # symlink
                continue
            if info.compress_size and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
                raise ReadError("refusing %s: it expands %.0fx, which is a compression bomb"
                                % (info.filename, info.file_size / max(info.compress_size, 1)))
            dest = (root / info.filename).resolve()
            if not str(dest).startswith(str(root) + os.sep):
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(dest, "wb") as fh:
                shutil.copyfileobj(src, fh)
            written += info.file_size
            if written > MAX_UNZIP_BYTES:
                raise ReadError("archive exceeded the %.0f GB extraction limit"
                                % (MAX_UNZIP_BYTES / 1e9))
    inner = [c for c in out.rglob("*") if c.is_file() and not _junk(str(c.relative_to(out)))]
    if not inner:
        raise ReadError("zip archive is empty")
    # Dispatch on CONTENT, not on count. A zip holding one volume plus a README is still a volume.
    vols = []
    for c in inner:
        try:
            k = sniff(c)
        except ReadError:
            continue
        if k not in ("dicom_file", "dicom_dir"):
            vols.append((c, k))
    if len(vols) == 1:
        return vols[0]
    # otherwise it is a DICOM series: hand back whichever directory holds the most files
    from collections import Counter
    best = Counter(c.parent for c in inner).most_common(1)[0][0]
    return best, "dicom_dir"


# --------------------------------------------------------------------------------- readers
def _read_dicom_series(p: Path):
    """A DICOM series, converted to HU via RescaleSlope/RescaleIntercept.

    Slice ORDER is taken from ImagePositionPatient projected on the slice normal, not from the file
    names and not from InstanceNumber. Names sort lexically (slice10 before slice2) and
    InstanceNumber is not required to follow geometry; getting the order wrong silently mirrors or
    interleaves the animal, and every landmark would then be placed in a plausible, wrong position.
    """
    import SimpleITK as sitk
    d = p if p.is_dir() else p.parent
    reader = sitk.ImageSeriesReader()
    ids = reader.GetGDCMSeriesIDs(str(d))
    if not ids:
        raise ReadError("no DICOM series found in %s" % d)
    if len(ids) > 1:                              # pick the longest series, and say so
        best = max(ids, key=lambda s: len(reader.GetGDCMSeriesFileNames(str(d), s)))
    else:
        best = ids[0]
    files = reader.GetGDCMSeriesFileNames(str(d), best)
    reader.SetFileNames(files)
    img = reader.Execute()                        # SimpleITK applies slope/intercept and sorts on geometry
    sx, sy, sz = img.GetSpacing()
    return (sitk.GetArrayFromImage(img), (sz, sy, sx),
            {"hu_calibrated": True, "spacing_source": "dicom",
             "series": best, "n_series": len(ids), "n_slices": len(files)})


def _read_nifti(p: Path):
    import SimpleITK as sitk
    img = sitk.ReadImage(str(p))
    sx, sy, sz = img.GetSpacing()
    return (sitk.GetArrayFromImage(img), (sz, sy, sx),
            {"hu_calibrated": False, "spacing_source": "nifti"})


def _read_tiff(p: Path):
    """TIFF stack. ImageJ keeps z spacing in its metadata and xy in the resolution tags."""
    import tifffile as tiff
    with tiff.TiffFile(str(p)) as tf:
        vol = tf.asarray()
        sp, src = None, "none"
        try:
            ij = tf.imagej_metadata or {}
            tags = tf.pages[0].tags

            def _res(nm):
                t = tags.get(nm)
                if t is None:
                    return None
                v = t.value
                if isinstance(v, (tuple, list)) and len(v) == 2 and v[1]:
                    return v[0] / v[1]
                return float(v)

            unit = str(ij.get("unit", "")).lower()
            scale = {"um": 1e-3, "micron": 1e-3, "microns": 1e-3, "\xb5m": 1e-3,
                     "mm": 1.0, "cm": 10.0}.get(unit, 1.0)
            xr, yr, zs = _res("XResolution"), _res("YResolution"), ij.get("spacing")
            if xr and yr and zs:
                sp, src = (float(zs) * scale, scale / yr, scale / xr), "imagej"
        except Exception:
            sp, src = None, "none"
    if vol.ndim == 4:                             # (Z, Y, X, C) -- take the single channel if there is one
        vol = vol[..., 0] if vol.shape[-1] == 1 else vol.mean(-1)
    return vol, sp, {"hu_calibrated": False, "spacing_source": src}


MAX_ARRAY_BYTES = 8 * 1024 ** 3


def _read_numpy(p: Path):
    """A raw array. It carries no spacing and no intensity convention, so both come from the user.

    SIZE IS CHECKED BEFORE DECOMPRESSION. The obvious loop over an .npz -- `[k for k in z.files if
    z[k].ndim == 3]` -- decompresses every array in the file just to read its ndim, so a 30 MB
    upload holding 30 GB of zeros is a memory bomb with no malice required. The zip directory
    already records each member's uncompressed size, and the .npy header records the shape, so both
    are read without inflating anything.
    """
    if p.suffix.lower() == ".npz":
        with zipfile.ZipFile(p) as z:
            cands = []
            for info in z.infolist():
                if not info.filename.endswith(".npy"):
                    continue
                with z.open(info) as fh:
                    try:
                        shape, _fortran, dtype = _npy_header(fh)
                    except Exception:
                        continue
                if len(shape) == 3:
                    cands.append((info.filename[:-4], shape, dtype, info.file_size))
            if not cands:
                raise ReadError(".npz contains no 3-D array")
            key, shape, dtype, _ = cands[0]
            nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
            if nbytes > MAX_ARRAY_BYTES:
                raise ReadError("array %s is %.1f GB, past the %.0f GB limit"
                                % (key, nbytes / 1e9, MAX_ARRAY_BYTES / 1e9))
        arr = np.load(p)[key]
        meta = {"npz_key": key, "npz_keys": [c[0] for c in cands]}
    else:
        arr = np.load(p, mmap_mode="r")
        if arr.nbytes > MAX_ARRAY_BYTES:
            raise ReadError("this array is %.1f GB, past the %.0f GB limit"
                            % (arr.nbytes / 1e9, MAX_ARRAY_BYTES / 1e9))
        meta = {}
    meta.update(hu_calibrated=False, spacing_source="none")
    return np.asarray(arr), None, meta


def _npy_header(fh):
    """(shape, fortran_order, dtype) from a .npy stream without reading the data."""
    import numpy.lib.format as npf
    version = npf.read_magic(fh)
    if version == (1, 0):
        shape, fortran, dtype = npf.read_array_header_1_0(fh)
    elif version == (2, 0):
        shape, fortran, dtype = npf.read_array_header_2_0(fh)
    else:
        raise ReadError("unsupported .npy version %s" % (version,))
    return shape, fortran, dtype


def _read_hdf5(p: Path, dataset: Optional[str] = None):
    """HDF5. The dataset is chosen explicitly, or it is the only 3-D one; ambiguity is an error
    rather than a guess, because picking the wrong dataset yields a volume that reads fine."""
    import h5py
    with h5py.File(p, "r") as f:
        found = []
        f.visititems(lambda n, o: found.append(n) if isinstance(o, h5py.Dataset) and o.ndim == 3 else None)
        if dataset is None:
            if len(found) != 1:
                raise ReadError("HDF5 has %d 3-D datasets, specify one of: %s"
                                % (len(found), ", ".join(found) if found else "(none)"))
            dataset = found[0]
        d = f[dataset]
        vol = d[()]
        sp = None
        for k in ("spacing", "element_size_um", "voxel_size", "resolution"):
            if k in d.attrs:
                v = np.asarray(d.attrs[k], float).ravel()
                if v.size == 3:
                    sp = tuple(v * (1e-3 if "um" in k else 1.0))
                    break
    return vol, sp, {"hu_calibrated": False, "spacing_source": "hdf5" if sp else "none",
                     "dataset": dataset, "datasets": found}


# --------------------------------------------------------------------------------- grid
def to_isotropic(vol: np.ndarray, spacing, iso: float = 0.15, order: int = 1,
                 max_voxels: int | None = None) -> np.ndarray:
    """Resample to the isotropic grid the whole pipeline is defined on.

    GRID CONVENTION. scipy's default grid_mode=False aligns the FIRST AND LAST VOXEL CENTRES, so the
    output's true spacing is sp*(N-1)/(M-1) rather than `iso` -- while everything downstream counts
    volume as 0.15**3 per voxel. Measured on a 27.000 mm object at 0.9 mm, the default returns
    27.300 mm (+1.1% per axis, +1.3% in volume, and worse on coarser inputs). grid_mode=True aligns
    voxel EDGES and returns 27.000 mm exactly, which is what sitk's resample did in the original
    pipeline, so that is what is used here.

    SIZE GUARD. ndi.zoom allocates the whole output up front. The zoom factor is (spacing/iso) per
    axis, so a clinical CT at 3 mm slices asks for a 2.3 TiB array -- and inside a memory-limited
    container the allocation SUCCEEDS and the process is OOM-killed while the pages fault in,
    taking every other session on the worker with it and printing nothing. Refusing early, with a
    message naming the voxel size, is the whole difference between a bad upload and a dead app.

    Linear interpolation, not nearest: this is a continuous attenuation image, and the fat band is
    applied after a 3x3x3 median filter, so interpolation noise is handled downstream.
    """
    sp = np.asarray(spacing, float)
    if sp.size != 3 or not np.all(np.isfinite(sp)) or np.any(sp <= 0):
        raise ReadError("voxel size must be three positive numbers, got %r" % (spacing,))
    if np.allclose(sp, iso, rtol=1e-4):
        return vol                                   # the common case stays bit-exact
    factors = sp / iso
    out_shape = np.round(np.asarray(vol.shape) * factors).astype(np.int64)
    n_out = int(np.prod(np.maximum(out_shape, 1)))
    if max_voxels is not None and n_out > max_voxels:
        raise ReadError(
            "resampling this %s volume from %.4f x %.4f x %.4f mm to %.2f mm would produce "
            "%s voxels (%.1f GB), past the %.0f M this app will allocate. Check the voxel size -- "
            "if it is right, this scan is larger than CTAdipo can process."
            % ("x".join(str(int(x)) for x in vol.shape), sp[0], sp[1], sp[2], iso,
               "x".join(str(int(x)) for x in out_shape), n_out * 4 / 1e9, max_voxels / 1e6))
    from scipy import ndimage as ndi
    return ndi.zoom(vol, factors, order=order, mode="grid-constant", grid_mode=True)
