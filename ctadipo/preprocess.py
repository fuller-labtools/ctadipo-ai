r"""The preprocessing CTAdipo must reproduce bit-for-bit, and the reasons each step is what it is.

Every function here is a port of a step in the locked pipeline, not a reinterpretation of it. Where
a constant looks arbitrary it is because it was measured, and the measurement is recorded beside it.
The regression test that matters is not "does this look right" but "does the app reproduce
_doz_long_v3.csv on scans that are already derived", so nothing here may drift.

THE ORDER IS LOAD-BEARING:
    animal_mask  ->  normalise  ->  (nnU-Net)  ->  body/cavity  ->  fat_mask  ->  depots
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

ISO = 0.15
AIR_THR = -500.0

# Two fat bands, chosen by the scan's own air percentile. The HU scale is NOT calibrated in this
# data: the air peak runs from -773 to -1424 HU across the two scanners, and soft tissue reads +41 HU
# in the lean 3M scans against -233 in the obese Aging ones. A single fixed band cannot serve both.
FAT_LO, FAT_HI = -491.0, -118.0        # narrow scanner, band from the peak midpoint
AGING_LO, AGING_HI = -490.0, -200.0    # wide scanner
SCALE_AIR = -1050.0                    # air_p1 above this means the wide (AGING) scanner

CAVITY = 3                             # nnU-Net class: 0 bg, 1 outer_body, 2 abdominal_wall, 3 cavity


def keep_largest3d(m: np.ndarray) -> np.ndarray:
    """Largest 3-D connected component. Drops hardware that is DETACHED from the animal.

    It cannot touch hardware FUSED to the animal -- on this rig the breathing pad is fused through
    the contact patch on some scans, and its dark interior is labelled cavity, i.e. counted as
    visceral fat. That case is handled by hand-drawn geometry in the original pipeline, which is
    keyed to scanner and crop and therefore cannot transfer to someone else's scanner. CTAdipo
    handles it by showing the user the mask instead of pretending it is solved.
    """
    lbl, n = ndi.label(m)
    if n <= 1:
        return m
    sz = np.bincount(lbl.ravel())
    sz[0] = 0
    return lbl == int(sz.argmax())


def animal_mask(raw: np.ndarray) -> np.ndarray:
    """Largest 3-D component above -500 HU, holes filled.

    3-D connectivity is the entire point. The sample holder can be the WIDEST object in any single
    slice, but in 3-D it is a separate object. A per-slice largest-object rule instead selects the
    hole-filled holder on wide fields of view, and that is not hypothetical: it put the
    normalisation mean 1.808 SD off on wide-field scans and translated 171 predictions by 1.5-1.8 mm.

    Hole filling is equally load-bearing, for the opposite reason: the lungs sit below -500 HU, so
    without filling they are excluded from the animal -- and the lungs are exactly what the thoracic
    landmarks are defined on.
    """
    lbl, n = ndi.label(raw > AIR_THR)
    if n == 0:
        raise RuntimeError("no body component above %.0f HU -- is this a CT volume?" % AIR_THR)
    sz = np.bincount(lbl.ravel())
    sz[0] = 0
    return ndi.binary_fill_holes(lbl == int(sz.argmax()))


def normalise(raw: np.ndarray, mask: np.ndarray | None = None):
    """The z-scoring nnU-Net actually sees. Returns (volume, mean, sd).

    dataset.json declares "noNorm", so nnU-Net applies NO normalisation of its own and THIS is the
    normalisation. Statistics come from the animal, never from a per-slice body mask -- see
    animal_mask for what happened when they did. Verified on 8056 hand-placed points: mouse dots
    inside the mask went 89.0% -> 98.8% and narrow-field predictions stayed bit-identical
    (dice 0.9996-0.9998).
    """
    a = animal_mask(raw) if mask is None else mask
    v = raw[a].astype(np.float32)
    m, sd = float(v.mean()), float(v.std()) + 1e-6
    return ((raw - m) / sd).astype(np.float32), m, sd


def scanner_band(raw: np.ndarray):
    """(air_p1, fat_lo, fat_hi) for this scan. The band follows the scan, not an assumed calibration."""
    air_p1 = float(np.percentile(raw, 1))
    lo, hi = (FAT_LO, FAT_HI) if air_p1 < SCALE_AIR else (AGING_LO, AGING_HI)
    return air_p1, lo, hi


def fat_mask(raw: np.ndarray, body: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """The fat band, applied to a 3x3x3 MEDIAN-FILTERED volume rather than the raw one.

    The band is ~290 HU wide and this data's noise runs at a median SD of ~153 HU inside the body, so
    grain alone moves individual voxels across much of it -- inventing and destroying fat at once.
    Measured against 709 hand-placed fat dots:
        raw       576/709 kept (81.2%), 126 tissue dots wrongly called fat
        median3   645/709 kept (91.0%),  79 tissue dots wrongly called fat
    It recovers 69 real dots and rejects 47 false ones, so it is strictly better on both counts. A
    coherence filter dropping fat components under 50 voxels was tested and REJECTED: it keeps fewer
    dots than doing nothing, because it deletes genuine small fat structures.

    The median filter is used ONLY for the band. air_p1, the lung logic and everything else read the
    unfiltered volume.
    """
    rf = ndi.median_filter(raw, size=3)
    return (rf >= lo) & (rf <= hi) & body


def compartments(pred: np.ndarray):
    """(body, cavity) from the nnU-Net prediction, each reduced to its coherent component.

    keep_largest3d on the cavity removes streak-artefact speckle. VAT and SUBQ are separated by this
    ONE boundary, so a cavity that has fragmented is the single most consequential failure mode in
    the whole pipeline -- the app checks it explicitly rather than assuming it.
    """
    return keep_largest3d(pred > 0), keep_largest3d(pred == CAVITY)


WALL_MM3_OK = 70.0          # below this the VAT/SUBQ SPLIT is untrustworthy (the total is fine)


def wall_quality(pred, body, cav):
    """The original pipeline's abdominal-wall QC, not a lookalike. Returns (wall_mm3, edge_on_wall).

    VAT and SUBQ are separated by ONE boundary, the cavity, and the cavity is only anchored where
    the model predicts abdominal wall. The wall is a few voxels thick, so on a noisy scan it can
    vanish and take the VAT/SUBQ distinction with it while both totals still look entirely
    plausible -- which is why this is measured and shown rather than assumed.

    The threshold is not arbitrary. Against the reviewer's verdicts the wall volume separates them
    with NO overlap: 72-1174 mm3 on scans he called good, 28-67 mm3 on scans he called bad, with
    45-65% of the cavity edge against wall on the good ones versus 6-17% on the bad.

    An earlier version of this file computed something else entirely -- the fraction of
    cavity-bearing slices carrying any wall voxel -- which disagrees in both directions: a thin but
    widespread wall passes it and fails this, and a thick wall over a short span does the reverse.
    """
    wall = (pred == 2) & body
    wall_mm3 = float(wall.sum()) * (ISO ** 3)
    edge = cav & ~ndi.binary_erosion(cav)
    near = ndi.binary_dilation(edge, iterations=2) & ~cav
    edge_on_wall = float((near & wall).sum()) / max(float(near.sum()), 1.0)
    return wall_mm3, edge_on_wall
