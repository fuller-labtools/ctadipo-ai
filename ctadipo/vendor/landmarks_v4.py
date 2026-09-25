#!/usr/bin/env python3
r"""LANDMARK DETECTORS, iteration 3. The broken signal is gone, not tuned.

WHAT THE AUDIT ESTABLISHED ABOUT ITERATION 2, with numbers:
  spine_connected_width kept the component with the SMALLEST lateral extent and reported that.
  Measured over 11,554 non-zero slices on the 24-mouse ladder: median 0.90 mm, 74% of slices under
  1.50 mm -- a mouse lumbar vertebral body is 2.5-3.5 mm. It was measuring a 4-10 voxel sliver.
  The consequence is deterministic, not noisy: the THORAX read NARROWER than the lumbar spine on
  22 of 24 mice, so argmin was ATTRACTED to the ribcage. The anchor never landed on a lumbar slice
  (bone components at the anchor: 4-33, median 13). And zeros from dropped slices won the argmin
  outright on 4 of 24.

  It is deleted rather than fixed: no threshold repairs an inverted sign.

WHAT REPLACES IT. The all-bone lateral extent per slice, which the audit measured on the same
slices as thorax 18.52 mm vs lumbar 2.25 mm -- the correct sign, on 20 of 20 checked. Dropped
slices are NaN, never 0, so a detector failure cannot win a minimum.

THE PELVIS IS RETURNED AS TWO EDGES, NOT ONE POSITION. "All fat caudal of the pelvis" is ambiguous
by 35-40 percentage points of whole-body SUBQ depending on which edge is meant, and the femur is
LATERAL to the pelvis rather than caudal to it, so no single plane separates leg fat from flank
fat. Both edges are reported and the choice is the reviewer's, made on rendered images.
"""
import numpy as np
from scipy import ndimage as ndi

ISO = 0.15


MAD_FLOOR = 1.0        # below this the MAD carries no scale information
BONE_PCTL = 99.0       # fallback threshold: the top 1% of in-body HU


def bone_mask(raw, body):
    """Bone as a robust upper outlier of the in-body HU distribution.

    THE MAD CAN COLLAPSE TO ZERO, and when it does this returns a solid white blob. On scan 634,
    54.8% of in-body voxels are exactly 0 HU, so median = 0 and MAD = 0; the threshold
    med + 8*1.4826*MAD becomes 0 HU and 23.3% of the body is called bone, which is unreviewable --
    the reviewer cannot see a skeleton to place landmarks on.

    A percentile is used instead when the MAD carries no scale. The percentile is taken over the
    NON-ZERO voxels: a MAD of zero means the distribution is dominated by exact zeros, so a
    percentile over the raw distribution is a percentile of the padding, not of tissue. On 634
    (54.8% exact zeros) p99 of everything is the 97.8th percentile of the real body and labels
    2.21% of tissue as bone against the cohort median of 1.5%.

    SCOPE, stated accurately. This is a GUARD against a future degenerate scan, not a live code
    path: the only scan in the cohort that ever triggered it (634) was subsequently EXCLUDED as
    unusable, because its mask is 54.8% blank slab and no threshold fixes that. Making it render
    was the wrong response to it, and this docstring previously argued for the fallback using a
    misread number -- "the cohort minimum MAD is 69.6" -- when the true minimum is 0.0 (634 itself)
    or 51.2 excluding it. The margin to the next scan is 51x, not 70x, and the 31 currently-excluded
    scans were never measured at all, so "provably inert" holds only for the 1186 usable ones.
    """
    v = raw[body].astype(np.float32)
    med = float(np.median(v)); mad = float(np.median(np.abs(v - med)))
    if mad < MAD_FLOOR:
        nz = v[v != 0]
        thr = float(np.percentile(nz if nz.size else v, BONE_PCTL))
    else:
        thr = med + 8 * 1.4826 * mad
    return (raw > thr) & body


def bone_profiles(bone, z0, z1):
    """All-bone descriptors per slice. NaN where there is no bone, never 0.

    Zero is a legitimate value of a width, so using it as 'missing' means a slice with no bone
    becomes the global minimum -- which is how iteration 2's anchor landed on empty slices on 4 of
    24 mice. NaN plus nan-aware reductions removes that failure entirely.
    """
    nz = bone.shape[0]
    width = np.full(nz, np.nan)
    area = np.full(nz, np.nan)
    comp = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        b = bone[z]
        if not b.any():
            continue
        cols = np.where(b.any(axis=0))[0]
        width[z] = (cols.max() - cols.min() + 1) * ISO
        area[z] = float(b.sum())
        comp[z] = float(ndi.label(b)[1])
    return dict(width=width, area=area, comp=comp)


def _smooth_nan(a, k):
    """Moving average that ignores NaN, so gaps do not propagate or win comparisons."""
    v = np.where(np.isfinite(a), a, 0.0)
    m = np.isfinite(a).astype(float)
    num = ndi.uniform_filter1d(v, k)
    den = ndi.uniform_filter1d(m, k)
    out = np.full_like(a, np.nan, dtype=float)
    ok = den > 0.2
    out[ok] = num[ok] / den[ok]
    return out


def lumbar_trough(width, z0, z1):
    """The waist: minimum all-bone lateral extent between the pelvic and thoracic maxima.

    Restricted to the middle 70% only so the snout and the tail tip cannot masquerade as the waist.
    That is a sanity bound on a MINIMUM -- no landmark is ever returned from it, which is the
    failure that killed iteration 1's detectors.
    """
    w = _smooth_nan(width, 15)
    n = z1 - z0 + 1
    a, b = z0 + int(0.15 * n), z0 + int(0.85 * n)
    seg = w[a:b]
    if not np.isfinite(seg).any():
        return (z0 + z1) // 2
    return a + int(np.nanargmin(seg))


def pelvis_edges(bone, width, zt, z0, z1, head_at):
    """The pelvic girdle's CRANIAL and CAUDAL edges, in z.

    Found as the sustained rise and fall of all-bone lateral extent caudal of the lumbar trough.
    Both are returned because "below the pelvis" is ambiguous between them by 35-40 percentage
    points of whole-body SUBQ, and that is an anatomical decision, not a coding one.
    """
    lo, hi = (z0, zt) if head_at == "high" else (zt, z1)
    w = _smooth_nan(width, 9)[lo:hi + 1]
    if not np.isfinite(w).any():
        return None, None
    base = float(np.nanpercentile(w, 20))
    peak = float(np.nanmax(w))
    if not np.isfinite(base) or not np.isfinite(peak) or peak <= base:
        return None, None
    thr = base + 0.40 * (peak - base)
    on = np.where(np.isfinite(w) & (w >= thr))[0]
    if not len(on):
        return None, None
    # contiguous run containing the maximum, so a stray wide slice cannot define an edge
    pk = int(np.nanargmax(w))
    lo_i = pk
    while lo_i - 1 >= 0 and np.isfinite(w[lo_i - 1]) and w[lo_i - 1] >= thr:
        lo_i -= 1
    hi_i = pk
    while hi_i + 1 < len(w) and np.isfinite(w[hi_i + 1]) and w[hi_i + 1] >= thr:
        hi_i += 1
    a, b = lo + lo_i, lo + hi_i
    # cranial edge is the one nearer the head, in this scan's z sense
    return (b, a) if head_at == "high" else (a, b)


def pelvic_brim(bone, width, zt, z0, z1, head_at):
    """ORANGE: the FRONT OF THE PELVIC BONE -- the brim, not the front of the limbs.

    Walking from the lumbar trough toward the tail, the first sustained rise in all-bone lateral
    extent above the lumbar baseline. Cranial of the pelvis there is only vertebral column, so the
    step up is large and it is the pelvis that causes it -- the femur is further caudal and cannot
    define this edge.

    This replaces pelvis_edges, which returned the two ends of a whole wide-bone RUN. That run
    spans the hemipelvis and the hindlimb fused together (measured: 39.6 mm of extent across 40
    separate fragments on scan D), so its caudal edge was somewhere along the leg and its
    cranial edge was the trough itself on ~8% of scans.
    """
    lo, hi = (z0, zt) if head_at == "high" else (zt, z1)
    w = _smooth_nan(width, 9)[lo:hi + 1]
    if not np.isfinite(w).any():
        return None
    base = float(np.nanpercentile(w, 20))
    peak = float(np.nanmax(w))
    if not np.isfinite(base) or not np.isfinite(peak) or peak <= base:
        return None
    thr = base + 0.35 * (peak - base)
    idx = np.where(np.isfinite(w) & (w >= thr))[0]
    if not len(idx):
        return None
    # the edge NEAREST the trough, i.e. the cranial end of the pelvic mass
    return lo + (int(idx.max()) if head_at == "high" else int(idx.min()))


def _lateral_pair_end(bone, z0, z1, head_at, cranial, min_sep_px=20, min_vox=12, run_len=6):
    """The far end of the PAIRED LIMB bones, walking inward from one end of the animal.

    Both limb landmarks are the same shape of problem: find where two well-separated bone objects
    sit either side of the midline. That is the limbs, and it is not the tail (a single thin
    midline object) nor the spine (also midline). Requiring the pair to persist for `run_len`
    slices stops one noisy slice from defining a landmark.

    cranial=False walks from the TAIL end and returns the caudal end of the hindlimbs.
    cranial=True  walks from the HEAD end and returns the cranial end of the forelimbs -- the
                  shoulder-blade region, which is what the reviewer defined the blue line as.
    """
    if head_at == "high":
        order = range(z0, z1 + 1) if not cranial else range(z1, z0 - 1, -1)
    else:
        order = range(z1, z0 - 1, -1) if not cranial else range(z0, z1 + 1)
    order = list(order)
    run = 0
    for i, z in enumerate(order):
        b = bone[z]
        if not b.any():
            run = 0
            continue
        lab, n = ndi.label(b)
        if n >= 2:
            sizes = np.bincount(lab.ravel())[1:]
            big = np.where(sizes >= min_vox)[0] + 1
            if len(big) >= 2:
                xs = [float(np.where((lab == k).any(axis=0))[0].mean()) for k in big]
                if max(xs) - min(xs) >= min_sep_px:
                    run += 1
                    if run >= run_len:
                        return order[max(i - run_len + 1, 0)]
                    continue
        run = 0
    return None


def hindlimb_end(bone, z0, z1, head_at):
    """RED: the caudal end of the hindlimb bones, tail excluded.

    Defined by the reviewer as "right at the end of the back limbs". The tail is excluded by
    construction: it is a single thin midline object, and this looks for a SEPARATED PAIR either
    side of the midline, which only the limbs produce. The previous detector took the caudal edge
    of a total-bone-width run, which the tail keeps alive, so it landed 4.5-16.3 mm cranial of the
    limb end depending on posture.
    """
    return _lateral_pair_end(bone, z0, z1, head_at, cranial=False)


def shoulder_blades(bone, rib, z0, z1, head_at):
    """BLUE: the SHOULDER BLADES -- the front of the forelimbs at the bend.

    Searched only CRANIAL OF THE LAST RIB, and taken as the slice where the paired lateral bone is
    most widely separated. The scapulae plus humeri are the broadest paired structure in that
    region; the skull and the forepaws are not.

    Two earlier versions failed here and both failed the same way -- by having no anchor:
      * a bare argmax of bone area went to the densest slice, which is the skull on head-at-low
        scans;
      * walking inward from the head and taking the FIRST separated pair returned the frame edge on
        every one of 24 test mice, because the jaw and the forepaw tips qualify immediately.
    Taking a maximum WITHIN a bounded region cannot return the boundary of that region unless the
    signal genuinely peaks there.
    """
    if rib is None:
        return None
    lo, hi = (rib, z1) if head_at == "high" else (z0, rib)
    if hi <= lo:
        return None
    best, bestsep = None, -1.0
    for z in range(lo, hi + 1):
        b = bone[z]
        if not b.any():
            continue
        lab, n = ndi.label(b)
        if n < 2:
            continue
        sizes = np.bincount(lab.ravel())[1:]
        big = np.where(sizes >= 25)[0] + 1
        if len(big) < 2:
            continue
        xs = [float(np.where((lab == k).any(axis=0))[0].mean()) for k in big]
        sep = max(xs) - min(xs)
        if sep > bestsep:
            best, bestsep = z, sep
    return best


def last_rib(comp, zt, z0, z1, head_at):
    """The caudal-most rib: walking from the lumbar trough toward the head, the first sustained
    rise in bone component count. Anchored at the trough, so it cannot return a window edge."""
    lo, hi = (zt, z1) if head_at == "high" else (z0, zt)
    c = _smooth_nan(comp, 7)[lo:hi + 1]
    if not np.isfinite(c).any():
        return None
    base = float(np.nanpercentile(c, 15)); peak = float(np.nanmax(c))
    if peak <= base:
        return None
    thr = base + 0.30 * (peak - base)
    order = range(len(c)) if head_at == "high" else range(len(c) - 1, -1, -1)
    run = 0
    for i in order:
        if np.isfinite(c[i]) and c[i] >= thr:
            run += 1
            if run >= 10:
                return lo + (i - 10 if head_at == "high" else i + 10)
        else:
            run = 0
    return None
