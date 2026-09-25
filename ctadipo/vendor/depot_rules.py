#!/usr/bin/env python3
r"""SUB-DEPOT RULES — candidate ways of partitioning fat into named depots, for comparison.

NOTHING HERE RE-DERIVES ORIENTATION OR GEOMETRY. The craniocaudal direction comes from
scan_decisions.direction and the four boundaries come from landmark_decisions.csv, both of which
are 100% human-reviewed. That is not a stylistic choice: the previous sub-depot code recomputed
orientation from a bone-mass heuristic (`cranial_low`) and disagreed with the reviewer's verdict on
185 of 1188 scans (15.6%), including 5 scans where the reviewer had EXPLICITLY corrected it. The
symptom was invisible because conservation still held exactly -- the depots summed to the right
total while anterior and gluteofemoral were swapped. Every depot in this file is defined against
the reviewed frame or it is not defined at all.

THE FRAME
  cc(z)     craniocaudal position, 0 at the tail end and 1 at the head end, built from `direction`.
            The four landmarks are, by construction, ordered
              pelvic_caudal < pelvic_cranial < last_rib < shoulder
            in cc, whatever the raw z direction is.
  dorsal    per slice, the side of the spine centroid the voxel is on. The spine is found in a
            narrow central x band sized against the vertebral body (+/- 1.75 mm), NOT as a fraction
            of the animal's width -- a fractional band was 8.7-9.1 mm wide, contained a median of 14
            disjoint bone components, and tracked limb posture rather than the vertebra.
  wall      distance from the cavity boundary, for depots defined by what they lie against.

WHAT IS AND IS NOT DECIDABLE FROM THESE SCANS is recorded in DEPOT_LITERATURE.md. The depots below
are the ones judged feasible; the uncertain five (mesenteric vs perigonadal separation, perirenal
proxy, mediastinal, preperitoneal) are implemented so they can be LOOKED AT, and are marked
`provisional=True` so nothing downstream treats them as settled.

DEPOTS MAY OVERLAP AND NEED NOT SUM TO THE TOTAL. That is deliberate and is the reviewer's
instruction: the point is to measure as many literature-defined depots as possible, including
traditionally defined ones that overlap, so the scheme is not self-defined. Conservation is
therefore checked only within the PARTITION set, never across the whole output.
"""
import sys

import numpy as np
from scipy import ndimage as ndi

# VENDORED from the research tree. Executable code is unchanged; the only edits are the two
# imports below and the redaction of cohort scan identifiers in comments (scan A, scan B, ...),
# verified by comparing this file's abstract syntax tree against the original.
from . import cohort_derive_shim as CD                                    # noqa: E402
from .landmarks_v4 import bone_mask                                         # noqa: E402

ISO = 0.15
VOX = ISO ** 3

# --------------------------------------------------------------------------------------------------
# THE LANDMARKS COME FROM THE REVIEWED ORGAN PASS, NOT THE EARLIER SKELETAL LADDER.
#
# This file used to read landmark_decisions.csv -- pelvic_caudal_z / pelvic_cranial_z / last_rib_z /
# shoulder_z, produced by the old bone detectors. That file was QUARANTINED on 2026-09-18 and is
# no longer read, because it is NOT interchangeable with the hand-placed organ
# landmarks. Measured difference between the two, in slices of 0.15 mm:
#
#     bladder_z       vs pelvic_caudal_z     median 46 slices (6.9 mm),   0.7% identical
#     kidney_caudal_z vs pelvic_cranial_z    median 35 slices (5.3 mm),   0.0% identical
#     diaphragm_z     vs last_rib_z          median 16 slices (2.4 mm),   2.3% identical
#     lung_apex_z     vs shoulder_z          median -48 slices (-7.2 mm), 0.0% identical
#
# Every band would have been cut 2.4-7.2 mm from where it was placed, on essentially every scan, and
# the depots would have looked entirely plausible. That is why the old partitions are not merely
# unused but removed: the failure mode is that they get included by accident.
#
# The aliasing is SUBDEPOT_PLAN.md Amendment 3:
#     bladder -> pelvic_caudal        kidney_caudal -> pelvic_cranial
#     diaphragm -> last_rib           lung_apex -> shoulder
# with kidney_cranial carried separately as an interior point (it anchors perirenal).
#
# Coverage is 1186/1186 on all five: 0 blank, 0 not_visible, 0 flagged.
import pathlib                                                             # noqa: E402

ORGAN_LANDMARKS = pathlib.Path("organ_decisions.csv")  # vendored: load_landmarks is never used by the app, which supplies landmarks from the model
ALIAS = {"pelvic_caudal": "bladder_z", "pelvic_cranial": "kidney_caudal_z",
         "last_rib": "diaphragm_z", "shoulder": "lung_apex_z"}
_LM = {}


def load_landmarks(aid):
    """The four ladder slots plus kidney_cranial for one scan, as raw z indices.

    Raises rather than guessing: every scan in the cohort was reviewed, so a missing value means the
    table and the volume disagree, and that must surface rather than be filled in.
    """
    if not _LM:
        import csv
        with open(ORGAN_LANDMARKS, newline="") as fh:
            for r in csv.DictReader(fh):
                _LM[r["Animal"]] = r
    r = _LM.get(str(aid))
    if r is None:
        raise KeyError(f"{aid}: not in {ORGAN_LANDMARKS.name}; every cohort scan was landmarked, so "
                       f"this is either a misnamed scan or one outside the reviewed set")
    out = {}
    for slot, col in ALIAS.items():
        v = r.get(col, "")
        if v in ("", None):
            raise ValueError(f"{aid}: {col} is blank in {ORGAN_LANDMARKS.name}")
        out[slot] = float(v)
    kc = r.get("kidney_cranial_z", "")
    out["_kidney_cranial"] = float(kc) if kc not in ("", None) else None
    out["_direction"] = r.get("direction")
    # THE DORSOVENTRAL VERDICT IS REVIEWED TOO, and was being thrown away. frame() re-derived it
    # from spine-vs-cavity and refused wherever its own margin was thin -- which both aborted scans
    # the reviewer had confirmed (7 of the first 430 in the cohort depot pass) and, worse,
    # silently OVERRODE the reviewer on the 4 scans where model and human disagree. On those the model's verdict
    # swaps dorsolumbar with inguinal and retroperitoneal with perigonadal: the exact failure this
    # module exists to prevent, arrived at from the other direction. dorsal_is_low_human is filled
    # on all 1186 scans and dorsal_confirmed is 1 on 1173 of them.
    _TRUE = ("1", "True", "true", "yes", "Y", "y")
    dh = r.get("dorsal_is_low_human", "")
    out["_dorsal_is_low"] = (str(dh).strip() in _TRUE) if dh not in ("", None) else None
    out["_dorsal_confirmed"] = str(r.get("dorsal_confirmed", "")).strip() in _TRUE
    return out
# --------------------------------------------------------------------------------------------------
MASS_FRAC = 0.25              # a component this large relative to the biggest is a paired pad
INTERSCAP_CC = 0.06           # +/- 6% of body length about the shoulder line
SPINE_HALF_MM = 1.75          # +/- about the midline: a mouse vertebral body is 2.5-3.5 mm
DORSAL_MIN_AGREE = 0.90       # fraction of cavity-bearing slices that must agree on which side is dorsal
DORSAL_MIN_MARGIN_MM = 1.0    # median spine-to-cavity separation below which the verdict is not trusted

# The depots each rule produces. `partition` members must be disjoint and sum to their compartment;
# `overlap` members are additional literature depots that may overlap anything.
VAT_PARTITION = ["perigonadal", "mesenteric", "retroperitoneal", "thoracic_VAT"]
SUBQ_PARTITION = ["anterior_SUBQ", "head_neck_SUBQ", "dorsolumbar_SUBQ", "inguinal_SUBQ",
                  "gluteal_SUBQ", "hindlimb_SUBQ"]
# Depots that are NOT settled. DEPOT_LITERATURE.md marks these as needing an image check before
# they are believed; the flag is emitted with the volumes so nothing downstream treats them as
# established. The previous docstring claimed this existed and it did not.
PROVISIONAL = {"mesenteric", "perirenal_proxy", "mediastinal", "preperitoneal", "thoracic_VAT"}
OVERLAP = ["posterior_SUBQ_whole", "interscapular", "perirenal_proxy",
           "mediastinal", "preperitoneal", "lumbar_window_VAT", "lumbar_window_SUBQ"]


# ---------------------------------------------------------------------------------------------
def frame(aid, landmarks, direction, masks=None, allow_uncertain_dorsal=False,
          dorsal_is_low=None):
    """Build the canonical frame for one scan. `landmarks` is the reviewed dict of raw z values.

    `allow_uncertain_dorsal` EXISTS ONLY FOR THE REVIEW VIEWER, and no depot rule may pass it.
    The dorsoventral verdict is normally a hard gate: below DORSAL_MIN_AGREE the function refuses to
    guess, because getting it wrong silently swaps retroperitoneal with perigonadal and dorsolumbar
    with inguinal, and the swap is invisible in the totals. That gate must stay closed for anything
    that MEASURES a depot.

    It is wrong for anything that merely LOOKS at the animal. In organ_review the verdict decides
    only which coronal panel is captioned "dorsal" and which "ventral"; the landmark the reviewer
    places is a craniocaudal position, unaffected either way, and the reviewer can move both planes
    by hand regardless. Refusing there would abort the tool on a scan the human could perfectly well
    review. With this flag the best-guess verdict is used and `dorsal_uncertain` is set in the
    returned frame so the caller can say so on screen.
    """
    M = masks or CD.derive_masks(aid)
    body_all, cav = M["body"], M["cav"]
    mouse = CD.keep_largest3d(body_all)
    raw = M["raw"]
    nz = mouse.shape[0]

    zb = np.where(mouse.any(axis=(1, 2)))[0]
    z0, z1 = int(zb.min()), int(zb.max())

    # cc: 0 at the TAIL end, 1 at the HEAD end, using the REVIEWED direction
    if str(direction).lower() == "high":
        cc = (np.arange(nz) - z0) / max(z1 - z0, 1)
    elif str(direction).lower() == "low":
        cc = (z1 - np.arange(nz)) / max(z1 - z0, 1)
    else:
        raise ValueError(f"{aid}: direction is {direction!r}, expected 'high' or 'low'")

    lm_cc = {}
    for k, v in landmarks.items():
        if v is None or not np.isfinite(v):
            raise ValueError(f"{aid}: landmark {k} is missing; every scan is reviewed, so this "
                             f"means the landmark table and the profile disagree")
        lm_cc[k] = float(cc[int(np.clip(v, 0, nz - 1))])
    order = ["pelvic_caudal", "pelvic_cranial", "last_rib", "shoulder"]
    vals = [lm_cc[k] for k in order]
    if not all(vals[i] < vals[i + 1] for i in range(3)):
        raise ValueError(f"{aid}: landmarks are not tail->head in cc: "
                         f"{dict(zip(order, np.round(vals, 3)))}. The reviewed direction and the "
                         f"reviewed landmarks disagree; refusing to guess which is right.")

    # ---- FAT IS RESTRICTED TO THE ANIMAL --------------------------------------------------------
    # cohort_derive builds fat against pred>0, which INCLUDES THE SCANNER HOLDER. Cutting depots
    # from that assigned holder fat to real depots: 36.5% of hindlimb_SUBQ on scan A
    # lay outside the animal, and that depot's centre of mass was nine body-thicknesses outside the
    # mouse. Conservation could not see it because the holder voxels were absorbed by the
    # open-ended terminal slabs. The excluded volume is reported, not silently dropped.
    vat = M["vat"] & mouse
    sat = M["sat"] & mouse
    holder_mm3 = float(((M["vat"] | M["sat"]) & ~mouse).sum()) * VOX

    bone = bone_mask(raw, mouse)

    # ---- THE VERTEBRAL COLUMN, PER SLICE --------------------------------------------------------
    # x_mid used to be the midpoint of the animal's whole x-extent -- splayed limbs, tail and head
    # included -- which sits up to 3.80 mm off the vertebral column against a band half-width of
    # 1.875 mm. On scan B the band missed the spine entirely (bone on 22 of ~180 lumbar slices),
    # the interpolator then ramped across a 66-slice gap, and the dorsoventral axis INVERTED:
    # 54.6% of the lumbar cavity was called dorsal, against 0-12% on every other mouse.
    # The midline is now taken from the BONE itself in the lumbar band, where the vertebra is the
    # dominant midline structure.
    lum = (cc >= lm_cc["pelvic_cranial"]) & (cc < lm_cc["last_rib"])
    lb = bone & lum[:, None, None]
    if lb.any():
        # THE PEAK OF THE x-HISTOGRAM, not the mean. A voxel-weighted MEAN over all lumbar bone is
        # dragged off the midline by the ilium, the last ribs and asymmetric femora: measured, it
        # missed the vertebral column by more than 1 mm on 7 of 30 scans and by up to 2.41 mm --
        # wider than the +/-1.75 mm search band, so on those the band did not contain the spine at
        # all. The vertebral column is the only bone present on EVERY lumbar slice, so it dominates
        # the histogram even where the mean does not.
        h = np.bincount(np.argwhere(lb)[:, 2], minlength=mouse.shape[2]).astype(float)
        h = ndi.uniform_filter1d(h, 9)
        x_mid = float(np.argmax(h))
    else:
        xs = np.where(mouse.any(axis=(0, 1)))[0]
        x_mid = float((xs.min() + xs.max()) / 2)

    half = int(round(SPINE_HALF_MM / ISO))
    lo_x, hi_x = max(int(x_mid - half), 0), int(x_mid + half) + 1
    spine_y = np.full(nz, np.nan)
    n_obs = 0
    for z in range(z0, z1 + 1):
        b = bone[z][:, lo_x:hi_x]
        if not b.any():
            continue
        ys = np.argwhere(b)[:, 0]
        # DORSAL-MOST cluster, not the mean of all midline bone. In the thorax the band also
        # contains the STERNUM (midline, 13-20 mm ventral), so the mean sat at mid-chest and made
        # `interscapular` the dorsal half of the whole thoracic shell -- 74% of anterior_SUBQ on
        # 674. A percentile toward the dorsal side tracks the vertebra through the ribcage.
        spine_y[z] = float(np.percentile(ys, 10 if _dorsal_low_guess(cav, mouse, z) else 90))
        n_obs += 1

    # ---- WHICH SIDE IS DORSAL: decided against the CAVITY, per slice ----------------------------
    # The abdominal cavity is ventral to the vertebral column in every mouse. That is an
    # anatomical invariant and it does not depend on how fat the animal is -- unlike the previous
    # rule (spine vs BODY CENTROID), whose margin grew from 0.26 mm in the leanest mouse to 8.73 mm
    # in the fattest, which is why every dorsally-defined depot drifted with adiposity.
    votes, margins = [], []
    for z in range(z0, z1 + 1):
        c = cav[z]
        if not c.any() or not np.isfinite(spine_y[z]):
            continue
        cy = float(np.argwhere(c)[:, 0].mean())
        votes.append(spine_y[z] < cy)
        margins.append(abs(spine_y[z] - cy) * ISO)
    if not votes:
        raise ValueError(f"{aid}: no slice has both a cavity and a detectable spine; "
                         f"the dorsoventral axis cannot be established")
    votes = np.array(votes); margins = np.array(margins)
    frac = float(votes.mean())
    measured_is_low = bool(frac >= 0.5)
    agree = frac if measured_is_low else 1 - frac
    # THE REVIEWED VERDICT WINS WHERE THERE IS ONE. The measurement above is still computed, and
    # still reported as dorsal_agree / dorsal_margin_mm, so a disagreement stays visible -- but it
    # no longer decides, and it no longer aborts a scan the reviewer has confirmed.
    dorsal_from_human = dorsal_is_low is not None
    if dorsal_from_human:
        dorsal_is_low = bool(dorsal_is_low)
        dorsal_disagrees = bool(dorsal_is_low != measured_is_low)
        dorsal_uncertain = False
    else:
        dorsal_is_low = measured_is_low
        dorsal_disagrees = False
        dorsal_uncertain = bool(agree < DORSAL_MIN_AGREE
                                or float(np.median(margins)) < DORSAL_MIN_MARGIN_MM)
    if dorsal_uncertain and not allow_uncertain_dorsal:
        raise ValueError(
            f"{aid}: dorsoventral axis is not confidently determined "
            f"({100*agree:.0f}% of {len(votes)} slices agree, median spine-to-cavity separation "
            f"{np.median(margins):.2f} mm). Refusing to guess -- a wrong verdict silently swaps "
            f"retroperitoneal with perigonadal and dorsolumbar with inguinal.")

    spine_y = _fill_nan(spine_y)
    ycent = np.full(nz, np.nan)
    for z in range(z0, z1 + 1):
        m = mouse[z]
        if m.any():
            ycent[z] = float(np.argwhere(m)[:, 0].mean())
    ycent = _fill_nan(ycent)

    M.pop("pred", None)          # not used after the masks are built; it is a full int volume
    return dict(aid=aid, M=M, mouse=mouse, cav=cav, vat=vat, sat=sat,
                bone=bone, nz=nz, z0=z0, z1=z1, cc=cc, lm=lm_cc, lm_z=landmarks,
                direction=str(direction).lower(), spine_y=spine_y, ycent=ycent,
                dorsal_is_low=dorsal_is_low, x_mid=x_mid, holder_mm3=holder_mm3,
                dorsal_agree=agree, dorsal_margin_mm=float(np.median(margins)),
                dorsal_uncertain=dorsal_uncertain,
                dorsal_from_human=dorsal_from_human, dorsal_disagrees=dorsal_disagrees,
                dorsal_measured_is_low=measured_is_low,
                spine_slices=n_obs, spine_slices_total=int(z1 - z0 + 1),
                _wall=None)


def _dorsal_low_guess(cav, mouse, z):
    """Cheap per-slice guess at which y direction is dorsal, used only to pick which tail of the
    midline bone distribution to take. The authoritative verdict is the cavity vote above."""
    c = cav[z]
    m = mouse[z]
    if not c.any() or not m.any():
        return True
    return float(np.argwhere(c)[:, 0].mean()) > float(np.argwhere(m)[:, 0].mean())


def _fill_nan(a):
    """Linear fill so a slice with no detectable spine does not create a hole in the frame."""
    i = np.arange(len(a))
    ok = np.isfinite(a)
    if ok.sum() < 2:
        return np.full_like(a, np.nan)
    return np.interp(i, i[ok], a[ok])


def slab(F, lo_key, hi_key):
    """Boolean per-slice mask for the cc band between two landmarks.

    THE TERMINAL SLABS ARE OPEN-ENDED (-inf / +inf), not clamped to [0, 1]. cc is normalised to the
    ANIMAL's z-extent, but `fat` is masked with pred>0, which includes the scanner holder, so fat
    voxels exist at slices outside the animal's own extent where cc < 0 or cc > 1. Clamping the
    ends to 0 and 1 silently dropped them: 13374 SUBQ voxels went unassigned on two test mice, and
    the partition looked complete because nothing counted the remainder.
    """
    lo = -np.inf if lo_key is None else F["lm"][lo_key]
    hi = np.inf if hi_key is None else F["lm"][hi_key]
    return (F["cc"] >= lo) & (F["cc"] < hi)


def dorsal_mask(F):
    """3-D boolean: True where a voxel is DORSAL of the spine centroid in its own slice."""
    nz, ny, nx = F["mouse"].shape
    yy = np.arange(ny)[None, :, None]
    sy = F["spine_y"][:, None, None]
    return (yy <= sy) if F["dorsal_is_low"] else (yy >= sy)


def wall_distance(F):
    """Distance in mm from the cavity boundary, inside the cavity. CACHED ON THE FRAME.

    This was recomputed on every call -- 5 times per animal across the three rules. Each call
    allocates a (3,nz,ny,nx) int32 index array plus a float64 copy: ~6.2 GiB transient on a
    540x529x529 AGING scan, which OOM-killed 10 of 24 test animals, and the bare `except` in
    depot_compare swallowed the MemoryError so the scan was silently dropped from both the figures
    and the results table.
    """
    if F.get("_wall") is None:
        # float32, not float64: on a 489^3 volume the float64 result alone is 356 MB, and the
        # frame already holds raw/pred/mouse/cav/vat/sat/bone. Scoring 19 rules per mouse ran the
        # machine out of memory after 13 of 30 animals. 0.15 mm precision needs nothing like
        # float64, and the saving is exactly half of the largest single array in the pipeline.
        F["_wall"] = (ndi.distance_transform_edt(F["cav"]) * ISO).astype(np.float32)
    return F["_wall"]


# ---------------------------------------------------------------------------------------------
# CANDIDATE RULES. Each takes a frame and returns {depot_name: bool 3-D mask}.
# They differ in WHICH SIGNAL separates the depots, not in cosmetic thresholds, so comparing them
# on images answers a real question rather than a tuning question.
# ---------------------------------------------------------------------------------------------

def rule_A_slabs(F):
    """A -- PURE LANDMARK SLABS. Craniocaudal position alone, plus dorsal/ventral for VAT.

    The simplest thing that could work, and the baseline the others must beat. Every boundary is a
    plane at a reviewed landmark, so it is completely reproducible and trivially explainable -- but
    it cannot separate two depots that occupy the same craniocaudal band (mesenteric from
    perigonadal), and it assigns limb fat by z rather than by belonging to the limb.
    """
    vat, sat = F["vat"], F["sat"]
    dor = dorsal_mask(F)
    Z = lambda m: m[:, None, None]

    caud   = Z(slab(F, None, "pelvic_caudal"))
    pelvic = Z(slab(F, "pelvic_caudal", "pelvic_cranial"))
    lumb   = Z(slab(F, "pelvic_cranial", "last_rib"))
    thorax = Z(slab(F, "last_rib", "shoulder"))
    cran   = Z(slab(F, "shoulder", None))

    d = {}
    # ---- VAT. All three boundaries use the SAME dorsal notion (the dorsal half of the cavity),
    # so the depots tile exactly. Mixing two definitions -- retroperitoneal by cavity, mesenteric
    # by spine centroid -- left 247464 voxels unassigned and 470060 double-assigned, which the
    # conservation check caught immediately.
    dh = _dorsal_half(F)
    d["retroperitoneal"] = vat & lumb & (wall_distance(F) < 1.5) & dh
    rest = vat & lumb & ~d["retroperitoneal"]
    # `caud` is included in perigonadal: the pad hangs into the caudal abdomen, and any cavity fat
    # caudal of the hindlimb line belongs with it. Omitting it left 87757 VAT voxels unassigned.
    d["perigonadal"]     = (vat & (caud | pelvic)) | (rest & Z(_caudal_half(F)))
    d["mesenteric"]      = rest & ~Z(_caudal_half(F))
    d["thoracic_VAT"]    = vat & (thorax | cran)
    # ---- SUBQ
    d["hindlimb_SUBQ"]    = sat & caud
    d["gluteal_SUBQ"]     = sat & pelvic
    d["dorsolumbar_SUBQ"] = sat & lumb & dor
    d["inguinal_SUBQ"]    = sat & lumb & ~dor
    # BOUNDED AT THE SHOULDER. Folding the head/neck into anterior made the depot mean different
    # things on different animals: the span cranial of the shoulder is 0-11.7% of body length, and
    # is zero on the 171 scans where the head is cropped out of frame. DEPOT_LITERATURE.md S4 rules
    # the cervical depot out as a cohort measure; it is reported separately, not silently merged.
    d["anterior_SUBQ"]    = sat & thorax
    d["head_neck_SUBQ"]   = sat & cran
    # ---- overlapping literature depots (NOT part of the partition)
    d["posterior_SUBQ_whole"] = sat & (caud | pelvic | lumb)
    # within +/- INTERSCAP_MM of the shoulder line, per DEPOT_LITERATURE.md S2 -- not the whole
    # dorsal half of the thoracic shell, which was 74% of anterior_SUBQ on 674
    _sh = F["lm"]["shoulder"]
    _near_sh = Z((F["cc"] >= _sh - INTERSCAP_CC) & (F["cc"] < _sh + INTERSCAP_CC))
    d["interscapular"]        = sat & dor & _near_sh
    d["perirenal_proxy"]      = d["retroperitoneal"] & Z(_cranial_third(F))
    d["mediastinal"]          = vat & thorax
    # the literature's abdominal window, so our depots can be read against Luu/Lubura/Judex
    d["lumbar_window_VAT"]    = vat & lumb
    d["lumbar_window_SUBQ"]   = sat & lumb
    d["preperitoneal"]        = vat & (wall_distance(F) < 1.0) & ~_dorsal_half(F)
    return d


def rule_B_wall(F):
    """B -- SLABS + DISTANCE FROM THE CAVITY WALL, for the visceral depots.

    Retroperitoneal fat lies AGAINST the dorsal body wall and mesenteric fat hangs FREE in the
    cavity interior; that is the anatomical distinction (Bagchi 2019, Cinti 2007), and it is a
    distance, not a plane. A craniocaudal slab cannot express it. SUBQ is unchanged from A, so any
    difference between A and B is attributable to the visceral rule alone.
    """
    d = rule_A_slabs(F)
    vat = F["vat"]
    dor = dorsal_mask(F)
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    wall = wall_distance(F)
    NEAR = 1.5                       # mm from the cavity boundary

    # AGAINST THE DORSAL BODY WALL, which is what the depot is anatomically -- not "dorsal of the
    # spine centroid", which is nearly empty because the mouse abdominal cavity barely extends
    # dorsal of the vertebrae: that definition gave exactly 0.0 mm3 on 4 of 20 mice and under 0.5%
    # of fat on 16, i.e. a depot that APPEARS with obesity instead of expanding.
    d["retroperitoneal"] = vat & lumb & (wall < NEAR) & _dorsal_half(F)
    # `& ~dor` is REQUIRED and was missing: without it, deep DORSAL lumbar fat -- the perirenal
    # region -- was labelled mesenteric (85% of rule B's mesenteric on scan B, 8-10% on the fat
    # mice), and mesenteric came out larger than perigonadal, inverting the published ordering.
    d["mesenteric"]      = vat & lumb & ~dor & (wall >= NEAR)
    d["perigonadal"]     = vat & ~d["mesenteric"] & ~d["retroperitoneal"] & \
                           Z(slab(F, None, "last_rib"))
    d["perirenal_proxy"] = d["retroperitoneal"] & Z(_cranial_third(F))
    return d


def rule_C_limb(F):
    """C -- SLABS + LIMB SEPARATION, for the subcutaneous depots.

    A craniocaudal plane cannot separate hindlimb fat from flank fat, because the femur is LATERAL
    to the pelvis rather than caudal to it. The first attempt at this used a WIDTH threshold --
    subcutaneous fat further from the midline than the pelvic bone's lateral extent -- and it was
    wrong in a way that mattered: `bone[pelvic slices]` contains the FEMORA, so the threshold was
    the half-width of the abducted hind limbs pooled over the whole band (9.5-22.7 mm, against a
    mouse bony pelvis of ~10-12 mm total). It exceeded the per-slice bone half-width on 97-100% of
    slices, reassigned between 0.3% and 50.6% of pelvic SUBQ depending on nothing but how splayed
    the legs were, and selected literally nothing on one animal. It made the boundary follow limb
    posture, which is the opposite of its purpose.

    THE LIMB IS WHERE THE ANIMAL PHYSICALLY SEPARATES INTO TWO. Per slice, the mouse mask is
    labelled; any component that does not contain the vertebral column is a limb. That is a
    topological fact about the animal, independent of posture, threshold and midline estimate.
    """
    d = rule_B_wall(F)
    sat, mouse = F["sat"], F["mouse"]
    Z = lambda m: m[:, None, None]
    caud = Z(slab(F, None, "pelvic_caudal"))
    pelvic = Z(slab(F, "pelvic_caudal", "pelvic_cranial"))

    limb = np.zeros_like(mouse)
    x_mid = int(round(F["x_mid"]))
    for z in range(F["z0"], F["z1"] + 1):
        m = mouse[z]
        if not m.any():
            continue
        lab, n = ndi.label(m)
        if n < 2:
            continue
        # the component carrying the spine is the trunk; everything else in this slice is limb
        sy = F["spine_y"][z]
        trunk = 0
        if np.isfinite(sy):
            yy = int(np.clip(round(sy), 0, m.shape[0] - 1))
            xx = int(np.clip(x_mid, 0, m.shape[1] - 1))
            trunk = lab[yy, xx]
            if trunk == 0:                       # spine pixel not inside any component: fall back
                sizes = np.bincount(lab.ravel())
                sizes[0] = 0
                trunk = int(sizes.argmax())
        else:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            trunk = int(sizes.argmax())
        limb[z] = (lab != trunk) & (lab != 0)

    d["hindlimb_SUBQ"] = sat & (caud | (pelvic & limb))
    d["gluteal_SUBQ"] = sat & pelvic & ~limb
    return d


def _caudal_half(F):
    """Per-slice mask for the caudal half of the lumbar band (toward the pelvis)."""
    lo, hi = F["lm"]["pelvic_cranial"], F["lm"]["last_rib"]
    return (F["cc"] >= lo) & (F["cc"] < lo + 0.5 * (hi - lo))


def _cranial_half(F):
    lo, hi = F["lm"]["pelvic_cranial"], F["lm"]["last_rib"]
    return (F["cc"] >= lo + 0.5 * (hi - lo)) & (F["cc"] < hi)




def volumes(d):
    return {k: round(float(v.sum()) * VOX, 1) for k, v in d.items()}


def check_partition(F, d, tol=1e-6):
    """Do the PARTITION depots tile their compartment exactly, with no overlap?

    Overlapping literature depots are excluded from this check by design. Conservation is an
    automatic correctness test for the partition and must not be quietly weakened to accommodate
    depots that were never meant to be disjoint.
    """
    out = {}
    for name, members, whole in (("VAT", VAT_PARTITION, F["vat"]),
                                 ("SUBQ", SUBQ_PARTITION, F["sat"])):
        have = [m for m in members if m in d]
        s = np.zeros_like(whole)
        overlap = 0
        for m in have:
            overlap += int((s & d[m]).sum())
            s |= d[m]
        out[name] = dict(missing=int((whole & ~s).sum()), extra=int((s & ~whole).sum()),
                         overlap=overlap)
    return out


def _dorsal_half(F):
    """Dorsal HALF of the cavity in each slice -- the dorsal body wall region.

    Retroperitoneal fat lies against the dorsal body wall, ventrolateral to the vertebral bodies,
    NOT dorsal to them. Defining it as "dorsal of the spine centroid" measured a sliver that does
    not exist in a lean mouse. This takes the dorsal half of the cavity's own y extent per slice,
    so it scales with the animal instead of with a plane whose position drifts with adiposity.
    """
    nz, ny, nx = F["mouse"].shape
    yy = np.arange(ny)[None, :, None]
    cav = F["cav"]
    lo = np.full(nz, np.nan); hi = np.full(nz, np.nan)
    for z in range(F["z0"], F["z1"] + 1):
        c = cav[z]
        if not c.any():
            continue
        ys = np.argwhere(c)[:, 0]
        lo[z], hi[z] = ys.min(), ys.max()
    lo = _fill_nan(lo)[:, None, None]; hi = _fill_nan(hi)[:, None, None]
    mid = 0.5 * (lo + hi)
    return (yy <= mid) if F["dorsal_is_low"] else (yy >= mid)


def _cranial_third(F):
    """The third of the lumbar band nearest the last rib (kidney level), per V4."""
    lo, hi = F["lm"]["pelvic_cranial"], F["lm"]["last_rib"]
    return (F["cc"] >= lo + (2.0 / 3.0) * (hi - lo)) & (F["cc"] < hi)


def rule_D_mass(F):
    """D -- THE VISCERAL SPLIT BY COHERENCE, not by a plane or a distance.

    Rules A and B both fail the perigonadal/mesenteric boundary, in opposite directions and by a
    factor of two on the same animal (437: A gives 17540/8662, B gives 9065/16489). A splits them
    on an arbitrary craniocaudal plane and happens to get the published ordering right for the
    wrong reason; B carves an erosion core with a distance threshold and INVERTS the ordering,
    making mesenteric the largest visceral depot, which no mouse anatomy supports.

    The real distinction is morphological. The perigonadal pad is a DISCRETE COHERENT MASS attached
    to the gonads and hanging free in the caudal abdomen. Mesenteric fat is DIFFUSE, threaded
    between gut loops as thin sheets. So: label the ventral cavity fat in 3-D, take the dominant
    connected mass as perigonadal, and call the diffuse remainder mesenteric. That is falsifiable
    by eye in a way neither a plane nor a threshold is -- if the largest component is not a pad,
    the pictures will show it.

    Retroperitoneal and the subcutaneous depots are rule A's, so any A<->D difference is
    attributable to the visceral split alone.
    """
    d = rule_A_slabs(F)
    vat = F["vat"]
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    caud = Z(slab(F, None, "pelvic_cranial"))
    dor = _dorsal_half(F)

    # everything visceral that is not retroperitoneal and not thoracic is up for splitting
    pool = vat & (lumb | caud) & ~d["retroperitoneal"]
    lab, n = ndi.label(pool)
    if n == 0:
        d["perigonadal"] = pool
        d["mesenteric"] = pool & False
        return d
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    big = int(sizes.argmax())
    mass = lab == big
    # a second component within MASS_FRAC of the largest is the contralateral pad, not mesentery
    others = [i for i in range(1, n + 1)
              if i != big and sizes[i] >= MASS_FRAC * sizes[big]]
    for i in others:
        mass |= (lab == i)
    d["perigonadal"] = mass
    d["mesenteric"] = pool & ~mass
    return d




def depot_flags(F, d):
    """Per-scan quality flags emitted alongside the volumes, so a caveat cannot be lost."""
    f = {}
    f["provisional"] = ";".join(sorted(k for k in d if k in PROVISIONAL))
    f["holder_mm3"] = round(F.get("holder_mm3", 0.0), 2)
    f["dorsal_agree"] = round(F.get("dorsal_agree", float("nan")), 3)
    f["dorsal_margin_mm"] = round(F.get("dorsal_margin_mm", float("nan")), 2)
    f["spine_slices_seen"] = F.get("spine_slices")
    f["spine_slices_total"] = F.get("spine_slices_total")
    return f


def sanity(F, d):
    """Assertions that a SWAP would trip. check_partition cannot see one: swapping two depots
    wholesale leaves it reporting 0 missing / 0 extra / 0 overlap, which is how the dorsoventral
    inversion on scan B passed every automatic test."""
    out = {}
    lumb = slab(F, "pelvic_cranial", "last_rib")[:, None, None]
    cav = F["cav"]
    # dorsal_mask (spine-based), NOT _dorsal_half (the cavity's own bounding box). _dorsal_half is
    # ~0.5 BY CONSTRUCTION -- it splits the cavity at its own mid-height -- so the test had a margin
    # of 0.02-0.05 instead of ~0.45 and raised a false "axis INVERTED" alarm on 6 of 30 scans that
    # are demonstrably correct. Measured against the spine the fraction is 0.000-0.097, which is
    # what makes >0.5 a real signal.
    dorsal_cav = float((cav & lumb & dorsal_mask(F)).sum()) / max(float((cav & lumb).sum()), 1)
    out["lumbar_cavity_dorsal_frac"] = round(dorsal_cav, 4)
    # the abdominal cavity is ventral to the spine in every mouse; more than half of it being
    # dorsal means the axis is inverted
    out["FAIL_axis_inverted"] = bool(dorsal_cav > 0.5)
    # retroperitoneal must exist in lean animals, not appear with obesity
    tot = sum(float(v.sum()) for k, v in d.items() if k in VAT_PARTITION)
    out["retro_frac_of_VAT"] = round(float(d["retroperitoneal"].sum()) / max(tot, 1), 4)
    out["FAIL_retro_empty"] = bool(d["retroperitoneal"].sum() == 0)
    return out


GUT_NEAR_MM = 1.2             # how close to bowel gas counts as mesentery
OPEN_R_MM = 1.0               # radius separating a PAD from a SHEET


def rule_E_gut(F):
    """E -- MESENTERIC BY GUT ADJACENCY. Mesentery is the tissue that suspends the intestines, so
    mesenteric fat is by definition the fat lying against bowel. Bowel is locatable without an
    organ label: it is the gas inside the abdominal cavity. Everything visceral that is NOT near
    bowel, in the caudal abdomen, is the perigonadal pad.

    This is the first rule here that uses a signal specific to the depot rather than a geometric
    convention -- a plane (A), a distance from the wall (B) or a connected mass (D) would all give
    the same answer whatever organ the fat was attached to.
    """
    d = rule_A_slabs(F)
    vat, cav, raw = F["vat"], F["cav"], F["M"]["raw"]
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    caud = Z(slab(F, None, "pelvic_cranial"))

    # bowel gas: below this scan's own fat-band floor, inside the cavity. No absolute HU cut --
    # the calibration varies continuously across these two scanners.
    gas = (raw < F["M"]["flo"]) & cav
    if gas.any():
        near = ndi.binary_dilation(gas, iterations=max(int(round(GUT_NEAR_MM / ISO)), 1))
    else:
        near = np.zeros_like(cav)
    pool = vat & (lumb | caud) & ~d["retroperitoneal"]
    d["mesenteric"] = pool & near
    d["perigonadal"] = pool & ~near
    return d


def rule_F_thick(F):
    """F -- PAD vs SHEET BY MORPHOLOGY. The perigonadal depot is a thick coherent pad; mesenteric
    fat is thin sheets threaded between gut loops. A morphological opening with a ball of radius r
    keeps everything thicker than r and deletes everything thinner, which is exactly that
    distinction and nothing else.

    Unlike D (connectivity) this does not collapse when the fat becomes confluent in an obese
    mouse -- confluent fat is still THICK where the pad is and THIN where the mesentery is -- and
    unlike B (distance from the cavity wall) it does not depend on where the wall happens to be.
    """
    d = rule_A_slabs(F)
    vat = F["vat"]
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    caud = Z(slab(F, None, "pelvic_cranial"))
    pool = vat & (lumb | caud) & ~d["retroperitoneal"]
    r = max(int(round(OPEN_R_MM / ISO)), 1)
    thick = ndi.binary_opening(pool, ndi.generate_binary_structure(3, 1), iterations=r)
    d["perigonadal"] = thick
    d["mesenteric"] = pool & ~thick
    return d


def rule_G_dvfrac(F):
    """G -- SUBQ SPLIT AT A FRACTION OF BODY THICKNESS, not at the spine.

    dorsolumbar vs inguinal is currently split at the spine line, which sits at very different
    relative depths in a lean and an obese mouse because the belly grows ventrally. Splitting at
    the midpoint of the ANIMAL's own dorsoventral extent in each slice scales with the animal, so
    the boundary means the same thing at both ends of the adiposity range. Visceral depots are
    rule A's, so any A<->G difference is attributable to the subcutaneous rule alone.
    """
    d = rule_A_slabs(F)
    sat = F["sat"]
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    nz, ny, nx = sat.shape
    yy = np.arange(ny)[None, :, None]
    lo = np.full(nz, np.nan); hi = np.full(nz, np.nan)
    for z in range(F["z0"], F["z1"] + 1):
        m = F["mouse"][z]
        if not m.any():
            continue
        ys = np.argwhere(m)[:, 0]
        lo[z], hi[z] = ys.min(), ys.max()
    mid = 0.5 * (_fill_nan(lo) + _fill_nan(hi))[:, None, None]
    dorsal = (yy <= mid) if F["dorsal_is_low"] else (yy >= mid)
    d["dorsolumbar_SUBQ"] = sat & lumb & dorsal
    d["inguinal_SUBQ"] = sat & lumb & ~dorsal
    return d


RULES = {"A_slabs": rule_A_slabs, "B_wall": rule_B_wall, "C_limb": rule_C_limb,
         "D_mass": rule_D_mass, "E_gut": rule_E_gut, "F_thick": rule_F_thick,
         "G_dvfrac": rule_G_dvfrac}


# =============================================================================================
# FACTORIAL RULE SET. The visceral split and the subcutaneous split are INDEPENDENT decisions, so
# writing them as a handful of named rules both hides the combinations and repeats work. Here each
# axis is a variant, and any pair can be composed -- which is what actually needs comparing.
#
# NOTHING IN THIS SECTION IS SELECTED BY AGREEMENT WITH A PRIOR ABOUT DEPOT SIZES. Depot
# proportions in mice are under strain-specific genetic control (a Chr 9 locus moves gonadal depot
# weight at LOD 5.3 and retroperitoneal at LOD 0.9), differ between BXD strains in opposite
# directions on the same diet, and differ between C57BL/6J SUBSTRAINS from different vendors under
# HFD vs western diet. This cohort is 8 inbred strains plus a diversity panel on chow and western
# diets, so there is no cross-strain expectation to test against, and imposing one would guarantee
# that a strain which breaks it could never be observed. Rules are judged on internal criteria
# only: conservation, absence of centroid migration with adiposity, coverage, insensitivity to
# arbitrary thresholds, cross-scanner consistency, and what the images show.
# =============================================================================================

def _visceral_pool(F, d):
    """Everything visceral that is up for the perigonadal/mesenteric split."""
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    caud = Z(slab(F, None, "pelvic_cranial"))
    return F["vat"] & (lumb | caud) & ~d["retroperitoneal"]


def vis_plane(F, d):
    """Craniocaudal plane at the middle of the lumbar band. A geometric convention, no anatomy.

    THE CAUDAL SLAB MUST BE ON THE PERIGONADAL SIDE. `_caudal_half` is defined only inside
    [pelvic_cranial, last_rib), but the pool also contains everything caudal of pelvic_cranial --
    so every one of those voxels failed the test and fell through into mesenteric. Measured:
    100% of the caudal pool was labelled mesenteric on every scan, a median 30% of the whole
    visceral pool, and 58% on scan 375 where it took perigonadal from 1062 to 306 mm3. Conservation
    could not see it because it is a swap, not a gap.
    """
    Z = lambda m: m[:, None, None]
    pool = _visceral_pool(F, d)
    caudal = Z(_caudal_half(F)) | Z(slab(F, None, "pelvic_cranial"))
    return pool & caudal, pool & ~caudal


def vis_wall(F, d, near=1.5):
    """Distance from the cavity wall: mesentery hangs free, the pad lies against the wall."""
    pool = _visceral_pool(F, d)
    w = wall_distance(F)
    return pool & (w < near), pool & (w >= near)


def vis_gut(F, d, near_mm=0.45):
    """Adjacency to bowel gas -- the only variant using a signal specific to mesentery.

    The dilation radius was 1.2 mm and collapsed on obese mice (ratio 0.01: essentially all
    visceral fat called mesenteric), because once the abdomen is full every fat voxel is within
    1.2 mm of some gas. It is now 0.45 mm, and speckle-sized gas pockets are removed first -- a
    few stray air voxels from a streak artefact are not bowel and should not recruit fat around
    them.
    """
    pool = _visceral_pool(F, d)
    gas = (F["M"]["raw"] < F["M"]["flo"]) & F["cav"]
    if gas.any():
        lab, n = ndi.label(gas)
        if n:
            sz = np.bincount(lab.ravel()); sz[0] = 0
            keep = np.where(sz * VOX >= 1.0)[0]          # >= 1 mm3: a gut lumen, not speckle
            gas = np.isin(lab, keep) if len(keep) else np.zeros_like(gas)
    near = (ndi.binary_dilation(gas, iterations=max(int(round(near_mm / ISO)), 1))
            if gas.any() else np.zeros_like(F["cav"]))
    return pool & ~near, pool & near


def vis_thick(F, d, r_mm=0.45):
    """Pad vs sheet by morphological opening, AFTER CLOSING THE SPECKLE.

    The first attempt opened the raw fat mask with a 1.0 mm radius and returned an EMPTY
    perigonadal on all 8 test mice. That was a parameter error, not a property of the approach: a
    voxel-level fat mask is porous -- fat voxels interleaved with non-fat -- so seven erosions
    annihilate it before the dilation can restore anything. Closing first fills the pores so the
    opening measures the thickness of the STRUCTURE rather than of the speckle, and the radius is
    dropped to 0.45 mm, which is the scale that separates a sheet from a pad at 0.15 mm voxels.
    """
    pool = _visceral_pool(F, d)
    st = ndi.generate_binary_structure(3, 1)
    solid = ndi.binary_closing(pool, st, iterations=2)
    r = max(int(round(r_mm / ISO)), 1)
    thick = ndi.binary_opening(solid, st, iterations=r) & pool
    return thick, pool & ~thick


def vis_gut_thick(F, d):
    """COMBINATION: mesentery must be BOTH near bowel AND thin. Each signal alone mislabels --
    fat near a gas-filled caecum can still be a thick pad, and thin fat far from any bowel is more
    likely a fascial plane than mesentery. Requiring both is the stricter, more specific rule."""
    _, mes_gut = vis_gut(F, d)
    _, mes_thin = vis_thick(F, d)
    pool = _visceral_pool(F, d)
    mes = mes_gut & mes_thin
    return pool & ~mes, mes


def sub_spine(F, d):
    """dorsolumbar / inguinal split at the spine line."""
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    dor = dorsal_mask(F)
    return F["sat"] & lumb & dor, F["sat"] & lumb & ~dor


def sub_thickfrac(F, d):
    """dorsolumbar / inguinal split at the midpoint of the animal's own dorsoventral extent, which
    scales with the animal instead of sitting at a different relative depth in lean and fat mice."""
    Z = lambda m: m[:, None, None]
    lumb = Z(slab(F, "pelvic_cranial", "last_rib"))
    sat, mouse = F["sat"], F["mouse"]
    nz, ny, nx = sat.shape
    yy = np.arange(ny)[None, :, None]
    lo = np.full(nz, np.nan); hi = np.full(nz, np.nan)
    for z in range(F["z0"], F["z1"] + 1):
        m = mouse[z]
        if m.any():
            ys = np.argwhere(m)[:, 0]
            lo[z], hi[z] = ys.min(), ys.max()
    mid = 0.5 * (_fill_nan(lo) + _fill_nan(hi))[:, None, None]
    dor = (yy <= mid) if F["dorsal_is_low"] else (yy >= mid)
    return sat & lumb & dor, sat & lumb & ~dor


def vis_radial(F, d, frac=0.5):
    """Radial position within the cavity: the pad sits peripherally, mesentery centrally. Uses the
    cavity's own radius per slice, so it scales with the animal rather than fixing a millimetre
    distance -- the criticism that made `wall` threshold-dominated."""
    pool = _visceral_pool(F, d)
    w = wall_distance(F)
    nz = F["mouse"].shape[0]
    rad = np.zeros(nz)
    for z in range(F["z0"], F["z1"] + 1):
        if F["cav"][z].any():
            rad[z] = w[z].max()
    rad = np.where(rad > 0, rad, np.nan)
    rad = _fill_nan(rad)[:, None, None]
    deep = w >= (frac * rad)
    return pool & ~deep, pool & deep


VIS_VARIANTS = {"plane": vis_plane, "wall": vis_wall, "gut": vis_gut,
                "thick": vis_thick, "gutthick": vis_gut_thick, "radial": vis_radial}
SUB_VARIANTS = {"spine": sub_spine, "thickfrac": sub_thickfrac}


def compose(vis, sub):
    """Build a rule from one visceral variant and one subcutaneous variant."""
    def _rule(F):
        d = rule_A_slabs(F)
        d["perigonadal"], d["mesenteric"] = VIS_VARIANTS[vis](F, d)
        d["dorsolumbar_SUBQ"], d["inguinal_SUBQ"] = SUB_VARIANTS[sub](F, d)
        return d
    _rule.__name__ = f"rule_{vis}_{sub}"
    return _rule


# The stand-alone named rules D/E/F are SUPERSEDED and removed from the comparison:
#   D_mass   connectivity collapses when fat is confluent -- 10521 vs 16.5 mm3 on scan C
#   E_gut    1.2 mm gas dilation reaches all visceral fat in an obese abdomen -- ratio 0.007 on 437
#   F_thick  a 1.0 mm opening annihilates a porous fat mask -- perigonadal 0.0 mm3 on 7 of 8
#   C_limb   topological limb separation moves a median of 0.00 mm3; byte-identical to B_wall
# vis_gut and vis_thick are the repaired versions and remain in the factorial.
for _dead in ("D_mass", "E_gut", "F_thick", "C_limb", "G_dvfrac"):
    RULES.pop(_dead, None)

for _v in VIS_VARIANTS:
    for _s in SUB_VARIANTS:
        RULES[f"{_v}+{_s}"] = compose(_v, _s)
