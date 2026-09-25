r"""One upload in, the full depot table out. Everything between is here.

This is the headless core; the Shiny app is a thin wrapper on it, which is deliberate -- the numbers
have to be testable without a browser, because the acceptance test for the whole project is that a
scan already in _doz_long_v3.csv comes back with the same volumes.

THE MASK DICTIONARY IS THE CONTRACT. cohort_derive.derive_masks defines the shape the depot rules
consume, and this function reproduces that shape exactly so depot_rules runs UNCHANGED against it.
That matters more than it looks: the ten depot definitions, the dorsoventral frame, the spine
midline and the conservation checks are all code that has already been validated page by page, and
re-implementing any of it to suit a new caller is how an app quietly stops agreeing with the paper.

WHAT CTAdipo SUPPLIES THAT THE ORIGINAL PIPELINE TOOK FROM A HUMAN TABLE:
    five landmark planes     the landmark model
    craniocaudal direction   implied by the landmarks (lung apex relative to bladder)
    dorsoventral axis        geometry, with the decision margin surfaced and a manual override
    lung mask                the same sweep the cohort used, pinned to the reviewed configuration
    hardware polygons        NOT supplied, and deliberately so

HARDWARE. The original derivation subtracts hand-drawn polygons keyed to scanner AND reconstruction
crop, because on that rig the breathing pad fuses to the animal and its dark interior is labelled
cavity, i.e. counted as visceral fat. Those polygons cannot transfer to another scanner and shipping
them would be worse than useless. CTAdipo does what generalises -- keep_largest3d, which removes
DETACHED hardware -- and shows the user the mask so fused hardware is visible rather than silent.

LUNG. Lung reads -300 to -400 HU, inside both fat bands, so unsubtracted lung is counted as fat.
Measured across the adiposity range, that is ~1.5% of fat in the fattest animals and ~16% in the
leanest, correlating with log total fat at r = -0.83. Skipping it is therefore NOT neutral: it
inflates lean animals specifically, along the exact axis these measurements are used to study. The
caller passes `lung_fn`; when it is absent that fact is reported, never passed over quietly.
"""
from __future__ import annotations

import gc

import numpy as np

from . import preprocess as P
from . import landmark_model as LMM

DENSITY_G_PER_ML = 0.9          # adipose tissue; volumes convert to mass by simple density
VOX_MM3 = P.ISO ** 3

# THE TEN that tile VAT and SUBQ exactly, and are what the paper reports.
VAT_DEPOTS = ["perigonadal", "mesenteric", "retroperitoneal", "thoracic_VAT"]
SUBQ_DEPOTS = ["anterior_SUBQ", "dorsolumbar_SUBQ", "inguinal_SUBQ",
               "gluteal_SUBQ", "hindlimb_SUBQ", "head_neck_SUBQ"]
PARTITION_DEPOTS = VAT_DEPOTS + SUBQ_DEPOTS

# rule_A_slabs ALSO returns depots that OVERLAP the ten and each other, so they must never be summed
# with them or shown beside them unmarked. Note this is depot_rules.OVERLAP, which is a different
# set from depot_rules.PROVISIONAL -- PROVISIONAL marks depots whose DEFINITION is not settled and
# includes `mesenteric` and `thoracic_VAT`, both of which are among the ten. The app reads
# PROVISIONAL off the rules module itself and flags those two rather than silently presenting them
# as settled; it is not this file's job to second-guess which list a depot belongs on.
OVERLAPPING = ["posterior_SUBQ_whole", "interscapular", "perirenal_proxy", "mediastinal",
               "lumbar_window_VAT", "lumbar_window_SUBQ", "preperitoneal"]


def build_masks(raw, pred, lung_fn=None):
    """The dictionary depot_rules consumes, built the way cohort_derive builds it.

    Returned keys and their meanings are cohort_derive.derive_masks's, not a new invention.
    """
    body, cav = P.compartments(pred)
    air_p1, flo, fhi = P.scanner_band(raw)
    fat = P.fat_mask(raw, body, flo, fhi)

    lung_reg = np.zeros_like(body)
    lung_mL, lung_floor, lung_used = 0.0, float("nan"), "no"
    if lung_fn is not None:
        lung_mL, lung_reg, lung_floor, _ = lung_fn(raw, P.keep_largest3d(body))
        lung_used = "yes"
    # What was actually taken OUT OF THE FAT, which is not the same as the lung region's volume:
    # only about 63% of the lung region falls inside the fat band, so reporting lung_mL as "fat
    # removed" overstates it by half as much again.
    removed = fat & lung_reg
    lung_fat_mL = float(removed.sum()) * VOX_MM3 / 1000.0
    fat = fat & ~lung_reg

    wall_mm3, edge_on_wall = P.wall_quality(pred, body, cav)
    return dict(pred=pred, raw=raw, body=body, cav=cav, cav_raw=(pred == P.CAVITY), fat=fat,
                vat=fat & cav, sat=fat & ~cav, lung_reg=lung_reg,
                air_p1=air_p1, flo=flo, fhi=fhi,
                lung_mL=lung_mL, lung_fat_mL=lung_fat_mL,
                lung_floor=lung_floor, lung_used=lung_used,
                wall_mm3=wall_mm3, edge_on_wall=edge_on_wall)


def analyse(raw, segmenter, landmark_nets, *, lung_fn=None, device="cpu",
            dorsal_is_low=None, depot_rules=None, aid="upload", progress=None,
            on_segmented=None):
    """Full measurement of one volume already on the 0.15 mm isotropic grid.

    `segmenter(z_scored_volume) -> label volume` keeps the nnU-Net call injectable, so the pipeline
    is testable against stored predictions without a GPU or a 410 MB checkpoint.
    """
    def say(msg):
        if progress:
            progress(msg)

    say("finding the animal")
    A = P.animal_mask(raw)

    say("normalising")
    z, mean, sd = P.normalise(raw, A)

    say("segmenting body, wall and cavity")
    pred = segmenter(z)
    # The network is finished with. It is ~1.07 GB of weights, and the single most
    # allocation-heavy step -- the depot partition, which builds seventeen full-resolution boolean
    # volumes at once -- is still to come. Measured on a 540x529x529 scan the partition alone takes
    # the process from 3.5 GB to 7.1 GB, so on an 8 GB worker the weights are the difference
    # between finishing and being killed. The caller decides how to release them, because it owns
    # the cache; the cost is reloading from local disk on the next scan, which is seconds.
    if on_segmented is not None:
        # Drop OUR reference before asking the caller to drop theirs. Python frees an object only
        # when the last name bound to it goes, so clearing a cache while this parameter still holds
        # the predictor frees nothing at all -- which is exactly what the first attempt at this did.
        segmenter = None
        on_segmented()
        gc.collect()
    # The z-scored copy is 604 MB on a normal scan and is dead the instant the network has run.
    # Holding it through the fat band, the lung sweep and the depot partition -- the three stages
    # that allocate most -- is what pushes an 8 GB worker over the edge.
    del z
    gc.collect()

    say("measuring fat")
    M = build_masks(raw, pred, lung_fn=lung_fn)

    say("placing landmarks")
    L = LMM.predict(raw, A, landmark_nets, device=device)

    out = {
        "landmarks": L, "normalisation": {"mean": mean, "sd": sd},
        "scanner": {"air_p1": M["air_p1"], "fat_band": (M["flo"], M["fhi"]),
                    "scale": "wide" if M["air_p1"] >= P.SCALE_AIR else "narrow"},
        "quality": quality_flags(M, L),
        "totals": totals(M),
    }

    if depot_rules is not None:
        # frame() reads only body, cav, raw, vat and sat (it discards pred itself). Everything else
        # in the mask set is finished with by now, and each one is a full-resolution boolean volume
        # -- about 450 MB together on a normal scan, freed immediately before the single most
        # allocation-heavy step in the pipeline.
        for dead in ("fat", "cav_raw", "lung_reg"):
            M.pop(dead, None)
        gc.collect()
        say("partitioning into depots")
        # The partition can legitimately fail -- an abdomen-only scan, or landmarks that come out
        # in the wrong order -- and when it does, the totals, the scanner band and the wall QC
        # above are all still correct. Letting the exception escape would throw those away too and
        # hand the user a traceback after a seven-minute wait, so it is caught HERE and reported.
        try:
            out["depots"] = partition(M, L, depot_rules, aid=aid, dorsal_is_low=dorsal_is_low)
        except Exception as e:
            out["depots_error"] = str(e)
    return out


def totals(M):
    """Whole-compartment volumes and masses. VAT and SUBQ are fat inside and outside the cavity."""
    def mL(m):
        return float(m.sum()) * VOX_MM3 / 1000.0
    v, s = mL(M["vat"]), mL(M["sat"])
    return {"VAT_mL": v, "SUBQ_mL": s, "TotalFat_mL": v + s,
            "VAT_g": v * DENSITY_G_PER_ML, "SUBQ_g": s * DENSITY_G_PER_ML,
            "TotalFat_g": (v + s) * DENSITY_G_PER_ML,
            "lung_fat_removed_mL": M["lung_fat_mL"],
            "lung_region_mL": M["lung_mL"]}


def partition(M, L, DR, aid="upload", dorsal_is_low=None):
    """The ten depots, via rule_A_slabs against the model's landmarks."""
    lm = {slot: L["z"][name] for slot, name in LMM.DEPOT_ALIAS.items()}
    lm["_kidney_cranial"] = L["z"]["kidney_cranial"]
    F = DR.frame(aid, lm, L["direction"], masks=M, dorsal_is_low=dorsal_is_low)
    d = DR.rule_A_slabs(F)
    v = DR.volumes(d)                                    # mm^3
    ten = {k: v[k] / 1000.0 for k in PARTITION_DEPOTS if k in v}
    return {"mL": ten,
            "g": {k: x * DENSITY_G_PER_ML for k, x in ten.items()},
            "overlapping_mL": {k: v[k] / 1000.0 for k in OVERLAPPING if k in v},
            "masks": d, "frame": F,
            "conservation": DR.check_partition(F, d)}


def quality_flags(M, L):
    """The things that go wrong silently, surfaced rather than hidden.

    Each of these has a documented history of producing a plausible, wrong number. The app shows
    them next to the result; none of them rejects a scan on its own.
    """
    f = {}
    f["wall_mm3"] = M["wall_mm3"]
    f["edge_on_wall"] = M["edge_on_wall"]
    f["cavity_ok"] = M["wall_mm3"] >= P.WALL_MM3_OK
    # VAT and SUBQ are separated by ONE boundary. Where the model has not found the abdominal wall
    # the cavity is unanchored, and the VAT/SUBQ split stops meaning anything while both totals
    # still look entirely reasonable. The 70 mm3 threshold separates the reviewer's own verdicts
    # with no overlap, and it is reproduced here EXACTLY -- `cavity_ok` is the locked criterion and
    # nothing else may be folded into it.
    #
    # This is advisory and deliberately separate. The same review recorded 45-65% of the cavity edge
    # against wall on good scans versus 6-17% on bad ones, but never made that a gate. A lean scan
    # can clear 70 mm3 on wall volume and still sit at 0.29 here, which is worth showing the user --
    # as a second opinion, not as a second threshold that would quietly change what passes.
    f["edge_on_wall_low"] = M["edge_on_wall"] < 0.45

    f["direction"] = L["direction"]
    f["direction_margin_mm"] = L["direction_margin_mm"]
    f["direction_uncertain"] = L["direction_uncertain"]
    # A direction flip is the most dangerous failure in the whole chain precisely because it is
    # invisible: it swaps anterior with gluteal and retroperitoneal with perigonadal while
    # conservation still holds exactly and every total is unchanged.

    f["landmarks_ordered"] = L["ordered"]
    # The depot rules require bladder < kidney_caudal < diaphragm < lung_apex along the body. Out of
    # order means the landmarks and the implied direction disagree, and the partition would be
    # guessing which is wrong.

    f["landmark_spread_mm"] = L["spread_mm"]
    f["landmark_uncertain"] = {k: v > 3.0 for k, v in L["spread_mm"].items()}
    # The model predicts a distribution over z, so a wide one is a genuine "this scan does not look
    # like the training data" signal rather than a confident answer from an unfamiliar image.

    f["lung_subtracted"] = M["lung_used"] == "yes"
    fat_mL = float(M["fat"].sum()) * VOX_MM3 / 1000.0
    f["lung_share_of_fat"] = M["lung_fat_mL"] / max(fat_mL + M["lung_fat_mL"], 1e-9)

    # Almost no fat inside the band. The bands were derived on UNCALIBRATED microCT; a genuinely
    # HU-calibrated scan puts adipose at -190 to -30 HU, which barely overlaps either band, so the
    # app would otherwise report ~0 mL with a clean panel. Also catches an all-zero volume, which
    # passes every other check because 0 > -500 is true everywhere.
    body_mL = float(M["body"].sum()) * VOX_MM3 / 1000.0
    f["fat_fraction_of_body"] = fat_mL / max(body_mL, 1e-9)
    f["band_suspect"] = f["fat_fraction_of_body"] < 0.005
    return f
