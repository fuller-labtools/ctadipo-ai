r"""CANDIDATE LUNG/AIR MASK v2 -- no fixed HU anywhere, chosen per scan by STABILITY.

VENDORED INTO CTAdipo. Executable code unchanged; cohort scan identifiers in the comments were
replaced with anonymous labels (scan A, scan B, ...) before publication, one label per animal so
comparisons still read as comparisons. That no executable line moved is checked mechanically, by
comparing this file's abstract syntax tree against the research copy.

The line that used to stand here -- "NOT WIRED INTO THE PIPELINE, a proposal to be judged on
rendered images" -- was true when this was written and is not any more: this IS the lung step the
measurement uses, called with the configuration a reviewer signed off page by page over all 1219
scans (select="peak", coarse=True, erode at its default of 5). Any drift from that stops the
numbers matching the images that were approved.

WHY v1 FAILED, measured rather than guessed (2026-07-31)
  v1 called anything below the fat-band floor (-491 HU) 'air'. That floor is FIXED, but the HU scale
  is not: the lean 3M scans read soft tissue at +41 HU and the obese Aging scans read it at -233 --
  a ~275 HU offset in where the whole tissue scale sits. So -491 sat below the lean lungs entirely.

  Consequence, on the 156 scans rendered so far: 33 (21%) found NO lung, and every one of them had
  air/tissue separation in the TOP of the cohort range (0.873-0.912, median 0.890), while all 5
  genuinely broken scans DID get a lung. The flag was ANTI-correlated with scan quality -- the same
  failure as the rib metric, the fat-band fraction and the landmark confidence scores. Scans were
  being excluded for the detector's fault.

  Ruled out by measurement, so they are not the cause:
    the FOAM_HI=0.90 ceiling      the >0.90 'gas void' bin holds 0.01 mL, and relaxing it recovers
                                  nothing
    the lung being a hole in pred hole-filling the body mask adds 0.00 mL
    the DEEP_VOX=15 erosion       no compact component appears at ANY erosion depth on 660/639;
                                  everything found at shallow depths spans the whole 72 mm animal,
                                  which is the skin rind, not a lung

WHAT THIS DOES INSTEAD
  1. SWEEP the floor across a wide range instead of fixing it. At each floor, take air-like
     components that are ORGAN-SIZED and COMPACT (a z-span of 8-35 mm; the skin rind gives itself
     away by spanning the whole 60-110 mm animal).
  2. PICK THE PEAK, not a plateau. The first version of this looked for a stable plateau, on the
     maximally-stable-region argument. THE DATA HAS NO PLATEAU: on every scan examined the sweep
     climbs steeply and then COLLAPSES, and the plateau rule ended up choosing a point on the rising
     edge -- 0.16 mL on 639 where the curve peaks at 0.55 mL, covering about a third of the visible
     lung field. Judged on the images, that was under-segmentation, not stability.

     The collapse is the real signal, and it is physical: as the floor rises the mask takes in more
     of the lung, until it touches the surrounding tissue, merges, and blows through the size and
     z-span limits -- at which point the component is rejected and the curve drops to near zero. So
     the peak is THE MOST LUNG RECOVERABLE BEFORE IT STOPS LOOKING LIKE AN ORGAN, and the merge
     enforces the ceiling by itself. No calibration constant either way.

     WHY UNDER-SEGMENTING IS NOT THE SAFE DIRECTION HERE, contrary to what I first assumed. In the
     lean 3M scans the lung reads -300 to -400 HU, which is INSIDE the fat band [-491, -118]. Lung
     that is not subtracted is therefore counted as FAT. Those animals have 1.4-4.4 mL of fat in
     total (3M median 2.17 mL, minimum 0.24 mL), so a ~1 mL lung is a fifth to a third of the whole
     measurement -- and only in the LEAN animals, since the same lung is ~1.5% of an obese mouse's
     55 mL. That is a bias running along exactly the lean/obese axis under study.
  3. The sweep runs on a 2x-downsampled volume (0.30 mm) for speed; the final mask is computed once
     at full resolution using the chosen floor.

WHAT IT DELIBERATELY DOES NOT DO
  It does not try to prove a component is specifically LUNG rather than bowel gas or airway. That
  distinction does not matter for the measurement this feeds: none of them is fat, so subtracting
  any of them costs nothing, whereas the v0 failure -- selecting the air OUTSIDE the animal and
  deleting the subcutaneous fat beneath it -- destroyed up to 74% of the fat. Rejecting the RIND is
  the job; identifying the organ is not. The mask is returned so it can be DRAWN and judged.
"""
import numpy as np
from scipy import ndimage as ndi

ISO = 0.15
VOX = ISO ** 3 / 1000.0
FOAM_BOX = 7                    # 1.05 mm neighbourhood for the local air fraction
FOAM_LO, FOAM_HI = 0.20, 0.98   # upper bound only excludes a perfectly empty box
ERODE = 5                       # 0.75 mm: kills the skin/air interface without eating a lean thorax
MIN_mL, MAX_mL = 0.05, 2.50     # organ-sized
SPAN_LO_mm, SPAN_HI_mm = 6.0, 55.0     # lung/gas pocket, NOT a whole-animal rind
SPAN_FRAC = 0.60                # ...and never more than this fraction of the body length
# SPAN_HI_mm WAS 35.0 AND THAT WAS A BUG. It discarded real lungs for being a few mm too long --
# scan E's lung spans 35.4 mm and scan F's spans 40.6 mm, so both returned an EMPTY mask
# while their sweeps peaked at 0.85 and 1.28 mL. Both are the deepest component in their scan
# (relative depth 0.56 and 0.66). Airways are included by design, which lengthens the component.
NEW_COMP_mL = 0.30              # a new component this big means something else has joined in
JUMP = 2.0                      # ...as does an existing component more than doubling in one step


def tissue_mode(raw, body):
    """Peak of the IN-BODY HU histogram: where this scan's soft tissue and fat actually sit.

    This is the number that makes a scan behave as 'lean' or 'obese' for the lung floor -- lean 3M
    scans peak near +41 HU, obese Aging scans near -233. NOTHING BRANCHES ON IT: the floor is chosen
    per scan by the sweep, so there is no lean/obese assignment that can be made wrongly. It is
    reported so the choice can be CHECKED on the review page -- a floor that has climbed up near
    this peak is a floor that is about to start calling fat 'air', which is what to look for.
    """
    v = raw[body].astype(np.float32)
    v = v[(v > -1400) & (v < 800)]
    if v.size < 1000:
        return np.nan
    h, e = np.histogram(v, bins=280)
    c = 0.5 * (e[1:] + e[:-1])
    return float(c[int(np.argmax(ndi.uniform_filter1d(h.astype(float), 7)))])


def _components(raw, deep, flo, blen_mm, iso=ISO, box=FOAM_BOX):
    """Compact, organ-sized air-like components at one floor. Returns (total_mL, labels, keep).

    `box` is in VOXELS, so it must be rescaled when the volume is downsampled -- otherwise the sweep
    measures a physically different neighbourhood from the mask whose floor it is choosing. Leaving
    it at 7 on 2x-downsampled data made the sweep use 2.1 mm against the mask's 1.05 mm, and on 437
    that picked the wrong side of a double peak (coarse said 0.60 mL, the mask came out 0.36 mL).
    """
    frac = ndi.uniform_filter((raw < flo).astype(np.float32), size=box)
    m = (frac > FOAM_LO) & (frac < FOAM_HI) & deep
    lab, n = ndi.label(m)
    if n == 0:
        return 0.0, lab, []
    vox = iso ** 3 / 1000.0
    sz = np.bincount(lab.ravel())[1:] * vox
    keep = []
    span_hi = min(SPAN_HI_mm, SPAN_FRAC * blen_mm)
    for k in np.where((sz >= MIN_mL) & (sz <= MAX_mL))[0] + 1:
        zs = np.where((lab == k).any(axis=(1, 2)))[0]
        span = (zs.max() - zs.min() + 1) * iso
        if SPAN_LO_mm <= span <= span_hi:
            keep.append(k)
    return float(sum(sz[k - 1] for k in keep)), lab, keep


def _comp_list(raw, deep, flo, blen_mm, iso=ISO, box=FOAM_BOX, want_ids=False):
    """Accepted components at one floor.

    Returns [(volume_mL, z_centroid), ...] largest first, and -- if want_ids -- a uint8 array
    labelling ONLY the accepted components 1..N. That array is how components are matched between
    floors: by actual spatial OVERLAP rather than by centroid distance.

    WHY OVERLAP. Raising the floor only ever ADDS voxels, so a component at one floor is physically
    contained in the mask at the next. Identity is therefore an exact question, not a question of
    tolerance. Centroid matching was tried and cannot be made to work: at 8% of body length the
    lung's own centre drifting from 43% to 29% as it filled out read as a new object appearing
    (scan F stopped at 0.31 mL instead of 0.85); at 15% the fat arriving at 38% was absorbed
    into the lung at 29% and the sweep ran one floor too far (1.35 mL, including 0.48 mL of fat).
    There is no value that does both. Overlap needs no value at all.

    Only accepted components are labelled, so the array fits in uint8 -- there are rarely more than
    a handful, and a full int32 label volume would cost ~350 MB per floor.
    """
    frac = ndi.uniform_filter((raw < flo).astype(np.float32), size=box)
    m = (frac > FOAM_LO) & (frac < FOAM_HI) & deep
    lab, n = ndi.label(m)
    if n == 0:
        return ([], np.zeros(raw.shape, np.uint8)) if want_ids else []
    vox = iso ** 3 / 1000.0
    sz = np.bincount(lab.ravel())[1:] * vox
    span_hi = min(SPAN_HI_mm, SPAN_FRAC * blen_mm)
    out, ids = [], (np.zeros(raw.shape, np.uint8) if want_ids else None)
    for k in np.where((sz >= MIN_mL) & (sz <= MAX_mL))[0] + 1:
        zs = np.where((lab == k).any(axis=(1, 2)))[0]
        if SPAN_LO_mm <= (zs.max() - zs.min() + 1) * iso <= span_hi:
            out.append((float(sz[k - 1]), float(zs.mean()), int(k)))
    out.sort(reverse=True)
    if want_ids:
        for j, (_, _, k) in enumerate(out[:255], start=1):
            ids[lab == k] = j
        return [(v, z) for v, z, _ in out], ids
    return [(v, z) for v, z, _ in out]


def _stop_floor_overlap(raw_s, deep_s, floors, blen, iso_s, box):
    """The last floor before something OTHER THAN THE LUNG joins in, decided by OVERLAP.

    At each floor, every accepted component is compared with the accepted components of the previous
    floor that it physically contains:

      NEW      it overlaps nothing from the previous floor, and is >= NEW_COMP_mL
               -> a separate structure has appeared. On scan G this is 0.67 mL at 33% and
                  0.66 mL at 47% of the body arriving in one step, while the lung sits at 73%.
      MERGED   it overlaps previous components but is more than JUMP times their combined volume
               -> it has bled into surrounding tissue. On 432 this is 0.18 -> 1.23 mL in one step.

    The size guard on the merge test matters: early in the sweep the lung itself doubles repeatedly
    as it becomes visible (scan F runs 0.12 -> 0.31 -> 0.51 mL, verified on the images to be
    the lung throughout), so a component below NEW_COMP_mL is exempt from the jump test.

    Returns (chosen_floor, trace) where trace is [(floor, total_mL), ...].
    """
    prev_ids, prev_vols, pick, trace = None, None, None, []
    for f in floors:
        cur, ids = _comp_list(raw_s, deep_s, f, blen, iso=iso_s, box=box, want_ids=True)
        trace.append((f, sum(v for v, _ in cur)))
        if not cur:
            continue
        if prev_ids is not None:
            joined = False
            for j, (v, _) in enumerate(cur[:255], start=1):
                touched = np.unique(prev_ids[ids == j])
                touched = touched[touched != 0]
                if not len(touched):
                    if v >= NEW_COMP_mL:
                        joined = True
                else:
                    base = sum(prev_vols[t - 1] for t in touched)
                    if base >= NEW_COMP_mL and v > JUMP * base:
                        joined = True
            if joined:
                break
        pick = f
        prev_ids, prev_vols = ids, [v for v, _ in cur[:255]]
    return pick, trace


def _stop_floor(per_floor, floors, ztol):
    """The last floor before something OTHER THAN THE LUNG joins in.

    MEASURED BEHAVIOUR, across nine animals: the lung appears at a low floor and then grows
    SMOOTHLY on its own -- scan G runs as a single component from 0.53 to 0.86 mL over five
    consecutive floors, scan F from 0.12 to 0.85 over eight. Fat does not do that. It arrives
    ABRUPTLY, either as a NEW component (scan G gains 0.67 mL at 33% and 0.66 mL at 47% of the
    body in one step, going from 1 component to 12) or as an existing one that suddenly balloons
    (432's abdominal component goes 0.18 -> 1.23 mL in a single step).

    So: advance while the component set is stable, and stop at the last floor before either happens.
    No HU constant, and no threshold on the lung's own size or position.

    Components are matched between floors by z centroid, since a growing lung stays put.
    """
    prev, pick = None, None
    for f in floors:
        cur = per_floor.get(f) or []
        if not cur:
            continue
        if prev is not None:
            joined = False
            for v, z in cur:
                near = [(pv, pz) for pv, pz in prev if abs(pz - z) <= ztol]
                if not near:                       # something NEW and substantial appeared
                    if v >= NEW_COMP_mL:
                        joined = True
                else:
                    # ...or an existing, ALREADY SUBSTANTIAL component ballooned. The size guard
                    # matters: early in the sweep the lung itself doubles repeatedly as it becomes
                    # visible (scan F runs 0.12 -> 0.31 -> 0.51 mL), and without it the sweep
                    # stopped at the first of those and reported 0.12 mL instead of 0.85.
                    pmax = max(pv for pv, _ in near)
                    if pmax >= NEW_COMP_mL and v > JUMP * pmax:
                        joined = True
            if joined:
                break
        pick, prev = f, cur
    return pick


def lung_air(raw, body, floors=None, report=False, select="peak", coarse=True, erode=None):
    """Air inside the body, at the floor where the measurement is most STABLE.

    Returns (volume_mL, mask, chosen_floor, trace). trace is the sweep, for plotting next to the
    image so the choice can be audited rather than trusted.
    """
    empty = np.zeros_like(body)
    zb = np.where(body.any(axis=(1, 2)))[0]
    if not len(zb):
        return 0.0, empty, np.nan, []
    blen = (zb.max() - zb.min() + 1) * ISO

    # --- THE SWEEP RUNS AT FULL RESOLUTION, on exactly the data the final mask is built from.
    #
    #     A 2x-downsampled sweep was tried and FAILED. The emergence rule was derived from
    #     full-resolution measurements and reproduces the image-verified answer on all nine test
    #     animals by hand -- but run against the coarse volume it gave 0.53 mL for scan G
    #     (should be 0.86), 2.78 for scan F (0.85) and 0.16 for scan E (0.89). Components
    #     merge and split at different floors once the volume is downsampled, so the stop triggers
    #     in the wrong place. The same coarse/full mismatch had already caused two earlier faults:
    #     a floor chosen on the wrong side of a double peak (437), and empty masks whose sweeps
    #     peaked at 0.85-1.28 mL (scan E, scan F).
    #
    #     This costs about 55 s per scan instead of 12. That is the price of choosing the floor on
    #     the same data the answer is reported from, and it is not negotiable at this point.
    if floors is None:
        floors = list(range(-700, -199, 25))
    er = ERODE if erode is None else erode
    deep_s = ndi.binary_erosion(body, iterations=er) if not coarse else None
    if coarse:
        rs, bs = raw[::2, ::2, ::2], body[::2, ::2, ::2]
        deep_s = ndi.binary_erosion(bs, iterations=max(er // 2, 1))
        raw_s, iso_s = rs, ISO * 2
    else:
        raw_s, iso_s = raw, ISO
    if deep_s is None or not deep_s.any():
        return 0.0, empty, np.nan, []

    if select == "emerge":
        pick, trace = _stop_floor_overlap(raw_s, deep_s, floors, blen, iso_s, FOAM_BOX)
        if not any(v > 0 for _, v in trace):
            return 0.0, empty, np.nan, trace
        if pick is None:
            pick = min(f for f, v in trace if v > 0)
        per_floor = None
    else:
        trace, per_floor = [], {}
        for f in floors:
            cl = _comp_list(raw_s, deep_s, f, blen, iso=iso_s, box=FOAM_BOX)
            per_floor[f] = cl
            trace.append((f, sum(v for v, _ in cl)))
        if not any(v > 0 for _, v in trace):
            return 0.0, empty, np.nan, trace

    # --- PEAK: the floor giving the largest compact interior air volume. Past the peak the mask
    #     merges into surrounding tissue, breaks the size/span limits and is rejected, so the curve
    #     collapses -- the merge is the ceiling, and it enforces itself. Ties go to the MORE
    #     NEGATIVE floor, which is the more conservative mask. ---
    if select == "peak":
        vmax = max(v for _, v in trace)
        pick = min(f for f, v in trace if v >= vmax - 1e-12)
    else:
        # retained for comparison only; see the module docstring for why it under-segments
        best, i = (0, None), 0
        while i < len(trace):
            if trace[i][1] <= 0:
                i += 1
                continue
            j = i
            while j + 1 < len(trace) and trace[j + 1][1] > 0:
                run = [v for _, v in trace[i:j + 2]]
                med = float(np.median(run))
                if med <= 0 or max(abs(v - med) / med for v in run) > 0.25:
                    break
                j += 1
            if j - i + 1 > best[0]:
                best = (j - i + 1, (i, j))
            i = j + 1
        pick = (max(trace, key=lambda t: t[1])[0] if best[1] is None
                else trace[(best[1][0] + best[1][1]) // 2][0])

    # --- FINAL MASK AT FULL RESOLUTION, ONE PASS at the chosen floor.
    #
    #     A 'refinement' that re-measured the peak AND ITS TWO NEIGHBOURS at full resolution and kept
    #     the LARGEST was tried and REMOVED. Taking a maximum is biased upward by construction, so it
    #     walks into the merge rather than stopping before it: on scan G it moved the floor
    #     from -475 to -425 and the mask from 0.86 mL of lung to 2.53 mL that included SUBCUTANEOUS
    #     FAT on the outer edge of the animal -- the exact v0 failure this work exists to remove.
    #     Verified on the rendered axial slices, not on the number.
    deep = deep_s if not coarse else ndi.binary_erosion(body, iterations=ERODE)
    if deep is None or not deep.any():
        return 0.0, empty, pick, trace
    tot, lab, keep = _components(raw, deep, pick, blen)
    if not keep:
        return 0.0, empty, pick, trace
    mask = np.isin(lab, keep)
    if report:
        print(f"    chosen floor {pick} HU, {tot:.2f} mL in {len(keep)} component(s)")
    return float(tot), mask, pick, trace
