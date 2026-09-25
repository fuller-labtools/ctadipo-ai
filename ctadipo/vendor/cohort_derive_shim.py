"""Local stand-ins for the two cohort_derive helpers depot_rules needs.

Vendored for deployment: the original module lives beside the research data and reads tables that do
not exist on a server. Only these two names are used by depot_rules.
"""
import numpy as np
from scipy import ndimage as ndi


def keep_largest3d(m):
    """Largest 3-D connected component. Identical to cohort_derive.keep_largest3d."""
    lbl, n = ndi.label(m)
    if n <= 1:
        return m
    c = np.bincount(lbl.ravel())
    c[0] = 0
    return lbl == int(c.argmax())


def derive_masks(aid):
    """Unreachable in CTAdipo, which always passes masks into frame(). Fails loudly if that changes."""
    raise RuntimeError(
        "derive_masks is not available in the app: it reads theresearch  cohort's reviewed tables and "
        "predictions from local disk. CTAdipo builds its masks in ctadipo.pipeline.build_masks and "
        "passes them to frame(masks=...), so reaching this call means that contract was broken.")
