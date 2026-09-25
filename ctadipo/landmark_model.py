r"""The landmark model: ONE definition of the features and the network, shared by training and the app.

This file exists as a single module on purpose. The features are non-trivial -- body-masked
projections, hole-filled mask, per-view standardisation over body pixels only -- and if the app
re-implemented them alongside the trainer they would drift, silently, and the model would be fed
something slightly different from what it learned on. Every number CTAdipo reports depends on these
five planes, so there is one implementation and both sides import it.

WHAT IT PREDICTS: five craniocaudal planes -- lung apex, diaphragm, cranial and caudal kidney, and
bladder -- as fractions of the volume's z axis, plus the craniocaudal direction, which is not a
separate prediction but simply whether the lung apex came out cranial of the bladder.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage as ndi

ISO = 0.15
ZR, WR = 256, 128
LM = ["lung_apex", "diaphragm", "kidney_cranial", "kidney_caudal", "bladder"]
CHAN = ["cor_bone", "cor_air", "cor_bulk", "cor_thick",
        "sag_bone", "sag_air", "sag_bulk", "sag_thick"]

# The landmark names the depot rules ask for, and the model outputs they map to.
DIRECTION_MARGIN_MM = 28.0   # see predict(): catches 4/4 known direction errors, flags <1%

DEPOT_ALIAS = {"pelvic_caudal": "bladder", "pelvic_cranial": "kidney_caudal",
               "last_rib": "diaphragm", "shoulder": "lung_apex"}


# --------------------------------------------------------------------------------- features
def animal_mask(raw: np.ndarray) -> np.ndarray:
    """Largest 3-D component above -500 HU, holes FILLED. Same rule as the normalisation uses.

    Filling is what makes the lungs visible: they sit below the threshold, so without it they are
    excluded as background and the air channels lose the one structure the thoracic landmarks are
    defined on. 3-D connectivity is what excludes the sample holder, which can be the widest object
    in a single slice but is a separate object in 3-D.
    """
    lbl, n = ndi.label(raw > -500.0)
    if n == 0:
        raise RuntimeError("no body component above -500 HU")
    sz = np.bincount(lbl.ravel())
    sz[0] = 0
    return ndi.binary_fill_holes(lbl == int(sz.argmax()))


def _resize2d(a, h, w):
    """Area-average resize by index bins, so a z of 540 averages about two slices into each row."""
    zi = np.linspace(0, a.shape[0], h + 1).astype(int)
    xi = np.linspace(0, a.shape[1], w + 1).astype(int)
    rows = np.stack([a[zi[i]:max(zi[i + 1], zi[i] + 1)].mean(0) for i in range(h)])
    return np.stack([rows[:, xi[j]:max(xi[j + 1], xi[j] + 1)].mean(1)
                     for j in range(w)], 1).astype(np.float32)


def features(vol: np.ndarray, A: np.ndarray) -> np.ndarray:
    """(8, ZR, WR): coronal and sagittal x [bone | air | bulk | thickness], all INSIDE the animal.

    Projecting over the raw ray instead makes the air channels useless: background air at -1000 HU
    is darker than lung at -350, so the minimum on any ray leaving the animal is always background.

    Built slice by slice. Both projections collapse an in-plane axis, so each output row depends on
    one z slice only and the full volume is never copied.
    """
    Z, Y, X = vol.shape
    buf = {k: np.empty((Z, X if k.startswith("cor") else Y), np.float32) for k in CHAN}
    for z in range(Z):
        a = A[z]
        v = vol[z].astype(np.float32)
        for ax, tag in ((0, "cor"), (1, "sag")):
            cnt = a.sum(ax).astype(np.float32)
            ok = cnt > 0
            buf[tag + "_thick"][z] = cnt * ISO
            buf[tag + "_bone"][z] = np.where(ok, np.where(a, v, -3000.0).max(ax), 0.0)
            buf[tag + "_air"][z] = np.where(ok, np.where(a, v, 3000.0).min(ax), 0.0)
            buf[tag + "_bulk"][z] = np.where(ok, (v * a).sum(ax) / np.maximum(cnt, 1.0), 0.0)
    return np.stack([_resize2d(buf[k], ZR, WR) for k in CHAN]).astype(np.float32)


def standardise(feat: np.ndarray) -> np.ndarray:
    """(10, ZR, WR): per-scan z-score over BODY pixels only, plus a validity channel per view.

    Standardising over the whole frame would make the statistics depend on how much of it the animal
    fills, which varies with body size -- that leaks adiposity into the normalisation. The validity
    channel distinguishes fill from tissue that genuinely reads 0 HU, which the value cannot.
    """
    out = np.zeros((10, feat.shape[1], feat.shape[2]), np.float32)
    for v in (0, 1):
        m = feat[v * 4 + 3] > 0
        for c in range(4):
            a = feat[v * 4 + c]
            s = a[m]
            mu, sd = (s.mean(), s.std() + 1e-6) if s.size else (0.0, 1.0)
            out[v * 4 + c] = np.where(m, (a - mu) / sd, 0.0)
        out[8 + v] = m.astype(np.float32)
    return out


# --------------------------------------------------------------------------------- network
def _blk(a, b):
    return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.GELU(),
                         nn.Conv2d(b, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.GELU(),
                         nn.MaxPool2d((1, 2)))


class LandmarkNet(nn.Module):
    """Predicts a distribution over z for each landmark; the app reads its expectation and spread.

    Every pool is (1,2) -- lateral only. Lateral detail can be discarded because the label does not
    live there, but z resolution IS the answer, so it is kept at full 256 rows end to end. The
    dilated 1-D trunk then gives each row a receptive field spanning the whole animal, which it needs
    in order to know which end is the head.
    """

    def __init__(self, cin=10, n=5, w=128):
        super().__init__()
        self.enc = nn.Sequential(_blk(cin, 32), _blk(32, 48), _blk(48, 64), _blk(64, 96), _blk(96, w))
        self.proj = nn.Conv1d(2 * w, w, 1)
        self.trunk = nn.ModuleList([
            nn.Sequential(nn.Conv1d(w, w, 3, padding=d, dilation=d, bias=False),
                          nn.GroupNorm(8, w), nn.GELU())
            for d in (1, 2, 4, 8, 16, 32, 64)])
        self.ctx = nn.Sequential(nn.Linear(w, w), nn.GELU(), nn.Linear(w, w))
        self.head = nn.Conv1d(w, n, 1)

    def forward(self, x):
        h = self.enc(x)
        h = self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))
        h = h + self.ctx(h.mean(-1)).unsqueeze(-1)
        for t in self.trunk:
            h = h + t(h)
        return self.head(h)


def soft_argmax(logits):
    """Expectation of the predicted distribution, and the distribution itself."""
    p = logits.softmax(-1)
    pos = (torch.arange(logits.shape[-1], device=logits.device, dtype=p.dtype) + 0.5) / logits.shape[-1]
    return (p * pos).sum(-1), p


# --------------------------------------------------------------------------------- inference
def predict(vol: np.ndarray, mask: np.ndarray | None, nets, device="cpu", tta_flip=True):
    """Landmarks for one volume. Returns z planes in VOXELS, direction, and a spread per landmark.

    `nets` is a list of fold models, averaged in PROBABILITY space rather than by averaging their
    scalar answers: two folds disagreeing about which of two candidate planes is right should give a
    visibly bimodal, low-confidence result rather than a confident answer halfway between them.

    tta_flip averages the volume with its craniocaudal mirror. It is nearly free -- the model is
    small and runs on projections -- and the flip is an exact symmetry of the training distribution,
    so it costs nothing in correctness.
    """
    A = animal_mask(vol) if mask is None else mask
    x = torch.from_numpy(standardise(features(vol, A))[None]).to(device)
    Z = vol.shape[0]
    acc = None
    with torch.no_grad():
        for net in nets:
            net.eval()
            for flip in ((False, True) if tta_flip else (False,)):
                xi = torch.flip(x, dims=[2]) if flip else x
                p = net(xi).softmax(-1)
                if flip:
                    p = torch.flip(p, dims=[2])
                acc = p if acc is None else acc + p
    p = (acc / acc.sum(-1, keepdim=True))[0].cpu().numpy()          # (5, ZR)

    pos = (np.arange(p.shape[-1]) + 0.5) / p.shape[-1]
    frac = (p * pos).sum(-1)
    sd_mm = np.sqrt(((p * (pos[None] - frac[:, None]) ** 2).sum(-1))) * Z * ISO
    z = {k: float(frac[i] * Z - 0.5) for i, k in enumerate(LM)}

    # Direction is not a separate prediction: it is where the lung apex sits relative to the
    # bladder. Measured out of fold on all 1186 scans this agrees with the reviewed table on
    # 1182 -- and the four it misses are diagnosable rather than silent. On those the two planes
    # come out 0.4 to 25.1 mm apart, against a 1st percentile of 29.4 mm when the call is right, so
    # the SEPARATION is itself the confidence. Below DIRECTION_MARGIN_MM it catches all four while
    # flagging under 1% of scans, which matters because a direction flip swaps anterior with
    # gluteal and retroperitoneal with perigonadal while every total stays exactly right.
    direction = "high" if z["lung_apex"] > z["bladder"] else "low"
    margin_mm = abs(z["lung_apex"] - z["bladder"]) * ISO
    out = {"z": z, "direction": direction,
           "direction_margin_mm": float(margin_mm),
           "direction_uncertain": bool(margin_mm < DIRECTION_MARGIN_MM),
           "spread_mm": {k: float(sd_mm[i]) for i, k in enumerate(LM)},
           "heatmap": p, "Z": int(Z)}
    out["depot_landmarks"] = {slot: z[name] for slot, name in DEPOT_ALIAS.items()}
    out["ordered"] = _check_order(z, direction)
    return out


def _check_order(z, direction):
    """The depot rules REQUIRE pelvic_caudal < pelvic_cranial < last_rib < shoulder in cc, and raise
    if not. Reporting it here lets the app say which scan failed and why, rather than surfacing a
    traceback from deep inside the partition."""
    order = ["bladder", "kidney_caudal", "diaphragm", "lung_apex"]
    v = [z[k] for k in order]
    return all(v[i] < v[i + 1] for i in range(3)) if direction == "high" \
        else all(v[i] > v[i + 1] for i in range(3))


def load(model_dir, device="cpu"):
    """Every fold checkpoint in a directory, ready for predict()."""
    from pathlib import Path
    nets = []
    for p in sorted(Path(model_dir).glob("fold*.pt")):
        net = LandmarkNet().to(device)
        net.load_state_dict(torch.load(p, map_location=device))
        nets.append(net)
    if not nets:
        raise FileNotFoundError("no fold*.pt in %s" % model_dir)
    return nets
