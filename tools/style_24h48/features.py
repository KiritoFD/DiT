"""F1 features: three complementary representations of a real calligraphy image.

G  (global) : DINOv2-ViT-S/14 CLS (384d) + 18 handcrafted layout/ink scalars
L  (local)  : stroke-level statistics sampled ONLY on the skeleton (never
              background), aggregated into orientation x width histograms,
              width / curvature / ink quantiles, plus per-image moments.
G+L         : concatenation (used by the M1/M2 regressions).

Everything is deterministic and background-invariant: we always re-crop to the
ink bounding box (with a fixed margin) before resizing, so paper borders and
scan framing cannot leak into the representation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path('/home/ds/Workspace/DiT')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def to_gray01(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        if a.shape[-1] == 4:
            alpha = a[..., 3:4].astype(np.float32) / 255.0
            a = a[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        a = a.mean(axis=-1)
    a = a.astype(np.float32)
    if a.max() > 1.0 + 1e-6:
        a = a / 255.0
    return np.clip(a, 0.0, 1.0)


def crop_to_ink(g: np.ndarray, thresh: float = 0.75, margin: float = 0.08) -> np.ndarray:
    """Crop to the ink bbox plus a relative margin; makes framing irrelevant."""
    ink = g < thresh
    if ink.sum() < 8:
        return g
    ys, xs = np.nonzero(ink)
    h, w = g.shape
    mh, mw = int((ys.max() - ys.min() + 1) * margin), int((xs.max() - xs.min() + 1) * margin)
    y0, y1 = max(0, ys.min() - mh), min(h, ys.max() + 1 + mh)
    x0, x1 = max(0, xs.min() - mw), min(w, xs.max() + 1 + mw)
    return g[y0:y1, x0:x1]


_DINO = {}


def dino_model():
    if 'm' not in _DINO:
        import torch
        m = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', trust_repo=True,
                           verbose=False)
        _DINO['m'] = m.eval().requires_grad_(False).cuda()
        _DINO['torch'] = torch
    return _DINO['m'], _DINO['torch']


def dino_cls(grays01: list[np.ndarray], batch: int = 48) -> np.ndarray:
    """CLS embedding for background-cropped grayscale images, 224 input."""
    m, torch = dino_model()
    import torch.nn.functional as F
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    out = []
    for i in range(0, len(grays01), batch):
        chunk = grays01[i:i + batch]
        x = torch.from_numpy(np.stack(chunk)).float().unsqueeze(1).cuda()
        x = F.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True)
        x = x.clamp(0, 1).expand(-1, 3, -1, -1)
        x = (x - mean) / std
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            f = m(x)
        out.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


_LOCAL_KEYS = (
    # width distribution (pen pressure / stroke thickness)
    'w_p10', 'w_p25', 'w_p50', 'w_p75', 'w_p90', 'w_cv', 'w_iqr',
    # curvature (turning: round vs angular)
    'c_p25', 'c_p50', 'c_p75', 'c_p90',
    # ink value on the skeleton (dry vs wet brush)
    'ink_p10', 'ink_p50', 'ink_p90', 'ink_std',
    # local edge contrast
    'edge_p50', 'edge_p90',
    # orientation x width joint histogram, 8 x 6 = 48 cells
    *[f'dw_{i}_{j}' for i in range(8) for j in range(6)],
)


def local_features(g01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Stroke-local feature vector + global layout scalars.

    Returns (local L, layout G_layout). Both background-invariant; L is sampled
    only at skeleton pixels of the ink mask.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter, sobel
    from skimage.morphology import skeletonize

    g = crop_to_ink(g01)
    ink = g < 0.65
    if ink.sum() < 24:
        return np.zeros(len(_LOCAL_KEYS), np.float32), np.zeros(8, np.float32)
    skel = skeletonize(ink)
    if skel.sum() < 12:
        return np.zeros(len(_LOCAL_KEYS), np.float32), np.zeros(8, np.float32)

    edt = distance_transform_edt(ink)
    width = (2.0 * edt[skel]).astype(np.float32)
    wscale = max(float(np.percentile(width, 95)), 1e-6)

    # Orientation from smoothed skeleton geometry (gray inside a stroke is flat).
    sk_s = gaussian_filter(skel.astype(np.float32), 2.0)
    ang = (np.arctan2(sobel(sk_s, axis=0), sobel(sk_s, axis=1)) + np.pi / 2.0) % np.pi
    ang_sk = ang[skel]
    curv = np.hypot(sobel(ang, axis=1), sobel(ang, axis=0))[skel]
    mag = np.hypot(sobel(g, axis=1), sobel(g, axis=0))
    edge = mag[ink]

    jh = np.zeros((8, 6), np.float32)
    wbin = np.clip((width / wscale * 6).astype(int), 0, 5)
    abin = np.clip((ang_sk / np.pi * 8).astype(int), 0, 7)
    np.add.at(jh, (abin, wbin), 1.0)
    jh /= max(jh.sum(), 1e-9)

    def q(a, ps):
        return np.percentile(a, ps) if a.size else np.zeros(len(ps))

    wq = q(width, [10, 25, 50, 75, 90])
    cq = q(curv, [25, 50, 75, 90])
    gq = q(g[skel], [10, 50, 90])
    eq = q(edge, [50, 90])

    local = np.concatenate([
        wq, [float(width.std() / max(width.mean(), 1e-9)),
             float(np.percentile(width, 75) - np.percentile(width, 25))],
        cq,
        gq, [float(g[skel].std())],
        eq,
        jh.ravel(),
    ]).astype(np.float32)

    ys, xs = np.nonzero(ink)
    h, w = g.shape
    bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    box = np.zeros_like(ink)
    box[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
    layout = np.array([
        bw / max(bh, 1e-9),                                   # aspect
        (ys.mean() - (ys.min() + ys.max()) / 2) / max(bh, 1e-9),
        (xs.mean() - (xs.min() + xs.max()) / 2) / max(bw, 1e-9),
        1.0 - ink[box].mean(),                                 # whitespace in box
        ink.mean(),                                            # ink fraction
        float((g[skel] > 0.5).mean()),                         # dry-brush ratio
        float(ink.sum()) / max(h * w, 1),                      # absolute coverage
        float(width.mean()) / max(min(h, w), 1),               # relative pressure
    ], np.float32)
    return local, layout


def extract_all(items: list[dict], tag: str = '') -> dict:
    """items: [{'path': abs_path, ...}]. Returns dict of arrays."""
    from PIL import Image
    grays = []
    for it in items:
        grays.append(to_gray01(Image.open(it['path'])))
    din = dino_cls(grays)
    loc, lay = [], []
    for g in grays:
        l, y = local_features(g)
        loc.append(l)
        lay.append(y)
    return {'dino': din,
            'local': np.stack(loc),
            'layout': np.stack(lay),
            'local_keys': np.array(_LOCAL_KEYS)}
