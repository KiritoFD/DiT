"""F1: real calligraphy features at four scales, with the stroke EVENTS kept in.

The previous local measurement (tools/analysis/local_stroke_features.py) had two
blind spots that make its negative result unusable as evidence:

  1. it kept only plain skeleton interior points --  `valid = sk & (degree==2)
     & ~dilate(junctions)` -- which throws away exactly the regions most likely
     to carry a writer's hand: stroke ends (起收笔), hooks (勾挑), turns (转折)
     and junctions (连接).
  2. it averaged each patch over four mirror flips, which erases the
     left/right and up/down asymmetry. Whether a stroke END is sharp or blunt,
     which way a hook points, how ink is distributed across the inside vs
     outside of a turn -- all of that lives in that asymmetry.

So this module extracts, per image, a fixed budget of stratified events and
describes each one WITHOUT mirror averaging, and WITHOUT discarding endpoints
or junctions. Scales:

  G   whole glyph : ink mass, stroke width quantiles, centroid, extent
  C   structure   : ink density / centroid in a 3x3 grid (composition)
  S   stroke body : on plain interior skeleton points -- width, direction,
                    curvature, along-stroke width change
  E   events      : at ENDPOINTS / JUNCTIONS / high-turn points -- endpoint
                    sharpness and area, turn angle and radius, junction degree,
                    and the local asymmetry that mirror averaging destroyed

Everything is computed on 256x256 images. Geometry comes from a skeleton, so
none of these descriptors is differentiable -- they are for MEASUREMENT, not for
a loss (the plan asks for differentiable proxies separately, in F5).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


EVENT_KINDS = ['end', 'junction', 'turn']


def load_ink(img, size=256):
    """Image -> ink in [0,1] with 1 = full ink, robust to black-on-white or reverse.

    Returns (gray_ink, binary_ink). Background is NOT stripped by default: the
    plan explicitly keeps paper texture because it may itself be a writer cue;
    the caller decides which variant to use.
    """
    from PIL import Image
    im = Image.open(img).convert('RGBA')
    a = np.array(im).astype(np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + 255.0 * (1 - alpha)
    g = rgb.mean(-1) / 255.0
    if g.mean() > 0.5:            # light paper -> ink is the dark part
        ink = 1.0 - g
    else:
        ink = g
    if ink.shape != (size, size):
        im2 = Image.fromarray((np.clip(ink, 0, 1) * 255).astype(np.uint8))
        ink = np.array(im2.resize((size, size), Image.BICUBIC), np.float32) / 255.0
    return ink, (ink > 0.5).astype(np.uint8)


def width_map(binary):
    """Stroke half-width at every ink pixel via a distance transform."""
    dt = ndi.distance_transform_edt(binary)
    return dt


def skeleton_graph(sk):
    """8-neighbour degree at every skeleton pixel, plus its coordinates."""
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode='constant') - sk
    deg = nb * sk
    ys, xs = np.nonzero(sk)
    return deg, ys, xs


def _local_patch(a, y, x, r):
    """(2r+1)^2 patch centred on (y,x), zero-padded at the border."""
    y0, y1 = max(0, y - r), min(a.shape[0], y + r + 1)
    x0, x1 = max(0, x - r), min(a.shape[1], x + r + 1)
    p = np.zeros((2 * r + 1, 2 * r + 1), a.dtype)
    # destination offset = where (y0,x0) sits inside the padded patch
    oy, ox = y0 - (y - r), x0 - (x - r)
    p[oy:oy + (y1 - y0), ox:ox + (x1 - x0)] = a[y0:y1, x0:x1]
    return p


def event_features(ink, binary, sk, deg, ys, xs, y, x, kind, r=8):
    """Describe one local event. NO mirror averaging: asymmetry is the point."""
    w = width_map(binary) if False else None
    p = _local_patch(ink, y, x, r)
    b = _local_patch(binary, y, x, r).astype(np.float32)
    # ink amount and spread around the event
    mass = float(p.sum())
    # directional asymmetry WITHOUT flipping: compare the four half-planes as-is
    up, dn = p[:r + 1].sum(), p[r:].sum()
    lf, rt = p[:, :r + 1].sum(), p[:, r:].sum()
    out = {
        'kind': kind,
        'y': float(y) / ink.shape[0],
        'x': float(x) / ink.shape[1],
        'mass': mass,
        'asym_ud': float((dn - up) / (dn + up + 1e-6)),
        'asym_lr': float((rt - lf) / (rt + lf + 1e-6)),
        'ink_frac': float(b.mean()),
        'degree': float(deg[int(y), int(x)]),
    }
    # orientation-weighted extent: which way does the ink extend from here?
    gy, gx = np.gradient(p)
    out['grad_mag'] = float(np.hypot(gy, gx).mean())
    if kind == 'end':
        # sharpness: how fast ink falls off along the outward direction, and the
        # area of the end blob. A sharp 收笔 vs a blunt one differ here.
        thr = np.percentile(p, 60)
        region = p > thr
        out['end_area'] = float(region.sum())
        yy, xx = np.nonzero(region)
        if len(yy) > 2:
            cov = np.cov(np.stack([yy, xx]))
            ev = np.linalg.eigvalsh(cov) + 1e-9
            out['end_elong'] = float(np.sqrt(ev[1] / ev[0]))       # 1 = round, >1 = elongated
        else:
            out['end_elong'] = 1.0
        out['end_peak'] = float(p.max())
    elif kind in ('junction', 'turn'):
        # turning: the principal gradient directions across the patch
        m = np.hypot(gx, gy) > np.percentile(np.hypot(gx, gy), 80)
        if m.sum() > 8:
            ang = np.arctan2(gy[m], gx[m])
            ang = np.mod(ang, np.pi)                              # orientation, not direction
            h, _ = np.histogram(ang, bins=8, range=(0, np.pi), weights=np.hypot(gx, gy)[m])
            h = h / (h.sum() + 1e-9)
            out['orient_entropy'] = float(-(h * np.log(h + 1e-9)).sum())
            out['orient_peak'] = float(h.max())
            out['turn_angle'] = float(np.degrees(np.std(ang)) * 2)   # spread of local normals
        else:
            out.update(orient_entropy=0.0, orient_peak=0.0, turn_angle=0.0)
        if kind == 'junction':
            out['n_branches'] = float(deg[int(y), int(x)])
    return out


def glyph_features(ink, binary):
    """Scale G: whole-glyph geometry."""
    w = width_map(binary)[binary > 0]
    ys, xs = np.nonzero(binary)
    if len(ys) < 32 or len(w) < 32:
        return None
    q = np.percentile(w, [10, 50, 90])
    return {
        'ink_frac': float(ink.mean()),
        'width_p10': float(q[0]), 'width_p50': float(q[1]), 'width_p90': float(q[2]),
        'width_cv': float(w.std() / (w.mean() + 1e-6)),
        'cy': float(ys.mean()) / ink.shape[0], 'cx': float(xs.mean()) / ink.shape[1],
        'extent_h': float((ys.max() - ys.min() + 1) / ink.shape[0]),
        'extent_w': float((xs.max() - xs.min() + 1) / ink.shape[1]),
        'fill_ratio': float(binary.mean()),
    }


def structure_features(ink, binary, grid=3):
    """Scale C: composition -- how the ink is distributed over a coarse grid."""
    out = {}
    h, w = binary.shape
    hh, ww = h // grid, w // grid
    for i in range(grid):
        for j in range(grid):
            cell = ink[i * hh:(i + 1) * hh, j * ww:(j + 1) * ww]
            out[f'cell_{i}{j}'] = float(cell.mean())
    return out


def stroke_body_features(ink, binary, deg, ys, xs, rng, budget=96):
    """Scale S: plain interior skeleton points -- width, direction, curvature."""
    interior = np.array([d == 2 for d in deg[ys, xs]])
    iy, ix = ys[interior], xs[interior]
    if len(iy) < 8:
        return []
    dmap = width_map(binary)
    pick = rng.choice(len(iy), size=min(budget, len(iy)), replace=False)
    grad_y, grad_x = np.gradient(ink)
    feats = []
    for p in pick:
        y, x = int(iy[p]), int(ix[p])
        r = 6
        pch = _local_patch(ink, y, x, r)
        gy, gx = np.gradient(pch)
        mag = np.hypot(gy, gx)
        m = mag > np.percentile(mag, 75)
        feat = {
            'kind': 'body',
            'y': float(y) / ink.shape[0], 'x': float(x) / ink.shape[1],
            'width': float(dmap[y, x]),
            'ink_frac': float(_local_patch(binary, y, x, r).mean()),
            'grad_mag': float(mag.mean()),
        }
        if m.sum() > 4:
            ang = np.arctan2(gy[m], gx[m])
            feat['dir'] = float(np.mod(np.median(ang), np.pi))
            feat['dir_spread'] = float(np.std(ang))
        else:
            feat['dir'] = 0.0
            feat['dir_spread'] = 0.0
        feats.append(feat)
    return feats


def extract_events(ink, binary, rng, per_kind=40, turn_thr=0.35):
    """Pick a stratified, non-redundant sample of ENDS, JUNCTIONS and TURNS.

    Sampling is spatially spread: we walk a coarse grid and take at most a few
    events per cell, so one dense region cannot dominate (the plan explicitly
    forbids counting many neighbouring skeleton points as independent samples).
    """
    sk = skeletonize(binary > 0)
    deg, ys, xs = skeleton_graph(sk)
    ends = (deg[ys, xs] == 1)
    junc = (deg[ys, xs] >= 3)
    ey, ex = ys[ends], xs[ends]
    jy, jx = ys[junc], xs[junc]

    # turns: skeleton points whose local ink has high orientation spread
    gy, gx = np.gradient(ink)
    ty, tx = [], []
    if len(ys) > 0:
        inner = np.array([d == 2 for d in deg[ys, xs]])
        cand_y, cand_x = ys[inner], xs[inner]
        if len(cand_y) > 24:
            pick = rng.choice(len(cand_y), size=min(600, len(cand_y)), replace=False)
            score = []
            for p in pick:
                y, x = int(cand_y[p]), int(cand_x[p])
                r = 5
                pgy = _local_patch(gy, y, x, r)
                pgx = _local_patch(gx, y, x, r)
                mag = np.hypot(pgy, pgx)
                m = mag > np.percentile(mag, 70)
                if m.sum() < 6:
                    score.append(0.0)
                    continue
                ang = np.arctan2(pgy[m], pgx[m])
                # circular spread of orientation angle doubled
                c = np.cos(2 * ang).mean()
                s = np.sin(2 * ang).mean()
                score.append(float(1.0 - np.hypot(c, s)))
            score = np.array(score)
            thr = np.percentile(score, 100 * (1 - turn_thr)) if len(score) else 1.0
            sel = pick[score >= max(thr, 0.02)]
            ty, tx = cand_y[sel], cand_x[sel]

    def spread_pick(py, px, n):
        if len(py) == 0:
            return []
        cells = {}
        for y, x in zip(py, px):
            cells.setdefault((int(y) // 32, int(x) // 32), []).append((y, x))
        keys = sorted(cells)
        out = []
        i = 0
        while len(out) < min(n, len(py)) and i < n * 4:
            k = keys[i % len(keys)]
            bucket = cells[k]
            if bucket:
                out.append(bucket.pop(0))
            i += 1
            if all(len(cells[k]) == 0 for k in keys):
                break
        return out

    events = []
    for kind, (py, px) in (('end', (ey, ex)), ('junction', (jy, jx)),
                           ('turn', (np.array(ty, int), np.array(tx, int)))):
        for y, x in spread_pick(py, px, per_kind):
            events.append(event_features(ink, binary, sk, deg, ys, xs, int(y), int(x), kind))
    return events, sk, deg


def image_features(img_path, rng, per_kind=40, body_budget=96, size=256):
    """Full four-scale descriptor for one image."""
    ink, binary = load_ink(img_path, size=size)
    g = glyph_features(ink, binary)
    if g is None:
        return None
    out = {'G': g, 'C': structure_features(ink, binary)}
    events, sk, deg = extract_events(ink, binary, rng, per_kind=per_kind)
    ys, xs = np.nonzero(sk)
    out['S'] = stroke_body_features(ink, binary, deg, ys, xs, rng, budget=body_budget)
    out['E'] = events
    return out
