"""Shared contracts for read-only capacity diagnostics."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def vae_identity(path):
    base=Path(path)
    files={str(p.relative_to(base)):sha(p) for p in sorted(base.rglob('*'))
           if p.suffix in ('.bin','.safetensors','.json') and '.cache' not in p.parts}
    if not files:raise ValueError(f'No local VAE identity: {path}')
    return files


def summarize(records):
    groups = {}
    for key in ('split', 'writer', 'script', 'character_seen', 't'):
        for r in records:
            group = f'{key}={r[key]}'
            groups.setdefault(group, []).append(r['mse'])
    # Joint buckets are necessary: aggregate writer loss mixes different times.
    for r in records:
        key = f"{r['split']}|t={r['t']}|writer={r['writer']}|seen={r['character_seen']}"
        groups.setdefault(key, []).append(r['mse'])
        groups.setdefault(f"{r['split']}|t={r['t']}", []).append(r['mse'])
    result = {k: {'mean': float(np.mean(v)), 'n_records': len(v)} for k, v in groups.items()}
    for split in sorted({r['split'] for r in records}):
        for time in sorted({r['t'] for r in records}):
            chars = {}
            for r in records:
                if r['split']==split and r['t']==time:
                    chars.setdefault(r['character'], []).append(r['mse'])
            values=np.array([np.mean(v) for v in chars.values()])
            draw=np.random.default_rng(9017).integers(0,len(values),(2000,len(values)))
            result[f'{split}|t={time}']['character_macro_ci95']=np.quantile(values[draw].mean(1),[.025,.975]).tolist()
    return result


def structure(pred, gt):
    """Geometry diagnostics, not OCR or automatic calligraphic error labels."""
    from scipy.ndimage import binary_dilation, distance_transform_edt, label
    from skimage.morphology import skeletonize
    p = pred.mean(-1) < .5; g = gt.mean(-1) < .5
    skel = skeletonize(g); thin = skel & (distance_transform_edt(g) <= 2.5)
    gl, gn = label(g, np.ones((3, 3))); pl, pn = label(p, np.ones((3, 3)))
    missing = sum(int((gl == i).sum()) >= 5 and
                  float(p[gl == i].mean()) < .25 for i in range(1, gn+1))
    merged = 0
    for i in range(1, pn+1):
        ids, counts = np.unique(gl[pl == i], return_counts=True)
        merged += int(((ids > 0) & (counts >= 5)).sum() > 1)
    return {'thin_skeleton_recall': float(binary_dilation(p)[thin].mean()) if thin.any() else None,
            'missing_gt_components': int(missing), 'merged_pred_components': merged,
            'ink_ratio': float(p.sum()/max(g.sum(), 1))}
