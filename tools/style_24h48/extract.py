"""F1 extraction driver: build G / L / G+L features for the matched panel.

Only panel rows are needed for the M0/M1/M2 comparison (one image per
script x character x writer), which keeps this at ~14.5k forward passes instead
of 26k. A background half (train split only) is also extracted so we can report
how much of the fit is a source shortcut.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/ds/Workspace/DiT')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='assets/style_24h48_20261026')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    out = ROOT / args.root

    import csv
    rows = list(csv.DictReader(open(out / 'panel.csv', encoding='utf-8')))
    if args.limit:
        rows = rows[:args.limit]
    items = []
    for r in rows:
        p = ROOT / r['image_path']
        assert p.exists(), p
        items.append({'path': p})

    from tools.style_24h48.features import extract_all
    print(f'extracting {len(items)} panel images', flush=True)
    feats = extract_all(items, tag='panel')

    meta = {
        'script': np.array([r['script'] for r in rows]),
        'character': np.array([r['character'] for r in rows]),
        'writer': np.array([r['writer'] for r in rows]),
        'work': np.array([r['work'] for r in rows]),
        'split': np.array([r['split'] for r in rows]),
        'image_path': np.array([r['image_path'] for r in rows]),
        'std_path': np.array([r['std_path'] for r in rows]),
    }
    np.savez_compressed(out / 'panel_features.npz',
                        dino=feats['dino'], local=feats['local'], layout=feats['layout'],
                        local_keys=feats['local_keys'], **meta)
    print('saved', out / 'panel_features.npz',
          'dino', feats['dino'].shape, 'local', feats['local'].shape,
          'layout', feats['layout'].shape, flush=True)

    summary = {k: {'n': int(len(np.asarray(v))),
                   'unique': int(len(np.unique(v)))} for k, v in meta.items()}
    (out / 'panel_features.json').write_text(
        json.dumps({'rows': len(rows), 'columns': summary}, ensure_ascii=False, indent=2),
        encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
