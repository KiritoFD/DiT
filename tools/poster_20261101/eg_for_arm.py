"""Compute E-G writer-verification metrics for an arbitrary eval panel.

The published E-G numbers (docs/1029/domain_metrics/REPORT.md) never covered K or
any D arm, which left the style axis blank for the current mainline node. This
reuses the frozen bank and the frozen ResNet18 so a new panel is directly
comparable to the published rows.

Runs on CPU, so it can be used while a training job owns the GPU.

  python tools/poster_20261101/eg_for_arm.py <eval_dir> [--k 64] [--limit N]

<eval_dir> must contain eval_stdskel_batch.csv and
eval_samples_ctrl/step*/eval200fix/g<i>.png.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

BANK_DIRS = ['assets/eval_unpaired_20261029/4090_style',
             'assets/eval_unpaired_20261029/4090']


def load_bank(prefer=None):
    dirs = ([prefer] if prefer else []) + BANK_DIRS
    for d in dirs:
        root = Path(d)
        rows_f, feat_f = root / 'reference_rows.json', root / 'reference_features.npz'
        if rows_f.is_file() and feat_f.is_file():
            z = np.load(feat_f)
            if 'E-G' in z:
                rows = json.loads(rows_f.read_text())
                feats = np.asarray(z['E-G'], dtype=np.float32)
                if len(rows) != len(feats):
                    raise SystemExit(f'{d}: rows {len(rows)} != feats {len(feats)}')
                return str(root), rows, feats
    raise SystemExit('no frozen E-G bank found; looked in ' + ', '.join(dirs))


def panel_pngs(eval_dir):
    ctrl = Path(eval_dir) / 'eval_samples_ctrl'
    steps = sorted(ctrl.glob('step*'))
    if not steps:
        raise SystemExit(f'no eval_samples_ctrl/step* under {eval_dir}')
    step = steps[-1]
    # strict g<digits>.png -- a plain 'g*.png' glob also matches the gt<i>.png
    # ground-truth tiles and silently doubles the panel
    pngs = [p for p in (step / 'eval200fix').iterdir()
            if p.suffix == '.png' and p.stem[1:].isdigit() and p.stem[0] == 'g']
    pngs.sort(key=lambda p: int(p.stem[1:]))
    return step.name, pngs


def panel_rows(eval_dir, n):
    path = Path(eval_dir) / 'eval_stdskel_batch.csv'
    rows = list(csv.DictReader(open(path, encoding='utf-8-sig')))
    rows = rows[:n]
    for r in rows:
        r['character'] = r['char']
        r.pop('char', None)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('eval_dir')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--tag', default='')
    ap.add_argument('--bank', default='')
    ap.add_argument('--csv', default='')
    a = ap.parse_args()

    sys.path.insert(0, '.')
    from tools.eval_style_v2 import features_eg as eg
    from tools.eval_style_v2 import metrics as M

    bank_root, bank_rows, bank_feats = load_bank(a.bank or None)
    print(f'bank {bank_root}: {len(bank_rows)} rows, dim {bank_feats.shape[1]}',
          flush=True)

    step, pngs = panel_pngs(a.eval_dir)
    if a.limit:
        pngs = pngs[:a.limit]
    rows = (list(csv.DictReader(open(a.csv, encoding='utf-8-sig')))[:len(pngs)]
            if a.csv else panel_rows(a.eval_dir, len(pngs)))
    if len(rows) != len(pngs):
        raise SystemExit(f'rows {len(rows)} != pngs {len(pngs)}')
    print(f'panel {step}: {len(pngs)} images', flush=True)

    q = eg.eg_features([str(p) for p in pngs])
    res, _ = M.evaluate_feature(q, rows, bank_feats, bank_rows, a.k)
    out = dict(tag=a.tag or Path(a.eval_dir).name, step=step, n_png=len(pngs),
               k=a.k, bank=bank_root, n_bank=len(bank_rows),
               n_valid=res.get('n_valid'), auc=res.get('auc'),
               eer=res.get('eer'), map_macro=res.get('map_macro'),
               auc_ci95=res.get('auc_ci95'))
    print('RESULT ' + json.dumps(out, ensure_ascii=False), flush=True)
    print('FULL ' + json.dumps({k: v for k, v in res.items() if not isinstance(v, list)}, default=str), flush=True)


if __name__ == '__main__':
    main()


