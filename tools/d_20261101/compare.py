"""Paired comparison of every D arm against K on the val100 panel.

A 100-image mean difference of 0.002 is not a result by itself, so every delta is
tested per image (same image, same seed, same skeleton) and reported with a
bootstrap CI and a sign test. Columns come from the official eval kernel's
per-image CSV -- nothing here recomputes a metric.
"""
import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / 'scratch/d_eval'
ARMS = ['D1_2500', 'D2_3000', 'D3_11000', 'D3_11000_plus_D1dec']
BASE = 'K_A'
# name -> (higher_is_better, label)
METRICS = {
    'ssim': (True, 'SSIM'),
    'lpips': (False, 'LPIPS'),
    'ink_ssim': (True, 'Ink-SSIM'),
    'ink_iou': (True, 'Ink-IoU'),
    'skel_iou': (True, '容差骨架 F1'),
    'frag_ratio': (False, '碎片比'),
    'soft_ink_ratio': (True, '墨量比 (1.0=GT)'),
    'hole_pred': (False, '空洞率'),
}
RNG = np.random.default_rng(20261101)


def load(arm):
    rows = list(csv.DictReader(open(EVAL / f'{arm}.csv', encoding='utf-8-sig')))
    return {int(r['idx']): r for r in rows}


def boot_ci(d, n=10000):
    m = d.mean()
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(axis=1)
    return m, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def main():
    base = load(BASE)
    for arm in ARMS:
        cur = load(arm)
        keys = sorted(set(base) & set(cur))
        print(f'\n=== {arm}  vs {BASE}   (n={len(keys)} paired images)')
        print(f'{"metric":<18}{"K":>9}{"arm":>9}{"delta":>10}{"95% CI":>22}{"win/tie/loss":>16}  verdict')
        for m, (hib, label) in METRICS.items():
            b = np.array([float(base[k][m]) for k in keys])
            a = np.array([float(cur[k][m]) for k in keys])
            d = a - b
            if not hib:
                d = -d  # flip so positive always means "better"
            mean, lo, hi = boot_ci(d)
            w = int((d > 0).sum())
            t = int((d == 0).sum())
            l = int((d < 0).sum())
            sig = 'significant' if (lo > 0 or hi < 0) else 'n.s.'
            print(f'{label:<18}{b.mean():>9.4f}{a.mean():>9.4f}{mean:>+10.4f}'
                  f'  [{lo:+.4f}, {hi:+.4f}]{w:>8}/{t}/{l:<6}  {sig}')


if __name__ == '__main__':
    main()
