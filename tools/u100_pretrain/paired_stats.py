"""Paired per-character comparison against a base model with bootstrap CIs.

Joins eval_stdskel_batch.csv rows by idx (identical eval rows, noise and
protocol across models) and reports, per metric and per challenger: mean paired
delta, 95% bootstrap CI, and the win rate in the metric's better-direction.
Round-1 width CSVs (width_w1, heavy_p90_excess) join the same table.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

LOWER_BETTER = {'lpips', 'frag_ratio', 'width_w1', 'heavy_abs'}
BOOTSTRAP = 10000


def read_batch(path):
    rows = {}
    with open(path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            rows[int(row['idx'])] = row
    return rows


def read_width(path):
    rows = {}
    with open(Path(path) + '.csv' if not str(path).endswith('.csv') else path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows[int(row['idx'])] = row
    return rows


def paired(values_a, values_b, lower_better):
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    delta = b - a
    better = (delta < 0) if lower_better else (delta > 0)
    rng = np.random.default_rng(90)
    n = len(delta)
    boots = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        boots[i] = delta[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(boots, (2.5, 97.5))
    return dict(mean_delta=float(delta.mean()), ci95=[float(lo), float(hi)],
                win_rate=float(better.mean()), n=n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--base-width', default='', help='round1 CSV of the base model (width_w1 per idx)')
    parser.add_argument('--challengers', nargs='+', required=True, help='name=batch_csv[:width_csv]')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    base = read_batch(args.base)
    base_width = read_width(args.base_width) if args.base_width else {}
    metrics = ['ssim', 'lpips', 'ink_iou', 'frag_ratio']
    result = {}
    for spec in args.challengers:
        name, *rest = spec.split('=')
        parts = rest[0].split(':')
        chal = read_batch(parts[0])
        common = sorted(set(base) & set(chal))
        entry = {}
        for m in metrics:
            a = [float(base[i][m]) for i in common]
            b = [float(chal[i][m]) for i in common]
            entry[m] = paired(a, b, lower_better=m in LOWER_BETTER)
        if len(parts) > 1 and parts[1]:
            wc = read_width(parts[1])
            wcommon = [i for i in common if i in wc and i in base_width
                       and wc[i]['width_w1'] and base_width[i]['width_w1']]
            entry['width_w1'] = paired([float(base_width[i]['width_w1']) for i in wcommon],
                                       [float(wc[i]['width_w1']) for i in wcommon], True)
            entry['heavy_p90_excess'] = paired([float(base_width[i]['heavy_p90_excess']) for i in wcommon],
                                               [float(wc[i]['heavy_p90_excess']) for i in wcommon], True)
        result[name] = entry
    Path(args.out).write_text(json.dumps(result, indent=2))
    for name, entry in result.items():
        print(f'== {name} vs base ==')
        for m, st in entry.items():
            if st is None: continue
            print(f"  {m:12s} delta={st['mean_delta']:+.4f} CI95=[{st['ci95'][0]:+.4f},{st['ci95'][1]:+.4f}] win={st['win_rate']:.1%}")


if __name__ == '__main__':
    main()
