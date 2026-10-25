"""Top10 fine-tuning mixture: clean Top10 rows plus seeded generic replay.

Rows 0-9 receive their Top10 data; the replay keeps the pretrained generic
writer rows and content anchored while the backbone adapts. Both parts reuse
the group numbering from prepare.py, so one identity remap serves them all.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def read(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write(path, rows, fieldnames):
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', default='assets/generic48_kxl_top10_20261022')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024')
    parser.add_argument('--replay-frac', type=float, default=0.2)
    args = parser.parse_args()
    pool = Path(args.pool); out = Path(args.out)
    target = out / 'ft_train.csv'
    if target.exists(): raise FileExistsError(target)

    top10_clean = [r for r in read(pool / 'top10_train.csv') if r['generic'] == '0']
    if len(top10_clean) != 23485: raise ValueError(f'Expected 23485 Top10 rows, got {len(top10_clean)}')
    top10_rows = [dict(r, calligrapher_id=str(TOP10_RAW[int(r['calligrapher_id'])])) for r in top10_clean]

    generic = read(out / 'train_100k.csv')
    fieldnames = list(generic[0].keys())
    replay_n = round(len(top10_rows) * args.replay_frac / (1 - args.replay_frac))
    rng = np.random.default_rng(90)
    replay = [generic[i] for i in rng.choice(len(generic), size=replay_n, replace=False)]
    rows = top10_rows + replay
    write(target, rows, fieldnames)
    record = dict(train_rows=len(rows), top10_rows=len(top10_rows), replay_rows=replay_n,
                  replay_fraction=round(replay_n / len(rows), 4), seed=90,
                  note='rows0-9 trainable Top10; generic rows anchored by replay')
    (out / 'ft_manifest.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print('FT_DATA_READY', json.dumps(record))


if __name__ == '__main__':
    main()
