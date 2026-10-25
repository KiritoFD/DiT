"""Remap the Top10 feature bank into pipeline writer-row order and save.

reference_top10_20261024/features.npz stores writers in sorted-NAME order
(that finetune's own mapping); the pipeline uses the frozen raw-ID order
(TOP10_RAW). Names come from that run's data_manifest; raw ids come from the
pool top10_train.csv clean rows.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bank', default='assets/reference_top10_20261024/features.npz')
    parser.add_argument('--manifest', default='assets/reference_top10_20261024/data_manifest.json')
    parser.add_argument('--pool', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    names = json.loads(Path(args.manifest).read_text(encoding='utf-8'))['writers']
    name_to_raw = {}
    for r in csv.DictReader(open(args.pool, encoding='utf-8-sig')):
        if r['generic'] == '0':
            name_to_raw[r['calligrapher']] = int(r['calligrapher_id'])
    remap = {}
    for sorted_idx, name in enumerate(names):
        raw = name_to_raw[name]
        remap[sorted_idx] = TOP10_RAW[raw]
    d = np.load(args.bank)
    writers = d['writers'].copy()
    for old, new in remap.items():
        writers[writers == old] = new
    assert set(np.unique(writers)) <= set(range(10))
    np.savez(out, ids=d['ids'], features=d['features'], writers=writers,
             chars=d['chars'], scripts=d['scripts'])
    (out.with_suffix('.json')).write_text(json.dumps(
        dict(name_order=names, remap={str(k): v for k, v in remap.items()},
             rows=len(writers), order='pipeline TOP10_RAW'), indent=1))
    counts = {int(g): int((writers == g).sum()) for g in np.unique(writers)}
    print('TOP10_BANK_REMAPPED', json.dumps(counts))


if __name__ == '__main__':
    main()
