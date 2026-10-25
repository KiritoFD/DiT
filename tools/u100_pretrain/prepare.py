"""Writer-group datasets for the two U100 pretraining arms.

Shared row numbering across both arms: rows 0-9 are always reserved for the
Top10 writers (fixed raw-ID order from the frozen SupCon contract); generic
source-scoped style groups with >=min-group training images follow; the merged
OTHER bucket takes the last row before the model's null row.

Arm 120k = generic pool + Top10 clean training rows (Top10 rows receive data).
Arm 100k = generic pool only (rows 0-9 stay at their table initialization).
Inline evaluation for both arms is the Top10 validation-100 set, rewritten to
the same group IDs so arm A measures learned rows and arm B measures content
plus untrained rows under one protocol.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}
POOL_COLUMNS = None  # captured from the pool CSV header


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
    parser.add_argument('--reference', default='assets/reference_kxl_20261023/train.csv')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024')
    parser.add_argument('--min-group', type=int, default=32)
    args = parser.parse_args()
    pool = Path(args.pool); out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for existing in ('train_120k.csv', 'train_100k.csv', 'tables/callig_table.npy', 'manifest.json'):
        if (out / existing).exists(): raise FileExistsError(out / existing)
    (out / 'tables').mkdir(exist_ok=True)

    pool_train = read(pool / 'train.csv')
    global POOL_COLUMNS
    POOL_COLUMNS = list(pool_train[0].keys())
    top10_all = read(pool / 'top10_train.csv')
    top10_clean = [r for r in top10_all if r['generic'] == '0']
    if len(top10_clean) != 23485:
        raise ValueError(f'Expected 23485 clean Top10 rows, got {len(top10_clean)}')
    for r in top10_clean:
        if int(r['calligrapher_id']) not in TOP10_RAW:
            raise ValueError('Top10 raw id outside the frozen contract: ' + r['calligrapher_id'])

    reference = read(args.reference)
    style_key = {int(r['img_id']): (r['style_writer_key'], r['style_writer_name']) for r in reference}

    # Generic style groups from source-scoped keys; rows without a join fall into OTHER.
    counts = Counter()
    for r in pool_train:
        if int(r['img_id']) in style_key: counts[style_key[int(r['img_id'])][0]] += 1
    kept = sorted((k for k, n in counts.items() if n >= args.min_group),
                  key=lambda k: (-counts[k], k))
    group_of_key = {k: 10 + i for i, k in enumerate(kept)}
    other_row = 10 + len(kept)
    num_writers = other_row + 1

    def assign(row):
        raw = int(row['calligrapher_id'])
        if raw in TOP10_RAW and row['generic'] == '0':
            return TOP10_RAW[raw]
        entry = style_key.get(int(row['img_id']))
        return group_of_key.get(entry[0], other_row) if entry else other_row

    generic_rows = [dict(r, calligrapher_id=str(assign(r))) for r in pool_train]
    top10_rows = []
    for r in top10_clean:
        top10_rows.append(dict(r, calligrapher_id=str(TOP10_RAW[int(r['calligrapher_id'])])))
    write(out / 'train_100k.csv', generic_rows, POOL_COLUMNS)
    write(out / 'train_120k.csv', generic_rows + top10_rows, POOL_COLUMNS)

    used = Counter(int(r['calligrapher_id']) for r in generic_rows + top10_rows)
    unreachable = [g for g in range(10, num_writers) if g not in used]
    if unreachable: raise ValueError(f'Unreachable generic group rows: {unreachable}')
    if other_row not in used:
        raise ValueError('OTHER bucket is empty; dataset contract expects unjoinable rows')

    validation = read(pool / 'top10_validation.csv')
    for r in validation:
        raw = int(r['calligrapher_id'])
        if raw not in TOP10_RAW: raise ValueError('Validation writer outside Top10: ' + str(raw))
        r['calligrapher_id'] = str(TOP10_RAW[raw])
    write(out / 'validation_val100.csv', validation, list(validation[0].keys()))

    # Tables: rows 0-9 from the frozen SupCon contract; the rest seeded unit rows.
    supcon = np.load(pool / 'writer_table' / 'callig_table.npy')
    if supcon.shape != (10, 32): raise ValueError(f'SupCon table shape {supcon.shape}')
    rng = np.random.default_rng(90)
    rest = rng.normal(size=(num_writers - 10, 32)).astype(np.float32)
    rest /= np.linalg.norm(rest, axis=1, keepdims=True)
    table = np.concatenate([supcon, rest], axis=0)
    np.save(out / 'tables' / 'callig_table.npy', table)
    np.save(out / 'tables' / 'font_table.npy', np.load(pool / 'tables' / 'font_table.npy'))
    for name in ('char_remap.json', 'font_remap.json'):
        (out / 'tables' / name).write_bytes((pool / 'tables' / name).read_bytes())
    identity = {str(g): g for g in range(num_writers)}
    (out / 'tables' / 'callig_remap.json').write_text(json.dumps(identity), encoding='utf-8')

    manifest = dict(
        num_writers=num_writers, top10_raw_ids={str(k): v for k, v in sorted(TOP10_RAW.items())},
        generic_groups=len(kept), other_row=other_row,
        arm_120k_rows=len(generic_rows) + len(top10_rows), arm_100k_rows=len(generic_rows),
        top10_clean_rows=len(top10_clean),
        group_row_counts=dict(Counter({str(g): n for g, n in sorted(used.items())})),
        min_group=args.min_group,
        other_bucket_source=f'{sum(1 for r in pool_train if int(r["img_id"]) not in style_key)} unjoinable pool rows',
        pool_csv_sha256=hashlib.sha256((pool / 'train.csv').read_bytes()).hexdigest(),
        validation_rows=len(validation),
        tables='rows0-9 frozen SupCon init; 10+ seeded unit rows (trainable in pretraining)')
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print('PREPARE_READY', json.dumps({k: manifest[k] for k in
        ('num_writers', 'arm_120k_rows', 'arm_100k_rows', 'generic_groups', 'top10_clean_rows')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
