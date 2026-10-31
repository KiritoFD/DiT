"""Pick the cells that exist in BOTH eval panels, with their row indices.

Row i of a 187-row `eval200fix` panel and row j of a 100-row val100 panel show the
same (calligrapher, character) iff both CSVs agree on that pair — so one poster row
can carry both families. Also reports whether the ground truth is the same source
image, which decides if a single GT column is valid for the whole poster.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P200 = ROOT / 'exp-std/csv/eval200_fixed.csv'
P100 = ROOT / 'scratch/panels/validation_val100.csv'
OUT = ROOT / 'tools/poster_20261101/cells.json'


def rows(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def key(r):
    return (r['calligrapher'], r.get('character') or r.get('char') or '')


def main():
    a, b = rows(P200), rows(P100)
    ia, ib = defaultdict(list), defaultdict(list)
    for i, r in enumerate(a):
        ia[key(r)].append(i)
    for j, r in enumerate(b):
        ib[key(r)].append(j)

    shared = sorted(k for k in ia if k in ib)
    cells, same_gt, diff_gt = [], 0, []
    for w, ch in shared:
        i, j = ia[(w, ch)][0], ib[(w, ch)][0]
        ra, rb = a[i], b[j]
        sa = ra.get('src_image_path') or ra.get('image_path')
        sb = rb.get('src_image_path') or rb.get('image_path')
        if sa == sb:
            same_gt += 1
        else:
            diff_gt.append((w, ch, sa, sb))
        cells.append(dict(writer=w, char=ch, idx200=i, idx100=j,
                          src200=sa, src100=sb, same_gt=(sa == sb)))

    # group by writer, most-populated writer first, for a readable row order
    by_w = defaultdict(list)
    for c in cells:
        by_w[c['writer']].append(c)
    ordered = []
    for w in sorted(by_w, key=lambda w: (-len(by_w[w]), w)):
        ordered.extend(sorted(by_w[w], key=lambda c: c['char']))

    OUT.write_text(json.dumps(
        dict(n=len(ordered), n_same_gt=same_gt,
             writers={w: len(v) for w, v in by_w.items()},
             cells=ordered), ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'shared cells: {len(ordered)}   same GT source: {same_gt}/{len(ordered)}')
    for w in sorted(by_w, key=lambda w: -len(by_w[w])):
        print(f'  {w:<6} {len(by_w[w])}  {"".join(c["char"] for c in by_w[w])}')
    if diff_gt:
        print(f'\n{len(diff_gt)} cells have DIFFERENT gt sources between panels:')
        for w, ch, sa, sb in diff_gt[:6]:
            print(f'  {w}/{ch}: {sa}  vs  {sb}')
    print('\nwrote', OUT)


if __name__ == '__main__':
    main()
