"""Dump the exact rows for a set of curated panel substrings.

Kept separate from `build_table.py` so the numbers that end up in the report can be
eyeballed against the raw index without trusting the table builder.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load():
    rows = []
    for p in sorted((ROOT / 'assets/metrics_index').glob('index_*.json')):
        for r in json.loads(p.read_text(encoding='utf-8')):
            r['_host'] = p.stem.replace('index_', '')
            rows.append(r)
    return rows


def main():
    pats = sys.argv[1:] or ['matrix64']
    rows = load()
    for pat in pats:
        print(f'### {pat}')
        hits = [r for r in rows if pat in r['summary_path']]
        if not hits:
            print('   (none)')
        for r in sorted(hits, key=lambda r: r['summary_path']):
            print(f'  {r["summary_path"][-64:]:<64} ssim={float(r["ssim_mean"]):.4f} '
                  f'lpips={float(r.get("lpips_mean") or 0):.4f} '
                  f'iou={float(r.get("ink_iou_mean") or 0):.4f} '
                  f'frag={float(r.get("frag_ratio") or 0):.3f} '
                  f'skel={float(r.get("skel_iou_mean") or 0):.4f} '
                  f'n={r.get("n")} csv={str(r.get("eval_csv"))[-26:]} host={r["_host"]}')


if __name__ == '__main__':
    main()
