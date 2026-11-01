"""List the distinct eval panels in the collected indexes, with their head row.

Curating the lineage table needs the real `summary_path` values, not guesses about
directory names; this prints them so the whitelist in `build_table.py` is checked
against what actually exists.
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
IDX = ROOT / 'assets/metrics_index'


def load():
    rows = []
    for p in sorted(IDX.glob('index_*.json')):
        for r in json.loads(p.read_text(encoding='utf-8')):
            r['_host'] = p.stem.replace('index_', '')
            rows.append(r)
    return rows


def main():
    rows = load()
    groups = defaultdict(list)
    for r in rows:
        groups[r['summary_path']].append(r)
    print(f'{len(rows)} rows, {len(groups)} panels\n')
    print(f'{"n":>3} {"ssim":>8} {"lpips":>8} {"iou":>8} {"frag":>7}  panel')
    for path, rs in sorted(groups.items()):
        s = [float(r['ssim_mean']) for r in rs if r.get('ssim_mean')]
        if not s:
            continue
        best = max(rs, key=lambda r: float(r['ssim_mean']))
        print(f'{len(rs):>3} {float(best["ssim_mean"]):>8.4f} '
              f'{float(best.get("lpips_mean") or 0):>8.4f} '
              f'{float(best.get("ink_iou_mean") or 0):>8.4f} '
              f'{float(best.get("frag_ratio") or 0):>7.3f}  {path}')
        print(f'    eval_csv={best.get("eval_csv")}  exp={best.get("exp")} '
              f'step={best.get("step")}  t3={best.get("direction")}')


if __name__ == '__main__':
    main()
