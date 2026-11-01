"""Collect every official eval summary + T3 report into one machine-readable index.

Numbers are never recomputed here: `eval_stdskel_summary.csv` is written by the
official eval kernel and `t3_report.json` by the T3 tool, so this only gathers and
joins them. The `eval_csv` column travels with each row because the historical
lineage was measured on two different panels (187 vs 100) and the table must not
silently mix them.

Run on each machine; the two JSON indexes are merged by `build_table.py`.
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'assets/metrics_index'
SKIP = ('_archive_', '/smoke', 'smoke_test')

KEEP = ('exp', 'step', 'set', 'cfg', 'steps', 'eval_csv', 'n', 'ssim_mean',
        'ssim_med', 'mse_mean', 'lpips_mean', 'ink_ssim_mean', 'ink_iou_mean',
        'skel_iou_mean', 'frag_ratio', 'hole_pred', 'hole_gt', 'soft_ink_ratio',
        'near_blank_rate', 'ckpt_sha256')


def skipped(p):
    s = p.as_posix()
    return any(t in s for t in SKIP)


def t3_of(eval_dir):
    """Find the T3 report that belongs to this evaluation directory."""
    cands = sorted(eval_dir.rglob('t3_report.json'))
    if not cands:
        # historical runs keep it one level up, next to the eval dir
        cands = sorted((eval_dir.parent).rglob('t3_report.json'))
    if not cands:
        return None
    best = None
    for c in cands:
        try:
            d = json.loads(c.read_text(encoding='utf-8'))
        except Exception:
            continue
        # Prefer the richest report (most writer pairs).
        if best is None or (d.get('n_pairs') or 0) > (best.get('n_pairs') or 0):
            best = d
            best['_path'] = str(c.relative_to(ROOT))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(ROOT.rglob('eval_stdskel_summary.csv')):
        if skipped(p):
            continue
        try:
            recs = list(csv.DictReader(open(p, encoding='utf-8-sig')))
        except Exception as e:
            print(f'skip {p}: {e}')
            continue
        for r in recs:
            row = {k: r.get(k) for k in KEEP if k in r}
            row['summary_path'] = str(p.relative_to(ROOT))
            t3 = t3_of(p.parent)
            if t3:
                row['t3_ocr_top1'] = (t3.get('ocr') or {}).get('top1')
                row['t3_ocr_top5'] = (t3.get('ocr') or {}).get('top5')
                row['t3_ocr_n'] = (t3.get('ocr') or {}).get('n')
                row['direction'] = t3.get('direction')
                row['amplitude'] = t3.get('amplitude')
                row['n_pairs'] = t3.get('n_pairs')
                row['t3_path'] = t3['_path']
            rows.append(row)

    host = sys.argv[1] if len(sys.argv) > 1 else 'unknown'
    out = OUT / f'index_{host}.json'
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'[{host}] {len(rows)} summary rows from '
          f'{len({r["summary_path"] for r in rows})} panels -> {out}')
    with_t3 = [r for r in rows if r.get('direction') is not None]
    print(f'[{host}] {len(with_t3)} rows have T3 (OCR/direction)')


if __name__ == '__main__':
    main()
