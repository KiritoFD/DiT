"""Panel OCR for every panel that has images, using the same recogniser and the same
`ocr_and_ink` path that produced the published panel OCR.

The E-G axis now exists for 45 panels but OCR only existed for a dozen, which is what
kept the two axes from ever being plotted together.

  python tools/poster_20261101/ocr_for_panels.py --jobs eg_jobs.json --out ocr.json
"""
import argparse
import csv
import json
import sys
from pathlib import Path

# arms that carry no experimental meaning and would only pollute the plot
SKIP = ('smoke', 'smoke_test', 'smoke_test2', 'baseline', 'eval_cpu_mid',
        'top10_extended_runs', 'eval_offline_ft')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', default='assets/metrics_index/eg_jobs.json')
    ap.add_argument('--out', default='assets/metrics_index/ocr_panels_20261101.json')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    sys.path.insert(0, '.')

    jobs = json.loads(Path(a.jobs).read_text())
    if a.only:
        keep = set(a.only.split(','))
        jobs = [j for j in jobs if j['arm'] in keep]
    jobs = [j for j in jobs if not any(j['arm'].startswith(s) for s in SKIP)]

    out = Path(a.out)
    done = json.loads(out.read_text()) if out.is_file() else {}

    from tools.f_20261101.eval_arm import ocr_and_ink

    for j in jobs:
        key = f"{j['arm']}|{j['protocol']}"
        if key in done and 'error' not in done[key]:
            print(f'skip {key}', flush=True)
            continue
        rows = list(csv.DictReader(open(j['csv'], encoding='utf-8-sig')))
        try:
            ocr, ink, n = ocr_and_ink(Path(j['samples']), rows, a.device)
        except Exception as e:
            print(f'FAIL {key}: {type(e).__name__} {e}', flush=True)
            done[key] = dict(error=f'{type(e).__name__}: {e}')
            out.write_text(json.dumps(done, ensure_ascii=False, indent=1),
                           encoding='utf-8')
            continue
        done[key] = dict(arm=j['arm'], protocol=j['protocol'], n=n,
                         top1=ocr.get('top1'), top5=ocr.get('top5'),
                         n_ocr=ocr.get('n'), ink=ink)
        print(f"{key:<40} n={n:<4} top1={ocr.get('top1')}", flush=True)
        out.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')

    print(f'wrote {out} ({len(done)} entries)')


if __name__ == '__main__':
    main()
