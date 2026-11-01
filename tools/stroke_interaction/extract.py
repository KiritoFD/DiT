"""F1: extract four-scale features for the whole 26,002-image real panel.

Output is one npz per scale, plus an event table that keeps every event's image
index so F2/F3 can group by writer / script / character / work instead of
treating events as independent samples.

Memory discipline (the plan is explicit): do NOT build one giant Gram matrix, do
NOT cache full RGB patches. We cache
  - G, C: one row per image
  - S, E: flat rows plus an `img` index column
and nothing else. Patches are recomputed only for the small visual plates.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stroke_interaction.features import image_features  # noqa: E402

G_KEYS = ['ink_frac', 'width_p10', 'width_p50', 'width_p90', 'width_cv',
          'cy', 'cx', 'extent_h', 'extent_w', 'fill_ratio']
C_KEYS = [f'cell_{i}{j}' for i in range(3) for j in range(3)]
S_KEYS = ['y', 'x', 'width', 'ink_frac', 'grad_mag', 'dir', 'dir_spread']
E_NUM = ['y', 'x', 'mass', 'asym_ud', 'asym_lr', 'ink_frac', 'degree', 'grad_mag',
         'orient_entropy', 'orient_peak', 'turn_angle', 'end_area', 'end_elong',
         'end_peak', 'n_branches']
E_KINDS = ['end', 'junction', 'turn']


def work_key(row):
    """Work identity, same convention as the existing 48 protocol."""
    import hashlib
    src = row.get('src_image_path') or row.get('source') or row['image_path']
    return hashlib.sha256(('style24h:' + src).encode()).hexdigest()[:16]


def one(args_tuple):
    idx, path, rng_seed, per_kind = args_tuple
    rng = np.random.default_rng(rng_seed + idx)
    try:
        f = image_features(ROOT / path, rng, per_kind=per_kind)
    except Exception as e:                        # noqa: BLE001
        return idx, None, f'{type(e).__name__}: {e}'
    return idx, f, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--per-kind', type=int, default=24)
    ap.add_argument('--workers', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / a.csv, encoding='utf-8')))
    if a.limit:
        rows = rows[:a.limit]
    print(f'[f1x] {len(rows)} images, per_kind={a.per_kind}', flush=True)

    jobs = [(i, r['image_path'], 1234, a.per_kind) for i, r in enumerate(rows)]
    results = {}
    failures = {}
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        for n, (idx, f, err) in enumerate(pool.imap_unordered(one, jobs, chunksize=8), 1):
            if f is None:
                failures[idx] = err
            else:
                results[idx] = f
            if n % 500 == 0:
                print(f'[f1x] {n}/{len(jobs)} done, {len(failures)} failed', flush=True)

    ok = sorted(results)
    print(f'[f1x] extracted {len(ok)}, failures {len(failures)}', flush=True)
    if failures:
        (out / 'extract_failures.json').write_text(
            json.dumps({rows[i]['image_path']: e for i, e in list(failures.items())[:200]},
                       ensure_ascii=False, indent=2), encoding='utf-8')

    G = np.array([[results[i]['G'][k] for k in G_KEYS] for i in ok], np.float32)
    C = np.array([[results[i]['C'][k] for k in C_KEYS] for i in ok], np.float32)

    s_rows, s_img = [], []
    e_rows, e_img, e_kind = [], [], []
    for j, i in enumerate(ok):
        for s in results[i]['S']:
            s_rows.append([s.get(k, 0.0) for k in S_KEYS])
            s_img.append(j)
        for e in results[i]['E']:
            e_rows.append([e.get(k, 0.0) for k in E_NUM])
            e_img.append(j)
            e_kind.append(E_KINDS.index(e['kind']))
    S = np.array(s_rows, np.float32) if s_rows else np.zeros((0, len(S_KEYS)), np.float32)
    E = np.array(e_rows, np.float32) if e_rows else np.zeros((0, len(E_NUM)), np.float32)

    meta = {
        'image_path': np.array([rows[i]['image_path'] for i in ok]),
        'writer': np.array([rows[i]['calligrapher'] for i in ok]),
        'script': np.array([rows[i]['script'] for i in ok]),
        'character': np.array([rows[i]['character'] for i in ok]),
        'work': np.array([work_key(rows[i]) for i in ok]),
        'source': np.array([rows[i].get('source', '') for i in ok]),
    }
    np.savez_compressed(out / 'event_features.npz',
                        G=G, C=C, S=S, E=E,
                        S_img=np.array(s_img, np.int32), E_img=np.array(e_img, np.int32),
                        E_kind=np.array(e_kind, np.int8),
                        G_keys=np.array(G_KEYS), C_keys=np.array(C_KEYS),
                        S_keys=np.array(S_KEYS), E_keys=np.array(E_NUM),
                        E_kinds=np.array(E_KINDS), **meta)

    counts = {}
    for k in E_KINDS:
        counts[k] = int((np.array(e_kind) == E_KINDS.index(k)).sum()) if e_kind else 0
    cov = {'n_images': len(ok), 'n_excluded': len(failures),
           'events_by_kind': counts, 'n_body_points': int(len(S)),
           'failures': len(failures)}
    (out / 'data_coverage.json').write_text(
        json.dumps(cov, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[f1x] events {counts}, body {len(S)}', flush=True)
    print(f'[f1x] wrote {out / "event_features.npz"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
