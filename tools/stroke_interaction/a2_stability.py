"""A2: does F2's writer separation survive a change of extraction settings?

If the separability is a property of the calligraphy, it should be roughly
invariant to the ink threshold, the raster size, and the event budget. If it
moves a lot, F2 is measuring the detector.

For each configuration we recompute, on a fixed stratified subsample, the
per-event-kind writer-vs-within separation ratio, then report the spread across
configurations. A large spread means the headline numbers are not trustworthy;
a small spread means they are a property of the data.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.stroke_interaction.features import extract_events  # noqa: E402

E_KINDS = ['end', 'junction', 'turn']
GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
ALLF = GEO + ['mass', 'ink_frac']


def load_resized(p, size, thr):
    from PIL import Image
    im = Image.open(p).convert('L').resize((size, size), Image.BICUBIC)
    a = np.asarray(im, np.float32) / 255.0
    ink = 1.0 - a if a.mean() > 0.5 else a
    return ink, (ink > thr).astype(np.uint8)


def events_of(p, size, thr, rng, per_kind):
    ink, b = load_resized(p, size, thr)
    if b.sum() < 64:
        return {}
    ev, _, _ = extract_events(ink, b, rng, per_kind=per_kind)
    out = defaultdict(list)
    for e in ev:
        out[e['kind']].append([e.get(k, 0.0) for k in ALLF])
    return {k: np.array(v) for k, v in out.items()}


def sep_ratio(groups, rng):
    """groups: {(writer, char): array}; returns between/within energy ratio."""
    from scipy.spatial.distance import cdist

    def en(A, B, cap=120):
        if len(A) > cap:
            A = A[rng.choice(len(A), cap, replace=False)]
        if len(B) > cap:
            B = B[rng.choice(len(B), cap, replace=False)]
        if len(A) < 3 or len(B) < 3:
            return None
        return float(2 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())

    by_w = defaultdict(dict)
    for (w, c), A in groups.items():
        by_w[w][c] = A
    within, cross = [], []
    for w, d in by_w.items():
        cs = sorted(d)
        if len(cs) >= 2:
            h = len(cs) // 2
            A = np.concatenate([d[c] for c in cs[:h]])
            B = np.concatenate([d[c] for c in cs[h:]])
            e = en(A, B)
            if e is not None:
                within.append(e)
    ws = sorted(by_w)
    for i in range(len(ws)):
        for j in range(i + 1, len(ws)):
            shared = sorted(set(by_w[ws[i]]) & set(by_w[ws[j]]))
            if not shared:
                continue
            A = np.concatenate([by_w[ws[i]][c] for c in shared])
            B = np.concatenate([by_w[ws[j]][c] for c in shared])
            e = en(A, B)
            if e is not None:
                cross.append(e)
    if not within or not cross:
        return None
    return float(np.mean(cross)) / max(float(np.mean(within)), 1e-9)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--n-images', type=int, default=600)
    ap.add_argument('--per-kind', type=int, default=16)
    a = ap.parse_args()
    import csv

    rows = list(csv.DictReader(open(ROOT / a.csv, encoding='utf-8-sig')))
    # stratify: a few images per (writer, script) so every stratum contributes
    by = defaultdict(list)
    for r in rows:
        by[(r['calligrapher'], r['script'])].append(r)
    rng = np.random.default_rng(7)
    pick = []
    per = max(3, a.n_images // max(len(by), 1))
    for k, v in by.items():
        idx = rng.choice(len(v), size=min(per, len(v)), replace=False)
        pick += [v[i] for i in idx]
    print(f'[A2] {len(pick)} images across {len(by)} strata', flush=True)

    CONFIGS = [
        ('size256_thr0.50', 256, 0.50, 16),
        ('size256_thr0.35', 256, 0.35, 16),
        ('size256_thr0.65', 256, 0.65, 16),
        ('size192_thr0.50', 192, 0.50, 16),
        ('size320_thr0.50', 320, 0.50, 16),
        ('size256_budget8', 256, 0.50, 8),
        ('size256_budget24', 256, 0.50, 24),
    ]
    results = {}
    for name, size, thr, pk in CONFIGS:
        r2 = np.random.default_rng(11)
        # accumulate per (script, kind) -> groups
        data = defaultdict(lambda: defaultdict(dict))
        for r in pick:
            p = ROOT / r['image_path']
            if not p.exists():
                continue
            evs = events_of(p, size, thr, r2, pk)
            for k, A in evs.items():
                data[r['script']][k][(r['calligrapher'], r['character'])] = A
        out = {}
        for sc, kinds in data.items():
            for k, g in kinds.items():
                s = sep_ratio(g, np.random.default_rng(3))
                if s is not None:
                    out[f'{sc}|{k}'] = round(s, 3)
        results[name] = out
        print(f'[A2] {name}: {out}', flush=True)

    # spread across configurations, per (script, kind)
    keys = sorted({k for v in results.values() for k in v})
    summary = {}
    for k in keys:
        vals = [v[k] for v in results.values() if k in v]
        if len(vals) < 3:
            continue
        mu, sd = float(np.mean(vals)), float(np.std(vals))
        summary[k] = {'mean': round(mu, 3), 'sd': round(sd, 3),
                      'cv': round(sd / max(mu, 1e-9), 3), 'n_configs': len(vals)}
    order = sorted(summary.items(), key=lambda kv: kv[1]['cv'], reverse=True)
    print('\n[A2] stability across extraction settings (lower cv = more robust):')
    for k, v in order:
        print(f'  {k:16s} mean={v["mean"]:8.3f} sd={v["sd"]:7.3f} cv={v["cv"]:6.3f}')

    p = ROOT / a.out / 'A2_extraction_stability.json'
    p.write_text(json.dumps({'configs': results, 'summary': summary},
                            ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[A2] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
