"""D3: derive hard-character weights from the TRAINING set only, then oversample.

Difficulty is measured on training GT images with the repo's own
`tools/p125/quality.defect_features` (components / fragmentation / width jump /
ink), plus the character's own frequency in the training set. The validation
panel is never consulted, so D3 cannot leak held-out failures into the objective.

Oversampling is fractional and deterministic: a row with weight w is emitted
`1 + (stable_hash < (w-1))` times, so weights like 1.4 are exact on average
instead of collapsing to integer duplication.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, '.')
from tools.p125.quality import defect_features, load_gray   # noqa: E402

FEATURES = ('endpoint_density', 'fragment_share', 'width_jump', 'width_p50', 'ink_ratio')


def stable_unit(*parts):
    h = hashlib.sha256('|'.join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], 'big') / 2 ** 32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='assets/p125_20261030/stage_p2.csv')
    ap.add_argument('--out-dir', default='assets/d_20261101')
    ap.add_argument('--per-char', type=int, default=5, help='GT samples per character')
    ap.add_argument('--hard-frac', type=float, default=0.15)
    ap.add_argument('--mid-frac', type=float, default=0.25)
    ap.add_argument('--hard-weight', type=float, default=2.0)
    ap.add_argument('--mid-weight', type=float, default=1.4)
    ap.add_argument('--max-growth', type=float, default=1.6)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--refresh', action='store_true', help='ignore the feature cache')
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv, encoding='utf-8-sig')))
    print(f'train rows: {len(rows)}', flush=True)
    by_char = defaultdict(list)
    for r in rows:
        by_char[r['character']].append(r)
    print(f'unique characters: {len(by_char)}', flush=True)

    feats = {}
    cache = Path(a.out_dir) / 'char_features.json'
    todo = sorted(by_char)
    if cache.exists() and not a.refresh:
        feats = json.loads(cache.read_text(encoding='utf-8'))
        todo = [c for c in todo if c not in feats]
        print(f'feature cache: {len(feats)} reused, {len(todo)} to measure', flush=True)
    for k, ch in enumerate(todo):
        rs = by_char[ch]
        picked = rs[:: max(1, len(rs) // a.per_char)][: a.per_char]
        vals = defaultdict(list)
        for r in picked:
            p = Path(r['image_path'])
            if not p.exists():
                continue
            try:
                f = defect_features(load_gray(str(p)))
            except Exception:
                continue
            for key in FEATURES:
                if key in f and f[key] is not None:
                    vals[key].append(float(f[key]))
        feats[ch] = {k2: (float(np.mean(v)) if v else np.nan) for k2, v in vals.items()}
        if (k + 1) % 200 == 0:
            print(f'  measured {k+1}/{len(todo)} characters', flush=True)
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(feats, ensure_ascii=False), encoding='utf-8')

    chars = sorted(by_char)
    freq = np.array([len(by_char[c]) for c in chars], dtype=float)

    def z(key, sign=1.0):
        v = np.array([feats[c].get(key, np.nan) for c in chars], dtype=float)
        good = ~np.isnan(v)
        out = np.zeros_like(v)
        if good.sum() < 3:
            return out
        mu, sd = v[good].mean(), v[good].std() or 1.0
        out[good] = sign * (v[good] - mu) / sd
        return out

    # Hard = many stroke endpoints (more tips that can go missing), already
    # fragmented GT, unstable stroke width, thin ink. Rarity is included but
    # deliberately down-weighted (0.5x): if rarity dominated, D3 would just be a
    # "rare characters" experiment and would not touch common but badly written
    # glyphs. `n_components` is NOT used - on 256x256 calligraphy GT it counts
    # paper speckle (150-330 blobs), not stroke structure.
    freq_z = (np.log(freq) - np.log(freq).mean()) / max(1e-6, np.log(freq).std())
    score = (1.0 * z('endpoint_density') + 1.0 * z('fragment_share')
             + 0.5 * z('width_jump') - 1.0 * z('width_p50') - 1.0 * z('ink_ratio')
             - 0.5 * freq_z)
    order = np.argsort(-score)
    n_hard = max(1, int(len(chars) * a.hard_frac))
    n_mid = max(1, int(len(chars) * a.mid_frac))
    weight = {c: 1.0 for c in chars}
    for i in order[:n_hard]:
        weight[chars[i]] = a.hard_weight
    for i in order[n_hard:n_hard + n_mid]:
        weight[chars[i]] = a.mid_weight

    out_rows = []
    for r in rows:
        w = weight[r['character']]
        copies = 1 + (1 if stable_unit(a.seed, r['img_id']) < (w - 1.0) else 0)
        for _ in range(copies):
            out_rows.append(r)

    growth = len(out_rows) / len(rows)
    if growth > a.max_growth:
        raise SystemExit(f'growth {growth:.2f}x exceeds --max-growth {a.max_growth}')
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / 'stage_p2_hard.csv'
    with open(dst, 'w', newline='', encoding='utf-8') as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(out_rows)

    report = dict(source=a.csv, source_rows=len(rows), out_rows=len(out_rows),
                  growth=round(growth, 4), characters=len(chars),
                  hard_chars=[chars[i] for i in order[:n_hard]],
                  mid_chars=[chars[i] for i in order[n_hard:n_hard + n_mid]],
                  weights=dict(hard=a.hard_weight, mid=a.mid_weight, rest=1.0),
                  features_used=list(FEATURES) + ['log_frequency'],
                  top20=[dict(char=chars[i], weight=weight[chars[i]],
                              freq=int(freq[i]),
                              **{k: (None if np.isnan(v) else round(v, 4))
                                 for k, v in feats[chars[i]].items()})
                         for i in order[:20]])
    (out / 'hard_chars.json').write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                         encoding='utf-8')
    print(f'wrote {dst} ({len(out_rows)} rows, {growth:.2f}x)')
    print('hard chars:', ''.join(report['hard_chars'])[:80])
    print('top20:', [(d['char'], d['freq'], d['endpoint_density']) for d in report['top20'][:10]])


if __name__ == '__main__':
    main()
