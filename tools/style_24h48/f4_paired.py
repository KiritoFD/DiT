"""F4: is E-G's advantage real, or just an easier test set?

F2 already reports both holdouts, and `char_src` is already the double holdout
(held-out characters AND held-out works), so there is no new panel to build.
What was still missing is the statistical question the plan actually asks: the
two encoders are evaluated on the SAME images, so their accuracies are paired,
and the difference needs a confidence interval that respects the fact that
images cluster inside works.

Design:
  * images  = double holdout (te_char & split=='test')
  * scripts = evaluated separately (never pooled across scripts)
  * score   = Writer Macro Accuracy within script, i.e. every (writer, script)
              stratum contributes equally regardless of how many works it has
  * CI      = bootstrap with WORKS as the resampling unit (images of the same
              work are not independent), paired across encoders (same resampled
              works for both, so the comparison is within-image)

Output: F4_paired.json with per-script macro for both encoders, the paired
difference, its 95% CI, and the remaining independent works per writer.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.style_24h48.f2_ablations import (  # noqa: E402
    load_gray, load_meta, to_tensor, encode_all)


def macro_by_stratum(pred, y, strat):
    """Mean per-stratum accuracy; strata with no images are skipped."""
    acc = defaultdict(list)
    for p, t, s in zip(pred, y, strat):
        acc[s].append(float(p == t))
    per = {k: float(np.mean(v)) for k, v in acc.items()}
    return float(np.mean(list(per.values()))), per


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=90)
    ap.add_argument('--char-test-frac', type=float, default=0.25)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default='assets/style_24h48_20261026/F4_paired.json')
    a = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    z = load_meta()
    writer, char, split = z['writer'], z['character'], z['split']
    script, work, paths = z['script'], z['work'], z['image_path']

    writers = sorted(set(writer.tolist()))
    chars = np.array(sorted(set(char.tolist())))
    perm = np.random.default_rng(a.seed).permutation(len(chars))
    n_te = int(round(len(chars) * a.char_test_frac))
    test_chars = set(chars[perm[:n_te]].tolist())

    # Double holdout: unseen character AND unseen work.
    mask = np.array([c in test_chars for c in char]) & (split == 'test')
    ids = np.nonzero(mask)[0]
    print(f'[f4] double holdout: {len(ids)} images, '
          f'{len({char[i] for i in ids})} chars, {len({work[i] for i in ids})} works',
          flush=True)

    grays = load_gray([paths[i] for i in ids], ROOT)
    X = to_tensor(grays)
    y = np.array([writers.index(writer[i]) for i in ids])
    strat = np.array([f'{writer[i]}|{script[i]}' for i in ids])
    wkey = np.array([work[i] for i in ids])

    preds = {}
    for kind in ('G', 'L'):
        ck = torch.load(ROOT / f'assets/style_24h48_20261026/E{kind}_encoder.pt',
                        map_location='cpu', weights_only=False)
        from tools.style_24h48.f2_encoders import make_encoder
        m = make_encoder(ck['kind'], len(ck['writers']))
        m.load_state_dict(ck['state_dict'])
        m.eval().to(dev)
        _, lg = encode_all(m, grays, dev, bs=64)
        preds[kind] = lg.argmax(1)
        mac, per = macro_by_stratum(preds[kind], y, strat)
        print(f'[f4] E-{kind} overall macro={mac:.4f} strata={len(per)}', flush=True)

    # per script, so a script with few works cannot silently dominate
    rep = {'n_images': int(len(ids)), 'n_chars': int(len({char[i] for i in ids})),
           'n_works': int(len(set(wkey.tolist()))), 'n_writers': len(writers),
           'by_script': {}}

    works_per_stratum = defaultdict(set)
    for s, w in zip(strat, wkey):
        works_per_stratum[s].add(w)
    rep['works_remaining_per_stratum'] = {
        k: len(v) for k, v in sorted(works_per_stratum.items())}

    boot_works = np.array(sorted(set(wkey.tolist())))
    rng = np.random.default_rng(a.seed)
    for script_name in sorted({script[i] for i in ids}):
        sel = np.array([script[i] == script_name for i in ids])
        if sel.sum() < 30:
            continue
        y_s, st_s, wk_s = y[sel], strat[sel], wkey[sel]
        g_s, l_s = preds['G'][sel], preds['L'][sel]
        mac_g, per_g = macro_by_stratum(g_s, y_s, st_s)
        mac_l, per_l = macro_by_stratum(l_s, y_s, st_s)
        diff = mac_g - mac_l

        # Paired bootstrap resampling WORKS: all images of a drawn work enter
        # together, and both encoders see the same drawn works.
        present = np.array(sorted(set(wk_s.tolist())))
        idx_by_work = defaultdict(list)
        for j, w in enumerate(wk_s):
            idx_by_work[w].append(j)
        diffs = []
        for _ in range(a.boot):
            pick = rng.choice(present, size=len(present), replace=True)
            rows = np.concatenate([idx_by_work[w] for w in pick])
            mg, _ = macro_by_stratum(g_s[rows], y_s[rows], st_s[rows])
            ml, _ = macro_by_stratum(l_s[rows], y_s[rows], st_s[rows])
            diffs.append(mg - ml)
        diffs = np.array(diffs)
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        # A difference is credible only if the whole interval is above zero.
        n_pos = int((diffs > 0).sum())
        rep['by_script'][script_name] = {
            'n_images': int(sel.sum()), 'n_works': int(len(present)),
            'n_strata': len(per_g),
            'E-G_macro': round(mac_g, 4), 'E-L_macro': round(mac_l, 4),
            'paired_diff': round(float(diff), 4),
            'ci95': [round(float(lo), 4), round(float(hi), 4)],
            'boot_pos_frac': round(n_pos / a.boot, 4),
            'eg_wins': bool(lo > 0), 'el_wins': bool(hi < 0),
            'per_stratum_E-G': {k: round(v, 3) for k, v in sorted(per_g.items())},
            'per_stratum_E-L': {k: round(v, 3) for k, v in sorted(per_l.items())},
        }
        print(f'[f4] script {script_name}: E-G={mac_g:.4f} E-L={mac_l:.4f} '
              f'diff={diff:+.4f} CI[{lo:+.4f},{hi:+.4f}] works={len(present)}', flush=True)

    verdicts = [v for v in rep['by_script'].values()]
    rep['verdict'] = {
        'scripts_where_EG_wins': sum(1 for v in verdicts if v['eg_wins']),
        'scripts_where_EL_wins': sum(1 for v in verdicts if v['el_wins']),
        'conclusion': ('E-G advantage is credible (CI excludes 0 in every script)'
                       if verdicts and all(v['eg_wins'] for v in verdicts) else
                       'E-G advantage NOT established in every script; see per-script CI'),
    }
    p = ROOT / a.out
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[f4] {rep["verdict"]["conclusion"]}', flush=True)
    print(f'[f4] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
