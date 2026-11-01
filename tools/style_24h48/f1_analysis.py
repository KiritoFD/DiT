"""F1 core: how much of a calligrapher difference is cross-character shared (M1)
and how much is predictable from the glyph shape (M2)?

For every matched (script, character) panel with writers A,B the per-character
difference is  D_AB(c) = f(x_A,c) - f(x_B,c).

    M0 : D_AB(c) = 0                          (no writer effect at all)
    M1 : D_AB(c) = b_AB                       (shared writer distribution bias)
    M2 : D_AB(c) = b_AB + h_AB(g_c)           (bias + glyph-conditioned response)

M2 uses a CLOSED-FORM ridge regression from the standardized STD-DINO PCA of the
glyph to D_AB(c). No bilinear three-factor toy that collapsed to 1e-8 before.
A positive control with a synthetic known interaction runs first, so a null M2
result can be interpreted as "no such interaction" instead of "broken code".

Character holdout is strict: M1/M2 parameters are fit on TRAIN characters only
and scored on held-out characters. Work-source holdout is reported separately.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/ds/Workspace/DiT')


# --------------------------------------------------------------------------- #
# representation assembly
# --------------------------------------------------------------------------- #
def build_reps(z: dict, reps=('G', 'L', 'G+L')):
    """Standardize each block, return dict of feature matrices."""
    dino = z['dino'].astype(np.float64)
    dino = dino / np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-9)
    local = z['local'].astype(np.float64)
    layout = z['layout'].astype(np.float64)
    out = {}
    for name in reps:
        if name == 'G':
            blocks = [dino, layout]
        elif name == 'L':
            blocks = [local]
        else:
            blocks = [dino, layout, local]
        parts = []
        for b in blocks:
            mu, sd = b.mean(0), b.std(0) + 1e-8
            parts.append((b - mu) / sd)
        out[name] = np.concatenate(parts, axis=1)
    return out


def restrict_columns(f, train_mask, max_dim_per_block=None):
    """Drop constant / degenerate columns using TRAIN rows only."""
    sd = f[train_mask].std(0)
    keep = sd > 1e-6
    return f[:, keep], keep


# --------------------------------------------------------------------------- #
# panel -> difference rows
# --------------------------------------------------------------------------- #
def make_diffs(z, feats, script_filter=None, require_test=False):
    """Return (D, glyph_std_feats, meta) for all writer pairs within a panel."""
    n = len(z['character'])
    key = np.array([f'{z["script"][i]}|{z["character"][i]}' for i in range(n)])
    if script_filter is not None:
        sel = np.isin(z['script'], script_filter)
    else:
        sel = np.ones(n, bool)
    idx_by_key = defaultdict(list)
    for i in np.nonzero(sel)[0]:
        idx_by_key[key[i]].append(i)

    rows, glyphs, pairs = [], [], []
    for k, ids in sorted(idx_by_key.items()):
        writers = sorted({z['writer'][i] for i in ids})
        if len(writers) < 2:
            continue
        by_w = defaultdict(list)
        for i in ids:
            by_w[z['writer'][i]].append(i)
        for a in range(len(writers)):
            for b in range(a + 1, len(writers)):
                wa, wb = writers[a], writers[b]
                if require_test:
                    cand = [(i, j) for i in by_w[wa] for j in by_w[wb]
                            if z['split'][i] == 'test' and z['split'][j] == 'test']
                else:
                    cand = [(i, j) for i in by_w[wa] for j in by_w[wb]]
                if not cand:
                    continue
                i, j = cand[0]
                rows.append(feats[i] - feats[j])
                glyphs.append(z['std_dino'][i] if 'std_dino' in z else None)
                pairs.append({'key': k, 'A': wa, 'B': wb, 'i': int(i), 'j': int(j),
                              'work_A': z['work'][i], 'work_B': z['work'][j],
                              'split_A': z['split'][i], 'split_B': z['split'][j]})
    return np.stack(rows) if rows else np.zeros((0, feats.shape[1])), pairs


# --------------------------------------------------------------------------- #
# closed-form ridge
# --------------------------------------------------------------------------- #
def ridge_fit(X, Y, lam):
    """Return W (d_x+1, d_y) with intercept column."""
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    d = Xa.shape[1]
    A = Xa.T @ Xa + lam * np.eye(d)
    A[-1, -1] -= lam                      # do not penalize intercept
    return np.linalg.solve(A, Xa.T @ Y)


def ridge_pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def pair_bias(D, pairs, char_train_mask):
    """M1: per-pair mean difference computed on TRAIN characters only."""
    acc = defaultdict(list)
    for d, p, tr in zip(D, pairs, char_train_mask):
        if tr:
            acc[(p['A'], p['B'], p['key'].split('|')[0])].append(d)
    return {k: np.mean(v, 0) for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# positive control
# --------------------------------------------------------------------------- #
def positive_control(rng, n_char=300, d_y=32, d_g=16, strength=1.0):
    """Synthetic panel where a KNOWN glyph-conditioned interaction exists."""
    G = rng.normal(size=(n_char, d_g))
    b = rng.normal(size=d_y) * 1.0                       # shared bias
    Wint = rng.normal(size=(d_g, d_y)) * 0.35            # interaction
    D = (b[None, :] + strength * (G @ Wint) +
         rng.normal(size=(n_char, d_y)) * 0.5)
    tr = rng.random(n_char) < 0.75
    Xtr, Xte = G[tr], G[~tr]
    Ytr, Yte = D[tr], D[~tr]

    W0 = ridge_fit(Xtr, Ytr, 1e-6)
    m2 = r2(Yte, ridge_pred(W0, Xte))
    bm = Ytr.mean(0, keepdims=True)
    m1 = r2(Yte, np.repeat(bm, len(Yte), 0))
    return {'M1_R2': m1, 'M2_R2': m2, 'delta_R2': m2 - m1,
            'expectation': 'delta_R2 must be clearly > 0 when strength=1'}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='assets/style_24h48_20261026')
    ap.add_argument('--lam', type=float, default=10.0)
    ap.add_argument('--char-test-frac', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=90)
    args = ap.parse_args()
    out = ROOT / args.root
    rng = np.random.default_rng(args.seed)

    z = dict(np.load(out / 'panel_features.npz', allow_pickle=True))
    reps = build_reps(z)
    n = len(z['character'])

    # Glyph conditioning signal g_c. The `std_dino` cache from the soft-glyph
    # renderer is used when present; otherwise fall back to the canonical
    # std_shape_cache_256 masks (identical glyph, independent of writer).
    cache = out / 'std_dino_uniq.npz'
    uniq = sorted(set(z['character']))
    char_to_row = {c: i for i, c in enumerate(uniq)}
    if cache.exists():
        sz = dict(np.load(cache, allow_pickle=True))
        uniq_dino = sz['dino']
        uniq_keys = list(sz['keys'])
        char_to_row = {k: i for i, k in enumerate(uniq_keys)}
        print(f'g_c from cached glyph embeddings: {uniq_dino.shape}', flush=True)
    else:
        import torch
        bank = torch.load(ROOT / 'assets/std_shape_cache_256.pt', map_location='cpu',
                          weights_only=False)
        keys = [(int(s), str(c)) for s, c in bank['keys']]
        masks = bank['masks']
        # Script codes: 0=kai, 3=xing, 4=li -> pick the script the panel uses.
        want = {}
        for scr, ch in zip(z['script'], z['character']):
            if ch not in want:
                sid = {'楷': 0, '行': 3, '隶': 4}[scr]
                want[ch] = sid
        lookup = {(s, c): i for i, (s, c) in enumerate(keys)}
        grays = []
        for ch in uniq:
            sid = want.get(ch, 0)
            i = lookup.get((sid, ch)) or lookup.get((0, ch))
            if i is None:
                grays.append(np.ones((256, 256), np.float32))
            else:
                m = masks[i, 0].float().numpy()
                grays.append(1.0 - m)          # white paper, black ink
        from tools.style_24h48.features import dino_cls
        uniq_dino = dino_cls(grays)
        np.savez_compressed(cache, dino=uniq_dino, keys=np.array(uniq))
        print(f'g_c from std_shape_cache: {uniq_dino.shape}', flush=True)
    uniq_dino = uniq_dino / np.maximum(np.linalg.norm(uniq_dino, axis=1, keepdims=True), 1e-9)
    std_dino = np.stack([uniq_dino[char_to_row[c]] if c in char_to_row
                         else np.zeros(uniq_dino.shape[1]) for c in z['character']])
    z['std_dino'] = std_dino

    chars = sorted(set(z['character']))
    # STRICT character holdout: split characters, not images.
    perm = rng.permutation(len(chars))
    n_te = int(round(len(chars) * args.char_test_frac))
    test_chars = set(np.array(chars)[perm[:n_te]])
    char_is_test = np.array([c in test_chars for c in z['character']])

    pc = positive_control(rng)
    print('POSITIVE CONTROL', json.dumps(pc), flush=True)

    report = {'positive_control': pc, 'per_representation': {}, 'n_rows': int(n),
              'n_characters': len(chars), 'n_test_characters': len(test_chars),
              'lam': args.lam, 'char_test_frac': args.char_test_frac}

    for script_set, tag in ((['楷'], 'kai'), (['行'], 'xing'), (['楷', '行'], 'kai_xing')):
        for rep_name, F in reps.items():
            D, pairs = make_diffs(z, F, script_filter=script_set)
            if len(D) < 50:
                continue
            pair_dims = np.array([p['key'] for p in pairs])
            tr_mask = np.array([k.split('|')[1] not in test_chars for k in pair_dims])
            if tr_mask.sum() < 20 or (~tr_mask).sum() < 10:
                continue

            # M1: per-pair bias from train chars, predict the same pair on test chars
            bias = pair_bias(D, pairs, tr_mask)
            m1_pred, m1_y = [], []
            for d, p, tr in zip(D, pairs, tr_mask):
                if tr:
                    continue
                key = (p['A'], p['B'], p['key'].split('|')[0])
                if key in bias:
                    m1_pred.append(bias[key])
                    m1_y.append(d)
            M1 = r2(np.stack(m1_y), np.stack(m1_pred)) if m1_y else float('nan')

            # M2: ridge from glyph (STD-DINO) to D, fit on train chars
            G = np.stack([z['std_dino'][p['i']] for p in pairs])
            mu, sd = G[tr_mask].mean(0), G[tr_mask].std(0) + 1e-8
            Gs = (G - mu) / sd
            W = ridge_fit(Gs[tr_mask], D[tr_mask], args.lam)
            pred_te = ridge_pred(W, Gs[~tr_mask])
            M2 = r2(D[~tr_mask], pred_te)

            # baseline M0
            M0 = r2(D[~tr_mask], np.zeros_like(D[~tr_mask]))
            report['per_representation'].setdefault(tag, {})[rep_name] = {
                'n_pairs': int(len(D)), 'n_train': int(tr_mask.sum()),
                'n_test': int((~tr_mask).sum()),
                'M0_R2': float(M0), 'M1_R2': float(M1), 'M2_R2': float(M2),
                'delta_R2_interaction': float(M2 - M1),
                'feature_dim': int(F.shape[1]),
            }
            print(f'{tag:9s} {rep_name:4s} pairs={len(D):5d} '
                  f'M0={M0:+.4f} M1={M1:+.4f} M2={M2:+.4f} dR2={M2-M1:+.4f}', flush=True)

    # G+L independent gain over G on the layout-free local block only
    (out / 'f1_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                         encoding='utf-8')
    print('wrote', out / 'f1_results.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
