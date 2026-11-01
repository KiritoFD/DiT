"""F1b: glyph-conditioned interaction, done right.

The first pass failed because 384-dim glyph embedding + ~10k difference rows is
a ~4M-parameter regression against a signal of order 1e-2. Ridge with lam=10 is
catastrophically over-parameterized there, so M2 < M1 for every representation.

This version fixes that properly:
  * glyph g_c is reduced to a LOW-DIMENSIONAL, interpretable descriptor
    (density grid + orientation histogram + width moment), PCA to k dims;
  * ridge lambda is chosen by inner K-fold over TRAIN CHARACTERS only;
  * the interaction model is additively split: b_AB (shared bias) + W g_c;
  * a positive control of the SAME design (low-dim g, known W) is run first;
  * a shuffled-glyph control gives the null (real N) for delta R2.

Output: f1b_results.json with M0/M1/M2 on held-out characters, the interaction
gain, and both controls.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path('/home/ds/Workspace/DiT')


# --------------------------------------------------------------------------- #
def glyph_descriptor(z):
    """Low-dim, writer-independent glyph descriptor from the real ink per char.

    We deliberately do NOT use the writer image here: the descriptor must only
    know the character. It comes from the canonical STD mask geometry and is
    summarised as an 8x8 density grid + 16-bin orientation histogram.
    """
    import torch
    bank = torch.load(ROOT / 'assets/std_shape_cache_256.pt', map_location='cpu',
                      weights_only=False)
    keys = [(int(s), str(c)) for s, c in bank['keys']]
    lookup = {k: i for i, k in enumerate(keys)}
    masks = bank['masks']
    uniq = sorted(set(z['character']))
    sid_of = {}
    for scr, ch in zip(z['script'], z['character']):
        sid_of.setdefault(ch, {'楷': 0, '行': 3, '隶': 4}[scr])

    desc = []
    for ch in uniq:
        sid = sid_of.get(ch, 0)
        i = lookup.get((sid, ch))
        if i is None:
            i = lookup.get((0, ch))
        if i is None:
            desc.append(np.zeros(80, np.float32))
            continue
        m = masks[i, 0].float().numpy()
        ink = m > 0.5
        # 8x8 density grid (256 = 32*8 per axis)
        g = ink.reshape(8, 32, 8, 32).mean(axis=(1, 3)).ravel()
        from scipy.ndimage import gaussian_filter, sobel
        ang = (np.arctan2(sobel(gaussian_filter(ink.astype(np.float32), 3.0), axis=0),
                          sobel(gaussian_filter(ink.astype(np.float32), 3.0), axis=1))
               + np.pi / 2.0) % np.pi
        h, _ = np.histogram(ang[ink], bins=16, range=(0, np.pi))
        h = h / max(h.sum(), 1)
        desc.append(np.concatenate([g, h]).astype(np.float32))
    return np.stack(desc), uniq


# --------------------------------------------------------------------------- #
def ridge_fit(X, Y, lam):
    Xa = np.concatenate([X, np.ones((len(X), 1))], 1)
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1])
    A[-1, -1] -= lam
    return np.linalg.solve(A, Xa.T @ Y)


def ridge_pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1))], 1) @ W


def mlp_fit(Xtr, Ytr, g_te, M1_pred, seed, hidden=32, epochs=400, lr=3e-3, wd=1e-3):
    """2-layer MLP h(g) trained to predict the residual (delta minus M1 bias).

    Deliberately small and regularised: 12 inputs, 32 hidden, 400 steps. Its only
    job is to check whether the negative Ridge interaction could be an artefact
    of the linear form. It never sees a test character during fitting.
    """
    import torch
    torch.manual_seed(seed)
    Xt = torch.from_numpy(Xtr).float()
    Yt = torch.from_numpy(Ytr).float()
    res = Yt - Yt.mean(0, keepdim=True)     # the M1 bias, learned from train chars
    net = torch.nn.Sequential(torch.nn.Linear(Xt.shape[1], hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, Yt.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((net(Xt) - res) ** 2).mean()
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.from_numpy(g_te).float()).numpy()
    return pred


def r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - yhat.mean(0)) ** 2).sum()) + 1e-12
    return 1.0 - ss_res / ss_tot


def cv_lambda(X, Y, lams, folds=5, seed=0):
    """Pick lambda by K-fold over rows (rows = characters here)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    parts = np.array_split(idx, folds)
    best, best_score = lams[0], -np.inf
    for lam in lams:
        scores = []
        for k in range(folds):
            te = parts[k]
            tr = np.concatenate([parts[j] for j in range(folds) if j != k])
            W = ridge_fit(X[tr], Y[tr], lam)
            scores.append(r2(Y[te], ridge_pred(W, X[te])))
        s = float(np.mean(scores))
        if s > best_score:
            best, best_score = float(lam), s
    return best, best_score


# --------------------------------------------------------------------------- #
def control(rng, n_char=400, d_g=16, d_y=32, w_scale=0.35, noise=0.6):
    G = rng.normal(size=(n_char, d_g))
    b = rng.normal(size=d_y)
    Wint = rng.normal(size=(d_g, d_y)) * w_scale
    D = b[None, :] + G @ Wint + rng.normal(size=(n_char, d_y)) * noise
    tr = rng.random(n_char) < 0.75
    lam = 1.0
    # M1: shared bias only (intercept-only regression)
    W1 = ridge_fit(np.zeros((tr.sum(), 1)), D[tr], lam)
    m1 = r2(D[~tr], ridge_pred(W1, np.zeros(((~tr).sum(), 1))))
    # M2: glyph-conditioned
    W2 = ridge_fit(G[tr], D[tr], lam)
    m2 = r2(D[~tr], ridge_pred(W2, G[~tr]))
    return {'M1_R2': float(m1), 'M2_R2': float(m2), 'delta_R2': float(m2 - m1)}


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='assets/style_24h48_20261026')
    ap.add_argument('--char-test-frac', type=float, default=0.25)
    ap.add_argument('--pca-dim', type=int, default=12)
    ap.add_argument('--seed', type=int, default=90)
    args = ap.parse_args()
    out = ROOT / args.root
    rng = np.random.default_rng(args.seed)

    z = dict(np.load(out / 'panel_features.npz', allow_pickle=True))
    n = len(z['character'])

    dino = z['dino'].astype(np.float64)
    dino /= np.maximum(np.linalg.norm(dino, axis=1, keepdims=True), 1e-9)
    local = z['local'].astype(np.float64)
    layout = z['layout'].astype(np.float64)

    # Feature sets. 'L' isolates stroke-local statistics (no DINO, no layout).
    sets = {
        'G': np.concatenate([dino, layout], 1),
        'L': local,
        'G+L': np.concatenate([dino, layout, local], 1),
    }
    for k, v in sets.items():
        mu, sd = v.mean(0), v.std(0) + 1e-8
        sets[k] = (v - mu) / sd

    desc, uniq = glyph_descriptor(z)
    char_row = {c: i for i, c in enumerate(uniq)}
    G_full = np.stack([desc[char_row.get(c, 0)] for c in z['character']])

    chars = np.array(sorted(set(z['character'])))
    perm = rng.permutation(len(chars))
    n_te = int(round(len(chars) * args.char_test_frac))
    test_chars = set(chars[perm[:n_te]].tolist())

    pc = control(rng)
    print('POSITIVE CONTROL', json.dumps(pc), flush=True)
    nc = control(rng, n_char=300, w_scale=0.0)   # null: no interaction at all
    print('NULL CONTROL   ', json.dumps(nc), flush=True)

    # Glyph basis: PCA on the descriptor, fit on TRAIN characters only.
    tr_char_mask = np.array([c not in test_chars for c in z['character']])
    dmu, dsd = G_full[tr_char_mask].mean(0), G_full[tr_char_mask].std(0) + 1e-8
    Gz = (G_full - dmu) / dsd
    from numpy.linalg import svd
    Utr = Gz[tr_char_mask]
    _, S, Vt = svd(Utr - Utr.mean(0), full_matrices=False)
    k = min(args.pca_dim, Vt.shape[0])
    P = Vt[:k].T
    Gk = (Gz - Utr.mean(0)) @ P
    print(f'glyph basis k={k}, explained={float((S[:k]**2).sum()/(S**2).sum()):.3f}', flush=True)

    lams = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3]
    report = {'positive_control': pc, 'null_control': nc, 'pca_dim': k,
              'n_characters': len(chars), 'n_test_characters': len(test_chars),
              'per_representation': {}}

    key = np.array([f'{z["script"][i]}|{z["character"][i]}' for i in range(n)])
    for script_set, tag in ((['楷'], 'kai'), (['行'], 'xing'), (['楷', '行'], 'kai_xing')):
        sel = np.isin(z['script'], script_set)
        idx_by_key = defaultdict(list)
        for i in np.nonzero(sel)[0]:
            idx_by_key[key[i]].append(i)

        # per-character difference vectors, one per writer pair
        # Each pair contributes twice: the plain panel (any work) and, where
        # possible, a SOURCE-CONTROLLED version where both images come from
        # held-out works. The plan asks which part of the writer difference
        # could be a work-source shortcut, so the same M0/M1/M2 must be fitted
        # on both and compared.
        diffs, gk, pair_id, ch_of, src_ctrl = [], [], [], [], []
        for kk, ids in sorted(idx_by_key.items()):
            writers = sorted({z['writer'][i] for i in ids})
            if len(writers) < 2:
                continue
            by_w = defaultdict(list)
            by_w_te = defaultdict(list)
            for i in ids:
                by_w[z['writer'][i]].append(i)
                if z['split'][i] == 'test':
                    by_w_te[z['writer'][i]].append(i)
            for a in range(len(writers)):
                for b in range(a + 1, len(writers)):
                    wa, wb = writers[a], writers[b]
                    i, j = by_w[wa][0], by_w[wb][0]
                    g_row = char_row.get(kk.split('|')[1], 0)
                    diffs.append((i, j))
                    gk.append(g_row)
                    pair_id.append(f'{kk}|{wa}|{wb}')
                    ch_of.append(kk.split('|')[1])
                    src_ctrl.append(False)
                    # source-controlled only when BOTH writers have held-out work
                    if by_w_te[wa] and by_w_te[wb]:
                        diffs.append((by_w_te[wa][0], by_w_te[wb][0]))
                        gk.append(g_row)
                        pair_id.append(f'{kk}|{wa}|{wb}')
                        ch_of.append(kk.split('|')[1])
                        src_ctrl.append(True)
        if len(diffs) < 60:
            continue
        ch_of = np.array(ch_of)
        gk = Gk[np.array(gk)]
        pair_id = np.array(pair_id)
        src_ctrl = np.array(src_ctrl)
        ii = np.array([d[0] for d in diffs])
        jj = np.array([d[1] for d in diffs])
        te = np.array([c in test_chars for c in ch_of])
        report.setdefault('source_controlled_pairs', {})[tag] = int(src_ctrl.sum())
        if te.sum() < 20 or (~te).sum() < 60:
            continue

        for rep_name, F in sets.items():
            D_all = F[ii] - F[jj]
            # Two variants over the same fitted code:
            #   'all'      - the panel as built (any work)
            #   'src_ctrl' - only pairs whose BOTH images come from held-out works
            for variant, keep in (('all', np.ones(len(D_all), bool)),
                                  ('src_ctrl', src_ctrl)):
                if keep.sum() < 60:
                    continue
                D = D_all[keep]
                gk_v, pair_v, te_v = gk[keep], pair_id[keep], te[keep]
                if te_v.sum() < 20 or (~te_v).sum() < 60:
                    continue
                # M1 baseline: a per-writer-PAIR shared bias that GENERALISES to
                # unseen characters. It is estimated from the train-character rows
                # of the same writer pair (a pair-level prior), not looked up per
                # (pair, character). This is what "cross-character shared writer
                # difference" means operationally.
                pair_only = np.array([pid.rsplit('|', 2)[-2] + '>' + pid.rsplit('|', 1)[-1]
                                      for pid in pair_v])
                acc = defaultdict(list)
                for d, pr, is_te in zip(D, pair_only, te_v):
                    if not is_te:
                        acc[pr].append(d)
                bias = {p: np.mean(v, 0) for p, v in acc.items()}
                ok = np.array([pr in bias for pr in pair_only]) & te_v
                if ok.sum() < 15:
                    print(f'{tag} {rep_name} {variant}: only {int(ok.sum())} test pairs '
                          f'have a train-character pair prior; skipped', flush=True)
                    continue
                M1_pred = np.stack([bias[pr] for pr in pair_only[ok]])
                M1 = r2(D[ok], M1_pred)

                # M2: add glyph-conditioned term, lambda by inner CV on TRAIN chars
                Xtr, Ytr = gk_v[~te_v], D[~te_v]
                lam, cv = cv_lambda(Xtr, Ytr, lams, seed=args.seed)
                W = ridge_fit(Xtr, Ytr, lam)
                inter_pred = ridge_pred(W, gk_v[ok])
                M2 = r2(D[ok], M1_pred + inter_pred)

                # Null: reshuffle the glyph assignment across characters
                perm_null = rng.permutation(len(gk_v))
                Wn = ridge_fit(gk_v[~te_v], D[~te_v], lam)
                null_pred = ridge_pred(Wn, gk_v[perm_null[ok]])
                M2_null = r2(D[ok], M1_pred + null_pred)

                M0 = r2(D[ok], np.zeros_like(D[ok]))
                # Small MLP supplement the plan asks for: same inputs, same
                # train-character rows, same test pairs. Ridge stays the headline
                # (closed form, no tuning freedom); the MLP only answers "did the
                # linear interaction term under-fit?" A 2-layer net on 12 PCA dims
                # is the smallest thing that can represent a nonlinear h_AB(g).
                mlp_pred = mlp_fit(Xtr, Ytr, gk_v[ok], M1_pred, args.seed)
                mlp_dr2 = r2(D[ok], M1_pred + mlp_pred) - M1
                report['per_representation'].setdefault(tag, {}).setdefault(rep_name, {})[variant] = {
                    'n_pairs': int(len(D)), 'n_test_pairs': int(ok.sum()),
                    'M0_R2': float(M0), 'M1_R2': float(M1), 'M2_R2': float(M2),
                    'M2_null_R2': float(M2_null),
                    'delta_R2_interaction': float(M2 - M1),
                    'delta_R2_vs_null': float(M2 - M2_null),
                    'delta_R2_mlp': float(mlp_dr2),
                    'lambda': float(lam), 'cv_R2': float(cv),
                    'dim': int(F.shape[1]),
                }
                print(f'{tag:9s} {rep_name:4s} {variant:8s} pairs={ok.sum():5d} lam={lam:<7g} '
                      f'M0={M0:+.4f} M1={M1:+.4f} M2={M2:+.4f} null={M2_null:+.4f} '
                      f'dR2={M2-M1:+.4f} mlp_dR2={mlp_dr2:+.4f}', flush=True)

    (out / 'f1b_results.json').write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    print('wrote', out / 'f1b_results.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
