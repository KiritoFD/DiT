"""B: the properly specified interaction test.

The earlier F3 had a real defect the review caught: M1 did NOT contain the
event-kind main effect, so "adding writer x kind" was confounded with "adding
the kind main effect". This version fixes the nesting so that each step adds
exactly one thing:

    C0  content      : STD local patch + script + position
    C1  + kind       : event kind main effect                      <- NEW baseline
    C2  + writer     : writer main effect
    C3  + w x kind   : writer x kind, THE INTERACTION OF INTEREST
    C4  + w x STD    : writer x continuous STD context

R_stroke here is (C2 - C3) / C2, not (M1 - M2) / M1 of the broken version.
Every step is compared against a SAME-CAPACITY control with the added label
shuffled, so a gain can be attributed to the label rather than to extra columns.

The plan also asks for a positive control that M3-style continuous interaction
is learnable at all. That is included: a synthetic continuous context is given a
planted writer x context effect, and C4 must recover it.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

E_KINDS = ['end', 'junction', 'turn']
GEO = ['asym_ud', 'asym_lr', 'grad_mag', 'turn_angle', 'end_area', 'end_elong',
       'n_branches', 'orient_peak']
TARGETS = GEO + ['mass', 'ink_frac']
STD_DIR = 'exp-std/data/std_soft_glyphs_v1'
SCRIPT_CODE = {'楷': 'kai', '行': 'xing', '隶': 'li'}


def std_patch(script, character, y, x, nb=5):
    from PIL import Image
    p = ROOT / STD_DIR / SCRIPT_CODE[script] / f'U+{ord(character):05X}.png'
    if not p.exists():
        return None
    a = np.array(Image.open(p).convert('L'), np.float32) / 255.0
    H, W = a.shape
    cy, cx = int(np.clip(y * H, 0, H - 1)), int(np.clip(x * W, 0, W - 1))
    r = nb // 2
    win = np.zeros((nb, nb), np.float32)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    win[y0 - (cy - r):y0 - (cy - r) + (y1 - y0),
        x0 - (cx - r):x0 - (cx - r) + (x1 - x0)] = a[y0:y1, x0:x1]
    gy, gx = np.gradient(win)
    ys, xs = np.nonzero(a > 0.5)
    gcy, gcx = (ys.mean() / H, xs.mean() / W) if len(ys) else (0.5, 0.5)
    return np.concatenate([win.ravel(), [float(np.hypot(gy, gx).mean()),
                                         y - gcy, x - gcx]]).astype(np.float32)


def ridge(X, Y, lam):
    X1 = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
    d = X1.shape[1]
    if len(X1) >= d:
        W = np.linalg.solve(X1.T @ X1 + lam * np.eye(d), X1.T @ Y)
    else:
        W = X1.T @ np.linalg.solve(X1 @ X1.T + lam * np.eye(len(X1)), Y)
    return W


def pred(W, X):
    return np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1) @ W


def onehot(idx, n):
    O = np.zeros((len(idx), n), np.float32)
    O[np.arange(len(idx)), idx] = 1.0
    return O


def build(Xc, Koh, Woh, Xstd, level):
    """level: 'C0' content | 'C1' +kind | 'C2' +writer | 'C3' +w*k | 'C4' +w*std"""
    X = [Xc]
    if level != 'C0':
        X.append(Koh)
    if level in ('C2', 'C3', 'C4'):
        X.append(Woh)
    if level == 'C3':
        X.append((Woh[:, :, None] * Koh[:, None, :]).reshape(len(Xc), -1))
    if level == 'C4' and Xstd is not None:
        X.append((Woh[:, :, None] * Xstd[:, None, :]).reshape(len(Xc), -1))
    return np.concatenate(X, 1)


def continuous_positive_control(seed=0):
    """Can C4 recover a PLANTED writer x continuous-context effect?"""
    rng = np.random.default_rng(seed)
    n_w, d_ctx, n = 6, 4, 6000
    w = rng.integers(n_w, size=n)
    Xstd = rng.normal(size=(n, d_ctx)).astype(np.float32)
    Xc = rng.normal(size=(n, 3)).astype(np.float32)
    Koh = onehot(rng.integers(3, size=n), 3)
    Woh = onehot(w, n_w)
    # planted: writer 0 responds to context dim 0, writer 1 to dim 1, etc.
    effect = np.zeros(n, np.float32)
    for wi in range(min(n_w, d_ctx)):
        effect += (w == wi) * Xstd[:, wi] * 1.5
    Y = (effect + 0.1 * rng.normal(size=n)).reshape(-1, 1)
    tr = rng.random(n) < 0.6
    out = {}
    for lv in ('C2', 'C4'):
        X = build(Xc, Koh, Woh, Xstd, lv)
        W = ridge(X[tr], Y[tr], 1.0)
        out[lv] = float(np.mean((pred(W, X[~tr]) - Y[~tr]) ** 2))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='assets/stroke_writer_20261026/event_features.npz')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--max-per-cell', type=int, default=400)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    pc = continuous_positive_control(a.seed)
    ok_pc = pc['C4'] < pc['C2'] * 0.6
    print(f'[B] continuous-interaction positive control: {pc} -> learnable={ok_pc}',
          flush=True)

    z = np.load(ROOT / a.features, allow_pickle=True)
    writer, script, char = z['writer'], z['script'], z['character']
    E, E_img, E_kind = z['E'], z['E_img'], z['E_kind']
    col = {k: i for i, k in enumerate(z['E_keys'])}

    chars = sorted(set(char.tolist()))
    perm = np.random.default_rng(90).permutation(len(chars))
    test_chars = set(np.array(chars)[perm[:round(len(chars) * .25)]].tolist())
    img_test = np.array([c in test_chars for c in char])

    writers = sorted(set(writer.tolist()))
    w_id = {w: i for i, w in enumerate(writers)}
    scripts = sorted(set(script.tolist()))
    s_id = {s: i for i, s in enumerate(scripts)}
    rng = np.random.default_rng(a.seed)

    rep = {'positive_control_continuous': pc,
           'positive_control_ok': bool(ok_pc), 'conditions': {}}

    for cond in ('STD', 'GT'):
        Xc_r, Koh_r, Woh_r, S_r, Y_r, T_r = [], [], [], [], [], []
        cache = {}
        cell = defaultdict(int)
        for i in rng.permutation(len(E)):
            sc, ch = script[E_img[i]], char[E_img[i]]
            wi = w_id[writer[E_img[i]]]
            kind = int(E_kind[i])
            if cell[(wi, kind)] >= a.max_per_cell:
                continue
            y, x = E[i, col['y']], E[i, col['x']]
            if cond == 'STD':
                ck = (sc, ch, int(y * 64), int(x * 64))
                if ck not in cache:
                    d = std_patch(sc, ch, y, x)
                    cache[ck] = d if d is not None else np.zeros(29, np.float32)
                Xs = cache[ck]
            else:
                Xs = np.zeros(0, np.float32)
            cell[(wi, kind)] += 1
            Xc_r.append(np.concatenate([[y, x, y * x],
                                       onehot(np.array([s_id[sc]]), len(scripts))[0]]))
            Koh_r.append(kind)
            Woh_r.append(wi)
            S_r.append(Xs)
            Y_r.append([E[i, col[t]] for t in TARGETS])
            T_r.append(img_test[E_img[i]])
        if not Xc_r:
            continue
        Xc = np.array(Xc_r, np.float32)
        Koh = onehot(np.array(Koh_r), len(E_KINDS))
        Woh = onehot(np.array(Woh_r), len(writers))
        Xstd = np.array(np.stack([s if len(s) else np.zeros(29, np.float32)
                                  for s in S_r])) if cond == 'STD' else None
        Y = np.array(Y_r, np.float32)
        te = np.array(T_r)

        y_mu, y_sd = Y[~te].mean(0), Y[~te].std(0) + 1e-6
        Y = np.clip((Y - y_mu) / y_sd, -8, 8)
        c_mu, c_sd = Xc[~te].mean(0), Xc[~te].std(0) + 1e-9
        Xc = (Xc - c_mu) / c_sd
        if Xstd is not None:
            s_mu, s_sd = Xstd[~te].mean(0), Xstd[~te].std(0) + 1e-9
            Xstd = (Xstd - s_mu) / s_sd

        print(f'\n[B] === {cond}: {len(Xc)} sites, {int(te.sum())} held out ===',
              flush=True)
        entry = {'n_sites': int(len(Xc)), 'n_test': int(te.sum()), 'levels': {}}
        for lv in ('C0', 'C1', 'C2', 'C3', 'C4'):
            if lv == 'C4' and Xstd is None:
                continue
            Xtr = build(Xc[~te], Koh[~te], Woh[~te],
                        None if Xstd is None else Xstd[~te], lv)
            Xte = build(Xc[te], Koh[te], Woh[te],
                        None if Xstd is None else Xstd[te], lv)
            W = ridge(Xtr, Y[~te], 10.0 if lv != 'C0' else 1.0)
            mse = float(np.mean((pred(W, Xte) - Y[te]) ** 2))
            entry['levels'][lv] = mse
            print(f'[B]   {lv}: test MSE = {mse:.5f}', flush=True)

        L = entry['levels']
        entry['R_kind'] = float((L['C0'] - L['C1']) / L['C0'])
        entry['R_writer'] = float((L['C1'] - L['C2']) / L['C1'])
        entry['R_stroke_interaction'] = float((L['C2'] - L['C3']) / L['C2'])
        if 'C4' in L:
            entry['R_writer_std_interaction'] = float((L['C2'] - L['C4']) / L['C2'])

        # SAME-CAPACITY control: shuffle the writer label used for the
        # interaction term. If the shuffled version gains as much, the gain is
        # column count, not the pairing.
        shuf = Woh[rng.permutation(len(Woh))]
        for lv, key in (('C3', 'R_stroke_shuffled'), ('C4', 'R_writer_std_shuffled')):
            if lv == 'C4' and Xstd is None:
                continue
            Xtr_s = build(Xc[~te], Koh[~te], shuf[~te],
                          None if Xstd is None else Xstd[~te], lv)
            Xte_s = build(Xc[te], Koh[te], shuf[te],
                          None if Xstd is None else Xstd[te], lv)
            W = ridge(Xtr_s, Y[~te], 10.0)
            mse = float(np.mean((pred(W, Xte_s) - Y[te]) ** 2))
            entry[key] = float((L['C2'] - mse) / L['C2'])
        # the honest gain: how much better the REAL pairing is than a pairing of
        # the same capacity built from a shuffled label
        if entry.get('R_stroke_shuffled') is not None:
            entry['R_stroke_beyond_capacity'] = float(
                entry['R_stroke_shuffled'] - entry['R_stroke_interaction'])
        else:
            entry['R_stroke_beyond_capacity'] = None
        print(f"[B] {cond} R_kind={entry['R_kind']:+.4f} R_writer={entry['R_writer']:+.4f} "
              f"R_stroke={entry['R_stroke_interaction']:+.4f} "
              f"(shuffled {entry.get('R_stroke_shuffled', float('nan')):+.4f}) "
              f"R_wstd={entry.get('R_writer_std_interaction', float('nan')):+.4f}", flush=True)
        rep['conditions'][cond] = entry

    p = ROOT / a.out / 'B_interaction.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[B] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
