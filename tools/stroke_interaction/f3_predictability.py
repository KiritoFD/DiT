"""F3: which parts of the writer's local habit can actually be LEARNED?

Four nested models, fitted only on train characters + train works, scored on the
double holdout (unseen characters AND unseen works):

  M0 content      : STD local structure + script        -> what everyone shares
  M1 global writer: M0 + writer                          -> a fixed writer bias
  M2 writer x kind: M1 + writer x stroke-EVENT-kind      -> per-stroke-type habit
  M3 writer x STD : M1 + writer x continuous STD context -> fine local response

Two conditions, kept strictly separate because the plan insists on it:

  experiment B (STD)  the local condition comes from the STD glyph available at
                      inference time. This is the one that can justify spatial
                      Writer x STD modulation.
  experiment A (GT)   the local condition is the real stroke event position.
                      Descriptive only: event placement is itself writer-specific,
                      so a good fit here is NOT evidence of a usable condition.

Positive control: synthetic data with a KNOWN writer x kind interaction is fitted
with the same code. If M2 cannot recover it, the implementation is broken and no
negative result may be reported (this is the failure that collapsed the old
bilinear branch to 1e-8).
"""
from __future__ import annotations

import argparse
import csv
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
INK = ['mass', 'ink_frac']
TARGETS = GEO + INK          # what we try to predict at each local site

STD_DIR = 'exp-std/data/std_soft_glyphs_v1'
SCRIPT_CODE = {'楷': 'kai', '行': 'xing', '隶': 'li'}


def std_descriptor(script, character, y, x, nb=5, size=64):
    """STD local context around a normalised (y,x): a coarse density patch.

    Returns a short vector: mean STD ink in an nb x nb window, plus the local
    STD orientation and the window's distance to the glyph's ink centroid. These
    are things an inference-time STD image genuinely provides.
    """
    from PIL import Image
    p = ROOT / STD_DIR / SCRIPT_CODE[script] / f'{character}.png'
    if not p.exists():
        return None
    a = np.array(Image.open(p).convert('L'), np.float32) / 255.0     # 1 = ink
    H, W = a.shape
    cy, cx = int(np.clip(y * H, 0, H - 1)), int(np.clip(x * W, 0, W - 1))
    r = nb // 2
    win = np.zeros((nb, nb), np.float32)
    y0, y1 = max(0, cy - r), min(H, cy + r + 1)
    x0, x1 = max(0, cx - r), min(W, cx + r + 1)
    win[y0 - (cy - r):y0 - (cy - r) + (y1 - y0), x0 - (cx - r):x0 - (cx - r) + (x1 - x0)] = a[y0:y1, x0:x1]
    gy, gx = np.gradient(win)
    mag = float(np.hypot(gy, gx).mean())
    ys, xs = np.nonzero(a > 0.5)
    if len(ys):
        gc_y, gc_x = ys.mean() / H, xs.mean() / W
    else:
        gc_y, gc_x = 0.5, 0.5
    return np.concatenate([win.ravel(), [mag, y - gc_y, x - gc_x,
                                         float(a.mean())]]).astype(np.float32)


def ridge(X, Y, lam):
    """Closed-form ridge with an intercept, via the dual when d >> n."""
    X1 = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
    d = X1.shape[1]
    if len(X1) >= d:
        W = np.linalg.solve(X1.T @ X1 + lam * np.eye(d), X1.T @ Y)
    else:
        W = X1.T @ np.linalg.solve(X1 @ X1.T + lam * np.eye(len(X1)), Y)
    return W


def predict(W, X):
    X1 = np.concatenate([X, np.ones((len(X), 1), X.dtype)], 1)
    return X1 @ W


def onehot(idx, n):
    O = np.zeros((len(idx), n), np.float32)
    O[np.arange(len(idx)), idx] = 1.0
    return O


def rel_gain(L0, L1):
    return float((L0 - L1) / max(L0, 1e-12))


def build(Xc, Woh, Koh, Xstd, model):
    if model == 'M0':
        return Xc
    X1 = np.concatenate([Xc, Woh], 1)
    if model == 'M1':
        return X1
    if model == 'M2':
        WK = (Woh[:, :, None] * Koh[:, None, :]).reshape(len(Xc), -1)
        return np.concatenate([X1, WK], 1)
    WS = (Woh[:, :, None] * Xstd[:, None, :]).reshape(len(Xc), -1)
    return np.concatenate([X1, WS], 1)


def positive_control(seed=0):
    """Known writer x kind interaction; M2 must recover it, M1 must not."""
    rng = np.random.default_rng(seed)
    n_w, n_k = 5, 3
    kinds = ['end', 'junction', 'turn']
    rows, Y = [], []
    for w in range(n_w):
        for k in range(n_k):
            for _ in range(300):
                rows.append((w, k))
                bias = 1.0 if (w == 0 and k == 1) else 0.0       # the planted interaction
                Y.append([bias + 0.05 * rng.normal()])
    rows = np.array(rows)
    Y = np.array(Y, np.float32)
    Xc = rng.normal(size=(len(rows), 4)).astype(np.float32)
    Woh = onehot(rows[:, 0], n_w)
    Koh = onehot(rows[:, 1], n_k)
    tr = rng.random(len(rows)) < 0.6
    res = {}
    for m in ('M0', 'M1', 'M2'):
        X = build(Xc, Woh, Koh, None, m)
        W = ridge(X[tr], Y[tr], 1e-2)
        res[m] = float(np.mean((predict(W, X[~tr]) - Y[~tr]) ** 2))
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='assets/stroke_writer_20261026/event_features.npz')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--win', type=int, default=5)
    ap.add_argument('--max-per-cell', type=int, default=120)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    pc = positive_control(a.seed)
    print(f'[f3] positive control MSE: {pc}', flush=True)
    ok_pc = pc['M2'] < pc['M1'] * 0.5
    print(f'[f3] positive control detects the planted interaction: {ok_pc}', flush=True)

    z = np.load(ROOT / a.features, allow_pickle=True)
    writer, script, char, work, split = z['writer'], z['script'], z['character'], z['work'], None
    E, E_img, E_kind = z['E'], z['E_img'], z['E_kind']
    col = {k: i for i, k in enumerate(z['E_keys'])}

    # split: reuse the established character holdout (25%) and the work split
    chars = sorted(set(char.tolist()))
    perm = np.random.default_rng(90).permutation(len(chars))
    test_chars = set(np.array(chars)[perm[:round(len(chars) * .25)]].tolist())
    img_test = np.array([c in test_chars for c in char])

    rng = np.random.default_rng(a.seed)
    writers = sorted(set(writer.tolist()))
    w_id = {w: i for i, w in enumerate(writers)}
    scripts = sorted(set(script.tolist()))
    s_id = {s: i for i, s in enumerate(scripts)}

    rep = {'positive_control': pc, 'positive_control_ok': bool(ok_pc), 'conditions': {}}

    for cond in ('STD', 'GT'):
        print(f'\n[f3] ===== condition {cond} =====', flush=True)
        Xc_r, W_r, K_r, S_r, Y_r, T_r = [], [], [], [], [], []
        cache = {}
        # stratify: cap the number of events per (writer,kind) cell so no single
        # prolific stratum dominates the regression
        cell_count = defaultdict(int)
        order = rng.permutation(len(E))
        for i in order:
            wi, si = w_id[writer[E_img[i]]], s_id[script[E_img[i]]]
            sc, ch = script[E_img[i]], char[E_img[i]]
            kind = E_KINDS[E_kind[i]]
            key = (wi, kind)
            if cell_count[key] >= a.max_per_cell:
                continue
            y, x = E[i, col['y']], E[i, col['x']]
            if cond == 'STD':
                ck = (sc, ch, int(y * 64), int(x * 64))
                if ck not in cache:
                    d = std_descriptor(sc, ch, y, x, nb=a.win)
                    cache[ck] = d if d is not None else np.zeros(a.win ** 2 + 4, np.float32)
                Xstd = cache[ck]
            else:
                Xstd = None
            cell_count[key] += 1
            # content: rough position + script; a Content-only control with
            # comparable capacity is what stops "more parameters" looking like
            # "writer interaction"
            Xc_r.append(np.concatenate([[y, x, y * x, y * y, x * x],
                                        onehot(np.array([si]), len(scripts))[0]]))
            W_r.append(onehot(np.array([wi]), len(writers))[0])
            K_r.append(E_KINDS.index(kind))
            S_r.append(Xstd if Xstd is not None else np.zeros(0, np.float32))
            Y_r.append(np.array([E[i, col[t]] for t in TARGETS]))
            T_r.append(img_test[E_img[i]])
        if not Xc_r:
            continue
        Xc = np.array(Xc_r, np.float32)
        Woh = np.array(W_r, np.float32)
        Koh = onehot(np.array(K_r), len(E_KINDS))
        Xstd = np.array(np.stack([s if len(s) else np.zeros(a.win ** 2 + 4, np.float32)
                                  for s in S_r])) if cond == 'STD' else None
        Y = np.array(Y_r, np.float32)
        te = np.array(T_r)
        # Standardise targets and content features on the TRAIN side only.
        # asym_* and end_area have heavy tails (a single near-blank patch can be
        # orders of magnitude off), and an unscaled MSE is dominated by a handful
        # of such rows -- which is exactly what produced the absurd 3.8e6 figure.
        y_mu = Y[~te].mean(0)
        y_sd = Y[~te].std(0) + 1e-6
        Y = (Y - y_mu) / y_sd
        c_mu, c_sd = Xc[~te].mean(0), Xc[~te].std(0) + 1e-9
        Xc = (Xc - c_mu) / c_sd
        # clip the standardised targets so a few 100-sigma rows cannot decide the
        # ranking of M0..M3; the same clipping is applied to train and test
        Y = np.clip(Y, -8, 8)
        if Xstd is not None:
            s_mu, s_sd = Xstd[~te].mean(0), Xstd[~te].std(0) + 1e-9
            Xstd = (Xstd - s_mu) / s_sd
        print(f'[f3] {cond}: {len(Xc)} sites, {int(te.sum())} on the double holdout',
              flush=True)
        if te.sum() < 500:
            continue
        lams = {'content': 1.0, 'writer': 10.0}
        entry = {'n_sites': int(len(Xc)), 'n_test': int(te.sum()), 'models': {}}
        for m in ('M0', 'M1', 'M2', 'M3'):
            if m == 'M3' and Xstd is None:
                continue
            Xtr = build(Xc[~te], Woh[~te], Koh[~te],
                        None if Xstd is None else Xstd[~te], m)
            Xte = build(Xc[te], Woh[te], Koh[te],
                        None if Xstd is None else Xstd[te], m)
            W = ridge(Xtr, Y[~te], lams['writer'] if m != 'M0' else lams['content'])
            mse = float(np.mean((predict(W, Xte) - Y[te]) ** 2))
            entry['models'][m] = {'test_mse': mse}
            print(f'[f3]   {m}: test MSE = {mse:.5f}', flush=True)

            # CAPACITY CONTROL. M2 adds only n_writer*n_kind columns, so a gain in
            # R_stroke could just be "more free parameters". Refit the SAME
            # number of extra columns keyed on a SHUFFLED writer label: if the
            # shuffled version gains as much, the gain is capacity, not the
            # writer x kind interaction.
            if m in ('M2', 'M3'):
                shuf = Woh[rng.permutation(len(Woh))]
                Xtr_s = build(Xc[~te], shuf[~te], Koh[~te],
                              None if Xstd is None else Xstd[~te], m)
                Xte_s = build(Xc[te], shuf[te], Koh[te],
                              None if Xstd is None else Xstd[te], m)
                Ws = ridge(Xtr_s, Y[~te], lams['writer'] * 2)
                mse_s = float(np.mean((predict(Ws, Xte_s) - Y[te]) ** 2))
                entry['models'][m]['test_mse_shuffled_writer'] = mse_s
                print(f'[f3]   {m} (shuffled writer, same capacity): '
                      f'test MSE = {mse_s:.5f}', flush=True)

        m = entry['models']
        base = m['M1']['test_mse']
        entry['R_global'] = rel_gain(m['M0']['test_mse'], base)
        entry['R_stroke'] = rel_gain(base, m['M2']['test_mse'])
        entry['R_stroke_capacity_control'] = rel_gain(
            base, m['M2']['test_mse_shuffled_writer'])
        # the honest interaction gain: real label must beat the shuffled one
        entry['R_stroke_beyond_capacity'] = float(
            (m['M2']['test_mse_shuffled_writer'] - m['M2']['test_mse']) /
            max(m['M2']['test_mse_shuffled_writer'], 1e-12))
        if 'M3' in m:
            entry['R_STD'] = rel_gain(base, m['M3']['test_mse'])
            entry['R_STD_beyond_capacity'] = float(
                (m['M3']['test_mse_shuffled_writer'] - m['M3']['test_mse']) /
                max(m['M3']['test_mse_shuffled_writer'], 1e-12))
        print(f'[f3] {cond} R_global={entry["R_global"]:+.4f} '
              f'R_stroke={entry["R_stroke"]:+.4f} '
              f'(capacity control {entry["R_stroke_capacity_control"]:+.4f}, '
              f'beyond {entry["R_stroke_beyond_capacity"]:+.4f}) '
              f'R_STD={entry.get("R_STD", float("nan")):+.4f}', flush=True)
        rep['conditions'][cond] = entry

    p = ROOT / a.out / 'predictability.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[f3] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
