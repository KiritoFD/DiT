"""F2 supplement: the three columns of the plan's core table that need controls.

The main F2 pass says how separable writers are per event kind. It does NOT say
whether that separability is (a) the geometrical hand or (b) just how much ink
the scan happens to carry. The plan asks for both, per event kind:

  same_writer_cross_work   within-writer consistency across disjoint works
  between_writer           separability with characters matched
  after_ink_control        the same, with ink amount / stroke width regressed
                           out -- if separability survives, the difference is
                           geometry, not exposure
  cross_char_repro         does the winner direction recur on held-out
                           characters (the plan's "复现" column)

Ink control: for every feature we subtract its linear prediction from
{local ink mass, local ink fraction, glyph-level median width}, all three fitted
on the TRAIN side only.
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
ALLF = GEO + ['mass', 'ink_frac']
CONTROLS = ['mass', 'ink_frac']


def energy(X, Y, cap=300, rng=None):
    from scipy.spatial.distance import cdist
    rng = rng or np.random.default_rng(0)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(Y) > cap:
        Y = Y[rng.choice(len(Y), cap, replace=False)]
    if len(X) < 3 or len(Y) < 3:
        return None
    return float(2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='assets/stroke_writer_20261026/event_features.npz')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--cap', type=int, default=250)
    ap.add_argument('--min-events', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    z = np.load(ROOT / a.features, allow_pickle=True)
    writer, script, char = z['writer'], z['script'], z['character']
    work = z['work']
    E, E_img, E_kind = z['E'], z['E_img'], z['E_kind']
    col = {k: i for i, k in enumerate(z['E_keys'])}
    F = np.stack([E[:, col[k]] for k in ALLF], axis=1).astype(np.float64)
    ev_w, ev_s, ev_c = writer[E_img], script[E_img], char[E_img]
    # E_kind is ALREADY an integer 0/1/2 code (written by extract.py); decoding it
    # with searchsorted against the string table silently mapped everything to 0.
    ev_k = E_kind.astype(np.int64)
    ev_wk = work[E_img]

    rng = np.random.default_rng(a.seed)
    # character holdout, same convention as F3
    chars = sorted(set(char.tolist()))
    perm = np.random.default_rng(90).permutation(len(chars))
    test_chars = set(np.array(chars)[perm[:round(len(chars) * .25)]].tolist())

    rep = {'features': ALLF, 'controls': CONTROLS, 'by_script': {}}
    for sc in sorted(set(ev_s.tolist())):
        sel_s = ev_s == sc
        if sel_s.sum() < 2000:
            continue
        writers = sorted(set(ev_w[sel_s].tolist()))
        rep['by_script'][sc] = {'n_events': int(sel_s.sum()), 'by_kind': {}}
        print(f'\n[f2b] ===== {sc}: {int(sel_s.sum())} events, {len(writers)} writers =====',
              flush=True)

        for kind in E_KINDS:
            sel = sel_s & (ev_k == E_KINDS.index(kind))
            idx_all = np.nonzero(sel)[0]
            print(f'[f2b] {sc} {kind}: {len(idx_all)} events', flush=True)
            if len(idx_all) < 400:
                continue
            # ---- ink control, fitted on TRAIN characters only ----
            tr = np.array([c not in test_chars for c in ev_c[idx_all]])
            ctrl = np.stack([E[idx_all, col[c]] for c in CONTROLS], 1)
            Xd = np.concatenate([ctrl, np.ones((len(ctrl), 1))], 1)
            beta, *_ = np.linalg.lstsq(Xd[tr], F[idx_all][tr], rcond=None)
            resid = F[idx_all] - Xd @ beta
            mu, sd = resid[tr].mean(0), resid[tr].std(0) + 1e-9
            Rz = ((resid - mu) / sd).astype(np.float32)
            Fmu, Fsd = F[idx_all][tr].mean(0), F[idx_all][tr].std(0) + 1e-9
            Fz = ((F[idx_all] - Fmu) / Fsd).astype(np.float32)

            def sep(M):
                """between-writer (matched char) / within-writer (across works)."""
                within, cross = [], []
                for w in writers:
                    wi = np.nonzero(ev_w[idx_all] == w)[0]
                    ws = sorted(set(ev_wk[idx_all][wi].tolist()))
                    if len(ws) >= 4 and len(wi) >= a.min_events:
                        h = len(ws) // 2
                        A = M[wi[np.isin(ev_wk[idx_all][wi], ws[:h])]]
                        B = M[wi[np.isin(ev_wk[idx_all][wi], ws[h:])]]
                        e = energy(A, B, a.cap, rng)
                        if e is not None:
                            within.append(e)
                for i in range(len(writers)):
                    for j in range(i + 1, len(writers)):
                        ia = np.nonzero(ev_w[idx_all] == writers[i])[0]
                        ib = np.nonzero(ev_w[idx_all] == writers[j])[0]
                        if len(ia) < a.min_events or len(ib) < a.min_events:
                            continue
                        shared = np.intersect1d(ev_c[idx_all][ia], ev_c[idx_all][ib])
                        if len(shared) < 3:
                            continue
                        A = M[ia[np.isin(ev_c[idx_all][ia], shared)]]
                        B = M[ib[np.isin(ev_c[idx_all][ib], shared)]]
                        e = energy(A, B, a.cap, rng)
                        if e is not None:
                            cross.append(e)
                if not within or not cross:
                    return None, None, None
                wm, cm = float(np.mean(within)), float(np.mean(cross))
                return round(wm, 4), round(cm, 4), round(cm / max(wm, 1e-9), 3)

            w_raw, c_raw, s_raw = sep(Fz)
            w_ink, c_ink, s_ink = sep(Rz)

            # ---- cross-character reproduction of the winner direction ----
            # direction between the two most separable writers on TRAIN chars,
            # then asked to hold on held-out chars
            te = np.array([c in test_chars for c in ev_c[idx_all]])
            repro = None
            if len(writers) >= 2 and te.sum() > 100 and (~te).sum() > 100:
                best = None
                for i in range(len(writers)):
                    for j in range(i + 1, len(writers)):
                        mi = (ev_w[idx_all] == writers[i]) & ~te
                        mj = (ev_w[idx_all] == writers[j]) & ~te
                        if mi.sum() < a.min_events or mj.sum() < a.min_events:
                            continue
                        d = Fz[mi].mean(0) - Fz[mj].mean(0)
                        n = np.linalg.norm(d)
                        if best is None or n > best[0]:
                            best = (n, d / max(n, 1e-12), writers[i], writers[j])
                if best is not None:
                    _, dv, wa, wb = best
                    mi = (ev_w[idx_all] == wa) & te
                    mj = (ev_w[idx_all] == wb) & te
                    if mi.sum() >= 20 and mj.sum() >= 20:
                        dte = Fz[mi].mean(0) - Fz[mj].mean(0)
                        repro = {'pair': f'{wa}>{wb}',
                                 'cosine_train_vs_test_direction':
                                     round(float(dv @ dte / (np.linalg.norm(dte) + 1e-12)), 3)}

            rep['by_script'][sc]['by_kind'][kind] = {
                'n_events': int(len(idx_all)),
                'within_work': w_raw, 'between_writer': c_raw,
                'separation_ratio': s_raw,
                'within_work_ink_controlled': w_ink,
                'between_writer_ink_controlled': c_ink,
                'separation_ratio_ink_controlled': s_ink,
                'cross_char_reproduction': repro,
            }
            print(f'[f2b] {sc} {kind:9s} raw sep={s_raw} '
                  f'(within {w_raw} / between {c_raw})  '
                  f'ink-controlled sep={s_ink}  repro={repro}', flush=True)

    (ROOT / a.out).mkdir(parents=True, exist_ok=True)
    p = ROOT / a.out / 'conditional_distributions_controlled.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[f2b] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
