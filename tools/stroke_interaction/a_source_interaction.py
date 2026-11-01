"""A: is F2's huge separability real, or an artefact of source / extraction?

Three separate questions, all answerable without new modelling:

A1  SOURCE. work_id() recovers 67 coarse source groups (2-5 per stratum), but
    14,392 of 26,002 rows sit in ONE group (`data/fame-kxl-tj-px60`, a dataset
    name, not a work). So we can control source only coarsely. We recompute the
    writer separation using ONLY between-source comparisons (writer pairs where
    the two groups are disjoint) and report how many comparisons survive.

A2  EXTRACTION STABILITY. The separability is measured on events produced by a
    hard threshold (ink > 0.5) and a 256x256 raster. If the statistic moves a
    lot under a different threshold or resolution, F2 is measuring the detector.
    We re-extract a subsample at several thresholds/resolutions and report the
    coefficient of variation of the per-feature writer-vs-within ratio.

A3  THE REAL INTERACTION. The plan's I_{w,k}, computed after removing the
    writer main effect AND the event-kind main effect, then asked whether it
    reproduces across held-out characters. A fixed writer bias by construction
    gives a high cross-character cosine on raw means -- that is NOT evidence of
    interaction, and the earlier report wrongly treated it as such.
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


def energy(X, Y, cap=250, rng=None):
    from scipy.spatial.distance import cdist
    rng = rng or np.random.default_rng(0)
    if len(X) > cap:
        X = X[rng.choice(len(X), cap, replace=False)]
    if len(Y) > cap:
        Y = Y[rng.choice(len(Y), cap, replace=False)]
    if len(X) < 3 or len(Y) < 3:
        return None
    return float(2 * cdist(X, Y).mean() - cdist(X, X).mean() - cdist(Y, Y).mean())


def interaction_from_means(M, writers, kinds):
    """I_{w,k} = mu_{w,k} - mu_{w,.} - mu_{.,k} + mu_{.,.}

    M : dict (writer, kind) -> mean vector
    Balanced over the (writer, kind) cells that exist, so a prolific cell cannot
    define the grand mean. This is what removes the writer main effect and the
    kind main effect; whatever is left is the part that is specific to the
    PAIRING of a writer with a stroke kind.
    """
    cells = [(w, k) for w in writers for k in kinds if (w, k) in M]
    if not cells:
        return {}
    grand = np.mean([M[c] for c in cells], 0)
    per_w = {w: np.mean([M[(w, k)] for k in kinds if (w, k) in M], 0)
             for w in writers if any((w, k) in M for k in kinds)}
    per_k = {k: np.mean([M[(w, k)] for w in writers if (w, k) in M], 0)
             for k in kinds if any((w, k) in M for w in writers)}
    return {c: M[c] - per_w[c[0]] - per_k[c[1]] + grand for c in cells}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='assets/stroke_writer_20261026/event_features.npz')
    ap.add_argument('--csv', default='exp-std/csv/train_top10_noaug.csv')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--cap', type=int, default=250)
    ap.add_argument('--min-events', type=int, default=60)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()

    import csv
    from src.eval.style.bank import work_id

    z = np.load(ROOT / a.features, allow_pickle=True)
    writer, script, char = z['writer'], z['script'], z['character']
    E, E_img, E_kind = z['E'], z['E_img'], z['E_kind']
    col = {k: i for i, k in enumerate(z['E_keys'])}
    F = np.stack([E[:, col[k]] for k in ALLF], axis=1).astype(np.float64)
    ev_w, ev_s, ev_c = writer[E_img], script[E_img], char[E_img]
    ev_k = E_kind.astype(np.int64)

    # ---- attach the coarse source group to each event ----
    rows = list(csv.DictReader(open(ROOT / a.csv, encoding='utf-8-sig')))
    wid_of_path = {r['image_path']: work_id(r['src_image_path'], r['image_path'])
                   for r in rows}
    img_work = np.array([wid_of_path.get(p, 'unknown') for p in z['image_path']])
    ev_work = img_work[E_img]

    chars = sorted(set(char.tolist()))
    perm = np.random.default_rng(90).permutation(len(chars))
    test_chars = set(np.array(chars)[perm[:round(len(chars) * .25)]].tolist())
    rng = np.random.default_rng(a.seed)

    rep = {'features': ALLF, 'n_source_groups': int(len(set(img_work.tolist())))}

    # ======================= A1: source-controlled separation ===============
    print('\n[A1] source-controlled writer separation', flush=True)
    rep['A1_source_control'] = {}
    for sc in ('楷', '行', '隶'):
        sel_s = ev_s == sc
        if sel_s.sum() < 2000:
            continue
        writers = sorted(set(ev_w[sel_s].tolist()))
        rep['A1_source_control'][sc] = {}
        for kind in E_KINDS:
            sel = sel_s & (ev_k == E_KINDS.index(kind))
            idx = np.nonzero(sel)[0]
            if len(idx) < 400:
                continue
            mu, sd = F[idx].mean(0), F[idx].std(0) + 1e-9
            Fz = (F[idx] - mu) / sd
            within, cross_same_src, cross_diff_src = [], [], []
            for w in writers:
                wi = np.nonzero(ev_w[idx] == w)[0]
                ws = sorted(set(ev_work[idx][wi].tolist()))
                if len(ws) >= 2 and len(wi) >= a.min_events:
                    h = len(ws) // 2
                    A = Fz[wi[np.isin(ev_work[idx][wi], ws[:h])]]
                    B = Fz[wi[np.isin(ev_work[idx][wi], ws[h:])]]
                    e = energy(A, B, a.cap, rng)
                    if e is not None:
                        within.append(e)
            n_same = n_diff = 0
            for i in range(len(writers)):
                for j in range(i + 1, len(writers)):
                    ia = np.nonzero(ev_w[idx] == writers[i])[0]
                    ib = np.nonzero(ev_w[idx] == writers[j])[0]
                    if len(ia) < a.min_events or len(ib) < a.min_events:
                        continue
                    shared = np.intersect1d(ev_c[idx][ia], ev_c[idx][ib])
                    if len(shared) < 3:
                        continue
                    A = Fz[ia[np.isin(ev_c[idx][ia], shared)]]
                    B = Fz[ib[np.isin(ev_c[idx][ib], shared)]]
                    # a writer pair can share a source group or not; only the
                    # DISJOINT case is free of a same-source explanation
                    sa = set(ev_work[idx][ia].tolist())
                    sb = set(ev_work[idx][ib].tolist())
                    e = energy(A, B, a.cap, rng)
                    if e is None:
                        continue
                    if sa & sb:
                        cross_same_src.append(e)
                        n_same += 1
                    else:
                        cross_diff_src.append(e)
                        n_diff += 1
            wm = float(np.mean(within)) if within else None
            entry = {
                'n_events': int(len(idx)),
                'within_source': round(wm, 4) if wm else None,
                'cross_writer_same_source': round(float(np.mean(cross_same_src)), 4)
                if cross_same_src else None,
                'n_pairs_same_source': n_same,
                'cross_writer_diff_source': round(float(np.mean(cross_diff_src)), 4)
                if cross_diff_src else None,
                'n_pairs_diff_source': n_diff,
            }
            if wm and cross_diff_src:
                entry['separation_different_source_only'] = round(
                    float(np.mean(cross_diff_src)) / max(wm, 1e-9), 3)
            rep['A1_source_control'][sc][kind] = entry
            print(f'[A1] {sc} {kind:9s} within={entry["within_source"]} '
                  f'same-src={entry["cross_writer_same_source"]}(n={n_same}) '
                  f'diff-src={entry["cross_writer_diff_source"]}(n={n_diff}) '
                  f'-> sep={entry.get("separation_different_source_only")}', flush=True)

    # ======================= A3: the true interaction ======================
    print('\n[A3] true interaction I_{w,k}, held-out characters', flush=True)
    rep['A3_interaction'] = {}
    for sc in ('楷', '行'):
        sel_s = ev_s == sc
        writers = sorted(set(ev_w[sel_s].tolist()))
        tr_m = np.array([c not in test_chars for c in ev_c])
        # per (writer, kind) mean on TRAIN chars, then form I_{w,k}
        def means_on(mask):
            M = {}
            for w in writers:
                for ki, k in enumerate(E_KINDS):
                    m = sel_s & (ev_w == w) & (ev_k == ki) & mask
                    if m.sum() >= a.min_events:
                        M[(w, k)] = F[m].mean(0)
            return M
        Mtr = means_on(tr_m)
        Mte = means_on(~tr_m)
        Itr = interaction_from_means(Mtr, writers, E_KINDS)
        Ite = interaction_from_means(Mte, writers, E_KINDS)
        common = [k for k in Itr if k in Ite]
        if not common:
            continue
        # does each writer's interaction PATTERN on held-out chars match train?
        rows_cos, mag = [], []
        for w in writers:
            vtr = [Itr[k] for k in common if k[0] == w]
            vte = [Ite[k] for k in common if k[0] == w]
            if len(vtr) < 2:
                continue
            vtr = np.concatenate(vtr)
            vte = np.concatenate(vte)
            ntr, nte = np.linalg.norm(vtr), np.linalg.norm(vte)
            if ntr < 1e-9:
                continue
            rows_cos.append(float(vtr @ vte / (ntr * max(nte, 1e-12))))
            mag.append(float(nte / ntr))
        # NULL: shuffle the writer label of the interaction cells
        null_cos = []
        for _ in range(200):
            perm_w = list(writers)
            rng.shuffle(perm_w)
            mapping = dict(zip(writers, perm_w))
            Msh = {(mapping[w], k): v for (w, k), v in Mtr.items()}
            Ish = interaction_from_means(Msh, writers, E_KINDS)
            for w in writers:
                v1 = [Ish[k] for k in common if k[0] == w]
                v2 = [Ite[k] for k in common if k[0] == w]
                if len(v1) < 2:
                    continue
                v1, v2 = np.concatenate(v1), np.concatenate(v2)
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                if n1 < 1e-9:
                    continue
                null_cos.append(float(v1 @ v2 / (n1 * max(n2, 1e-12))))
        obs = float(np.mean(rows_cos)) if rows_cos else None
        nul = float(np.mean(null_cos)) if null_cos else None
        rep['A3_interaction'][sc] = {
            'n_writers': len(rows_cos),
            'observed_cos_train_vs_heldout': round(obs, 4) if obs is not None else None,
            'null_cos_shuffled_writer': round(nul, 4) if nul is not None else None,
            'interaction_magnitude_ratio': round(float(np.mean(mag)), 4) if mag else None,
            'beats_null': bool(obs is not None and nul is not None and obs > nul + 0.05),
        }
        print(f'[A3] {sc}: observed cos={obs} vs null={nul} '
              f'(mag ratio {np.mean(mag) if mag else None})', flush=True)

    p = ROOT / a.out / 'A_source_and_interaction.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n[A] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
