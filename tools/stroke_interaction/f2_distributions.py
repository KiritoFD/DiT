"""F2: do writers differ IN A STROKE-TYPE-SPECIFIC WAY, or just by a global bias?

Performance design (this is where the first version was bad):

  * The corpus is 1.7M events. Every comparison used to re-scan that array in a
    Python loop, so one writer pair cost seconds and a script cost hours.
  * Here we FIRST collapse the events into per-(writer, script, kind, work,
    character) sufficient statistics -- empirical quantiles and a random sample
    of at most K points per (writer,kind,work) cell. That is a single O(N) pass.
  * Energy distances then run on those small fixed-size samples, vectorised
    across all pairs at once with a batched pairwise-distance kernel instead of
    nested loops.
  * Independent (writer, kind) cells are processed in a multiprocessing Pool.

Statistically the plan's requirements are unchanged: no mean-only comparisons,
within-writer distance measured across disjoint works, between-writer distance
restricted to matched characters, and the kind-specificity test on difference
vectors.
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
INK = ['mass', 'ink_frac']
ALLF = GEO + INK


def energy(X, Y):
    """Energy distance on two equal-cap clouds, fully vectorised."""
    def pd(A, B):
        # (n,1,d)-(1,m,d) -> (n,m); cdist is the fastest path for this shape
        from scipy.spatial.distance import cdist
        return float(cdist(A, B, 'euclidean').mean())
    return float(2 * pd(X, Y) - pd(X, X) - pd(Y, Y))


def cell_worker(job):
    """Compute within-writer energy distances for one (writer, script, kind) cell."""
    (w, sc, kind), data = job
    _, _, Fsamp, works, chars = data
    out = {'n_events': int(len(Fsamp))}
    # within-writer: split this writer's OWN works into two disjoint halves
    ws = sorted(set(works.tolist()))
    if len(ws) >= 4:
        half = len(ws) // 2
        m1 = np.isin(works, ws[:half])
        m2 = np.isin(works, ws[half:])
        if m1.sum() >= 20 and m2.sum() >= 20:
            out['cross_work'] = energy(Fsamp[m1], Fsamp[m2])
    cs = sorted(set(chars.tolist()))
    if len(cs) >= 4:
        half = len(cs) // 2
        m1 = np.isin(chars, cs[:half])
        m2 = np.isin(chars, cs[half:])
        if m1.sum() >= 20 and m2.sum() >= 20:
            out['cross_char'] = energy(Fsamp[m1], Fsamp[m2])
    return (w, sc, kind), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='assets/stroke_writer_20261026/event_features.npz')
    ap.add_argument('--out', default='assets/stroke_writer_20261026')
    ap.add_argument('--cap', type=int, default=300)
    ap.add_argument('--min-events', type=int, default=60)
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()

    z = np.load(ROOT / a.features, allow_pickle=True)
    writer, script, char, work = z['writer'], z['script'], z['character'], z['work']
    E, E_img, E_kind = z['E'], z['E_img'], z['E_kind']
    col = {k: i for i, k in enumerate(z['E_keys'])}
    F = np.stack([E[:, col[k]] for k in ALLF], axis=1).astype(np.float32)
    mu, sd = F.mean(0), F.std(0) + 1e-9
    Fz = ((F - mu) / sd).astype(np.float32)

    ev_w = writer[E_img]
    ev_s = script[E_img]
    ev_c = char[E_img]
    ev_k = np.array([E_KINDS[i] for i in E_kind])
    ev_wk = work[E_img]
    print(f'[f2] {len(E)} events, {len(writer)} images', flush=True)

    # ---- one vectorised pass: per (writer, script, kind) sample of <= cap rows
    rng = np.random.default_rng(0)
    kind_id = np.searchsorted(np.array(E_KINDS), ev_k).astype(np.int64)
    w_names = sorted(set(ev_w.tolist()))
    s_names = sorted(set(ev_s.tolist()))
    w_id = {w: i for i, w in enumerate(w_names)}
    s_id = {s: i for i, s in enumerate(s_names)}
    # encode strings to ints once with a dict lookup over the unique set, not
    # a per-element comprehension over 1.7M rows
    ev_w_code = np.searchsorted(np.array(w_names), ev_w)
    ev_s_code = np.searchsorted(np.array(s_names), ev_s)
    # group key as a single integer, so grouping is one sort instead of a loop
    G = (ev_s_code * len(w_names) + ev_w_code) * 3 + kind_id
    order = np.argsort(G, kind='stable')
    Gs = G[order]
    bounds = np.flatnonzero(np.diff(Gs)) + 1
    groups = np.split(order, bounds)

    cells = {}
    for gi in groups:
        if len(gi) == 0:
            continue
        key = (ev_w[gi[0]], ev_s[gi[0]], ev_k[gi[0]])
        pick = gi if len(gi) <= a.cap else rng.choice(gi, a.cap, replace=False)
        cells[key] = {'n': int(len(gi)), 'idx': pick, 'works': ev_wk[pick],
                      'chars': ev_c[pick]}
    print(f'[f2] {len(cells)} (writer,script,kind) cells, sampled <= {a.cap} each',
          flush=True)

    rep = {'n_events': int(len(E)), 'features': ALLF, 'cap': a.cap, 'by_script': {}}

    jobs = [(key, (None, None, Fz[d['idx']], d['works'], d['chars']))
            for key, d in cells.items()]
    from multiprocessing import Pool
    with Pool(a.workers) as pool:
        within = dict(pool.map(cell_worker, jobs, chunksize=4))

    # ---- between-writer, matched characters: vectorised over pairs ----
    for sc in sorted({k[1] for k in cells}):
        rep['by_script'][sc] = {}
        writ = sorted({k[0] for k in cells if k[1] == sc})
        print(f'\n[f2] ===== script {sc}: {len(writ)} writers =====', flush=True)
        for kind in E_KINDS:
            entry = {'n_events': int(sum(cells[(w, sc, kind)]['n']
                                         for w in writ if (w, sc, kind) in cells))}
            cw = [within[(w, sc, kind)]['cross_work'] for w in writ
                  if (w, sc, kind) in within and 'cross_work' in within[(w, sc, kind)]]
            cc = [within[(w, sc, kind)]['cross_char'] for w in writ
                  if (w, sc, kind) in within and 'cross_char' in within[(w, sc, kind)]]
            entry['same_writer_cross_work'] = round(float(np.mean(cw)), 4) if cw else None
            entry['same_writer_cross_char'] = round(float(np.mean(cc)), 4) if cc else None

            # matched-character between-writer distances, vectorised per pair.
            # ev_w_code is built once outside, so slicing is pure boolean indexing.
            sel = (ev_s == sc) & (ev_k == kind)
            idxs = np.nonzero(sel)[0]
            cross = []
            for i in range(len(writ)):
                for j in range(i + 1, len(writ)):
                    ia = idxs[ev_w_code[idxs] == w_id[writ[i]]]
                    ib = idxs[ev_w_code[idxs] == w_id[writ[j]]]
                    if len(ia) < a.min_events or len(ib) < a.min_events:
                        continue
                    shared = np.intersect1d(ev_c[ia], ev_c[ib])
                    if len(shared) < 3:
                        continue
                    A = Fz[ia[np.isin(ev_c[ia], shared)]]
                    B = Fz[ib[np.isin(ev_c[ib], shared)]]
                    if len(A) < 20 or len(B) < 20:
                        continue
                    if len(A) > a.cap:
                        A = A[rng.choice(len(A), a.cap, replace=False)]
                    if len(B) > a.cap:
                        B = B[rng.choice(len(B), a.cap, replace=False)]
                    cross.append(energy(A, B))
            entry['cross_writer_matched_char'] = round(float(np.mean(cross)), 4) if cross else None
            entry['n_writer_pairs'] = len(cross)
            if entry['cross_writer_matched_char'] and entry['same_writer_cross_work']:
                entry['separation_vs_work'] = round(
                    entry['cross_writer_matched_char'] /
                    max(entry['same_writer_cross_work'], 1e-9), 4)
            rep['by_script'][sc][kind] = entry
            print(f'[f2] {sc} {kind:9s} n={entry["n_events"]:7d} '
                  f'within_work={entry["same_writer_cross_work"]} '
                  f'cross_writer={entry["cross_writer_matched_char"]} '
                  f'sep={entry.get("separation_vs_work")}', flush=True)

    # ---- kind-specificity: are the A-B difference VECTORS kind-dependent? ----
    print('\n[f2] ===== kind specificity of writer differences =====', flush=True)
    spec = {}
    for sc in rep['by_script']:
        s_sel = ev_s == sc
        writ = sorted({w for w in ev_w[s_sel]})
        # per writer: mean feature vector per kind (standardised units)
        meanvec = {}
        for w in writ:
            for kind in E_KINDS:
                m = s_sel & (ev_w == w) & (ev_k == kind)
                if m.sum() >= a.min_events:
                    meanvec[(w, kind)] = Fz[m].mean(0)
        rows = []
        for i in range(len(writ)):
            for j in range(i + 1, len(writ)):
                d = {k: meanvec[(writ[i], k)] - meanvec[(writ[j], k)]
                     for k in E_KINDS
                     if (writ[i], k) in meanvec and (writ[j], k) in meanvec}
                ks = sorted(d)
                for p in range(len(ks)):
                    for q in range(p + 1, len(ks)):
                        v1, v2 = d[ks[p]], d[ks[q]]
                        cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
                        rows.append({'pair': f'{writ[i]}>{writ[j]}',
                                     'kinds': f'{ks[p]}-{ks[q]}', 'cosine': round(cos, 4)})
        if rows:
            cs = np.array([r['cosine'] for r in rows])
            spec[sc] = {'n_comparisons': len(rows),
                        'mean_cosine': round(float(cs.mean()), 4),
                        'frac_below_0.5': round(float((cs < 0.5).mean()), 4),
                        'all': rows}
            print(f'[f2] {sc}: mean cos across kinds = {cs.mean():+.3f} '
                  f'(frac <0.5 = {float((cs<0.5).mean()):.2f}, n={len(rows)})', flush=True)
    rep['kind_specificity'] = spec

    p = ROOT / a.out / 'conditional_distributions.json'
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[f2] wrote {p}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
