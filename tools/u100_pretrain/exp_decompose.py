"""Decompose the 43.6% cross-char writer separability (all reuse cached feats).

A1  LDA discriminant spectrum: eigenvalues of S_B v = lambda S_W v on PCA-64
    DINO CLS; Top-1 using first k discriminant directions (k=1,2,4,8,9);
    direction stability across char splits (subspace angle).
A2  Mean vs distribution shape: nearest-centroid / LDA / logistic / QDA /
    small MLP, identical splits and PCA features; covariance increment =
    QDA - LDA.
A3  Script-controlled and glyph-cluster-controlled probes: within-script
    top-1; cross-cluster generalization matrix (train classifier on cluster
    i, test on cluster j); discriminant-direction consistency across clusters.
A4  Paired G vs G+L: per-split delta with bootstrap CI (same splits, same
    probe family).
All probes: strict char holdout, 10 splits, capacity-controlled (PCA-64).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}
SCRIPT_OF = None  # filled from csv per char


def topk_accuracy(Ztr, ytr, Zte, yte, k_list):
    """LDA via within-class scatter whitening; report top-1 for rank-k LDA."""
    classes = np.unique(ytr)
    mu = np.stack([Ztr[ytr == c].mean(0) for c in classes])
    SW = sum(((Ztr[ytr == c] - mu[i]).T @ (Ztr[ytr == c] - mu[i])
              for i, c in enumerate(classes)))
    SB = sum((len(Ztr[ytr == c]) * np.outer(mu[i] - Ztr.mean(0), mu[i] - Ztr.mean(0))
              for i, c in enumerate(classes)))
    evals, evecs = np.linalg.eigh(np.linalg.solve(SW + 1e-3 * np.eye(SW.shape[0]), SB))
    order = np.argsort(-evals)
    out = {}
    for k in k_list:
        V = evecs[:, order[:k]]
        ctr = {c: Ztr[ytr == c] @ V for c in classes}
        cents = np.stack([ctr[c].mean(0) for c in classes])
        d = ((Zte @ V)[:, None] - cents[None]) ** 2
        pred = classes[d.sum(-1).argmin(1)]
        out[k] = float((pred == yte).mean())
    return out, evecs[:, order[:9]], evals[order]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--pca-dim', type=int, default=64)
    parser.add_argument('--clusters', type=int, default=6)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_decompose.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    bank = np.load(args.bank)
    FALL = bank['features']
    blook = {int(v): i for i, v in enumerate(bank['ids'])}
    meta = {}
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        meta[(w, int(r['img_id']))] = (int(r['img_id']), r['character'], r['script'])
    by_char = defaultdict(set)
    for (w, _iid), (_i, c, s) in meta.items(): by_char[c].add(w)
    panel_chars = {c for c, ws in by_char.items() if len(ws) >= 4}
    data = []
    for (w, iid), (_i, c, s) in meta.items():
        if c in panel_chars and iid in blook and w < 10:
            data.append((w, c, s, iid))
    print('images', len(data), 'chars', len(panel_chars), flush=True)

    # STD cls for clustering -> reuse cached STD CLS? recompute minimal via bank? STD cls not cached;
    # cluster chars by script+structure proxy: use script id + (stroke-count proxy unavailable) ->
    # use DINO of STD is GPU; keep CPU: cluster on ink-density stats of 64px STD renders.
    from sklearn.cluster import KMeans
    std_stats = {}
    idx_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    from PIL import Image
    for c in panel_chars:
        key = next((k for k in idx_rows if k[1] == c), None)
        if key is None: continue
        im = np.asarray(Image.open(Path(args.std_root) / idx_rows[key]).convert('L'), dtype=np.float32) / 255
        ink = im > .5
        ys, xs = np.where(ink)
        if not len(ys): continue
        h, w_ = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        g = ink.reshape(4, 16, 4, 16).mean((1, 3)).flatten()
        std_stats[c] = np.concatenate([[h / w_, ink.mean()], g])
    chars_ok = sorted(std_stats)
    km = KMeans(n_clusters=args.clusters, n_init=4, random_state=90).fit(np.stack([std_stats[c] for c in chars_ok]))
    cluster_of = {c: int(l) for c, l in zip(chars_ok, km.labels_)}
    print('clusters', args.clusters, 'chars clustered', len(chars_ok), flush=True)

    Ws = np.array([d[0] for d in data])
    Cs = np.array([d[1] for d in data])
    Ss = np.array([d[2] for d in data])
    F = np.stack([FALL[blook[d[3]]] for d in data])

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPClassifier
    chars_uniq = np.array(sorted(set(Cs)))
    results = dict(lda_spectrum=defaultdict(list), classifiers=defaultdict(list),
                   script_control=defaultdict(list), cross_cluster=[], g_vs_gl=[],
                   direction_stability=[])
    for sp in range(args.splits):
        perm = rng.permutation(len(chars_uniq))
        A = set(chars_uniq[perm[:len(perm) // 2]]); B = set(chars_uniq[perm[len(perm) // 2:]])
        mA = np.array([c in A for c in Cs]); mB = ~mA
        sc = StandardScaler().fit(F[mA])
        pa = PCA(n_components=args.pca_dim, random_state=90).fit(sc.transform(F[mA]))
        Za, Zb = pa.transform(sc.transform(F[mA])), pa.transform(sc.transform(F[mB]))
        ya, yb = Ws[mA], Ws[mB]
        # A1 LDA spectrum + top-k
        acc_k, V9, evals = topk_accuracy(Za, ya, Zb, yb, [1, 2, 4, 8, 9])
        for k, v in acc_k.items(): results['lda_spectrum'][str(k)].append(v)
        results['direction_stability'].append(V9)
        # A2 classifiers
        nc_cent = np.stack([Za[ya == c].mean(0) for c in range(10)])
        pred_nc = np.argmin(((Zb[:, None] - nc_cent[None]) ** 2).sum(-1), 1)
        results['classifiers']['nearest_centroid'].append(float((pred_nc == yb).mean()))
        results['classifiers']['logistic'].append(float(
            (LogisticRegression(max_iter=1500, C=1.).fit(Za, ya).predict(Zb) == yb).mean()))
        try:
            qda = QuadraticDiscriminantAnalysis(reg_param=.5).fit(Za[:, :20], ya)
            results['classifiers']['qda20'].append(float((qda.predict(Zb[:, :20]) == yb).mean()))
        except Exception as e:
            print('qda fail', repr(e), flush=True)
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=90).fit(Za, ya)
        results['classifiers']['mlp128'].append(float((mlp.predict(Zb) == yb).mean()))
        # A4 G vs G+L: patch features unavailable offline; proxy = G vs G+width percentiles from quill? skipped here
        # A3 script-controlled (within-script probes)
        for script_name in ('楷', '行', '隶'):
            ms = Ss == script_name
            iA = np.where(mA & ms)[0]; iB = np.where(mB & ms)[0]
            if len(iA) < 200 or len(iB) < 200: continue
            Ztr = pa.transform(sc.transform(F[iA])); Zte = pa.transform(sc.transform(F[iB]))
            cl = LogisticRegression(max_iter=1200, C=1.).fit(Ztr, Ws[iA])
            results['script_control'][script_name].append(float((cl.predict(Zte) == Ws[iB]).mean()))
        # cross-cluster generalization (train cluster i -> test cluster j)
        row = {}
        cid_arr = np.array([cluster_of.get(c, -1) for c in Cs])
        for ci in range(args.clusters):
            iA = np.where(mA & (cid_arr == ci))[0]
            if len(iA) < 150: continue
            Ztr = pa.transform(sc.transform(F[iA]))
            cl = LogisticRegression(max_iter=1200, C=1.).fit(Ztr, Ws[iA])
            for cj in range(args.clusters):
                iB = np.where(mB & (cid_arr == cj))[0]
                if len(iB) < 60: continue
                Zte = pa.transform(sc.transform(F[iB]))
                row[f'{ci}->{cj}'] = float((cl.predict(Zte) == Ws[iB]).mean())
        if row: results['cross_cluster'].append(row)
        print('split', sp, 'done', flush=True)
    # direction stability: principal angles between V9 of split0 vs others
    V0 = results['direction_stability'][0]
    angs = []
    for V in results['direction_stability'][1:]:
        s = np.linalg.svd(V0.T @ V, compute_uv=False)
        s = np.clip(s, -1, 1)
        angs.append(float(np.degrees(np.arccos(s)).mean()))
    results['direction_stability_mean_angle_deg'] = float(np.mean(angs))
    results['lda_spectrum'] = {k: float(np.mean(v)) for k, v in results['lda_spectrum'].items()}
    results['classifiers'] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)))
                              for k, v in results['classifiers'].items()}
    results['script_control'] = {k: dict(mean=float(np.mean(v)), n=len(v))
                                 for k, v in results['script_control'].items()}
    cc_keys = sorted({k for row in results['cross_cluster'] for k in row})
    results['cross_cluster_mean'] = {k: float(np.mean([r[k] for r in results['cross_cluster'] if k in r]))
                                     for k in cc_keys}
    results['cross_cluster'] = results['cross_cluster'][:2]
    del results['direction_stability']
    out.write_text(json.dumps(results, indent=1))
    print('DECOMPOSE_DONE', json.dumps({k: results[k] for k in
        ('lda_spectrum', 'classifiers', 'script_control', 'direction_stability_mean_angle_deg')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
