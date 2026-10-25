"""Style structure measurements from cached features only (no DINO, no training).

S1  K-shot prototype aggregation: per writer, prototype from K distinct ref
    chars (one image each); classify test images (chars outside the writer's
    own ref set) by nearest prototype. K=1..32, random refs vs cluster-
    restricted refs; prototype direction stability across disjoint draws;
    K90 = smallest K reaching 90% of its own K=32 accuracy.
S2  Writer x glyph-cluster interaction on cell means:
    y[w,c] ~ alpha_{-w}(c) + b_w [+ gamma_{w,k(c)}], k from STD-stat clusters
    (4/8/16). Held-out error per cell (leave-this-char-out for all terms).
    Reports uncorrected interaction increment.
S3  Discriminant dimension spectrum: SVD of the logistic coefficient matrix,
    Top-1 at rank k=1..9, d_eff95; QDA vs LDA log-loss on the same splits.
Bootstrap CIs are grouped by character.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--draws', type=int, default=10)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_structure.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    bank = np.load(args.bank)
    FALL = bank['features']
    blook = {int(v): i for i, v in enumerate(bank['ids'])}
    imgs = []          # (w, c, feat)
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        iid = int(r['img_id'])
        if iid in blook and w < 10:
            imgs.append((w, ord(r['character']), iid))
    chars_by_w = defaultdict(set)
    for w, c, _ in imgs: chars_by_w[w].add(c)
    by_char = defaultdict(set)
    for w, c, _ in imgs: by_char[c].add(w)
    panel = {c for c, ws in by_char.items() if len(ws) >= 4}
    imgs = [(w, c, FALL[blook[i]]) for (w, c, i) in imgs if c in panel]
    by_wc = defaultdict(list)
    for w, c, f in imgs: by_wc[(w, c)].append(f)
    print('images', len(imgs), flush=True)

    from PIL import Image
    from sklearn.cluster import KMeans
    idx_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    stats = {}
    for c in panel:
        ch = chr(c)
        key = next((k for k in idx_rows if k[1] == ch), None)
        if key is None: continue
        im = np.asarray(Image.open(Path(args.std_root) / idx_rows[key]).convert('L'), dtype=np.float32) / 255
        ink = im > .5
        ys, xs = np.where(ink)
        if not len(ys): continue
        stats[c] = np.concatenate([[ (ys.max()-ys.min()+1) / (xs.max()-xs.min()+1), ink.mean()],
                                   ink.reshape(4, 16, 4, 16).mean((1, 3)).flatten()])
    chars_ok = np.array(sorted(stats))
    X = np.stack([stats[c] for c in chars_ok])
    clusters = {}
    for K in (4, 8, 16):
        km = KMeans(n_clusters=K, n_init=4, random_state=90).fit(X)
        clusters[K] = {c: int(l) for c, l in zip(chars_ok, km.labels_)}
    print('clusters done', flush=True)

    # S1 K-shot
    Ks = [1, 2, 4, 8, 16, 32]
    acc_random = defaultdict(list)
    acc_cluster = {}  # cluster-restricted variant dropped: needs all-writer matched protocol
    stab = defaultdict(list)
    for draw in range(args.draws):
        refs = {}
        for w in range(10):
            cw = sorted(chars_by_w[w] & panel)
            refs[w] = set(rng.choice(cw, min(32, len(cw)), replace=False))
        for K in Ks:
            protos = {}
            for w in range(10):
                chosen = sorted(refs[w])[:K]
                fs = [by_wc[(w, ch)][0] for ch in chosen]
                protos[w] = np.mean(fs, 0)
            P = np.stack([protos[w] for w in range(10)])
            Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
            correct, total, by_char_acc = 0, 0, defaultdict(list)
            for (w, c, f) in imgs:
                if c in refs[w]: continue
                fn = f / (np.linalg.norm(f) + 1e-9)
                pred = int((Pn @ fn).argmax())
                correct += pred == w; total += 1
                by_char_acc[(w, c)].append(pred == w)
            acc_random[K].append(correct / total)
        # prototype stability: two disjoint 8-char draws, cosine
        for K in (4, 8, 16):
            cs = []
            for w in range(10):
                cw = sorted(chars_by_w[w] & panel)
                if len(cw) < 2 * K: continue
                perm = rng.permutation(len(cw))
                A = [by_wc[(w, cw[i])][0] for i in perm[:K]]
                B = [by_wc[(w, cw[i])][0] for i in perm[K:2 * K]]
                pa, pb = np.mean(A, 0), np.mean(B, 0)
                cs.append(float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9)))
            stab[K].append(float(np.mean(cs)))
        print('draw', draw, 'done', flush=True)
    # K90 from the mean curve
    curve = {K: float(np.mean(acc_random[K])) for K in Ks}
    plateau = curve[32]
    k90 = next((K for K in Ks if curve[K] >= 0.9 * plateau), None)

    # S2 interaction on cell means
    cell = defaultdict(list)
    for w, c, f in imgs: cell[(w, c)].append(f)
    y = {k: np.mean(v, 0) for k, v in cell.items()}
    res_int = {}
    alpha = {}
    for (w, c), yy in y.items():
        others = [y[(v, c)] for v in by_char[c] if v != w and (v, c) in y]
        alpha[(w, c)] = np.mean(others, axis=0) if others else np.zeros_like(yy)
    resid = {k: (y[k] - alpha[k]) for k in y}
    for K in (4, 8, 16):
        clook = clusters[K]
        # b_w over all chars (leave-one-out exclusion handled per cell by subtract)
        sums = defaultdict(lambda: [np.zeros(768, dtype=np.float64), 0])
        for (w, c), r in resid.items():
            if c not in clook: continue
            sums[w][0] += r; sums[w][1] += 1
        # gamma_{w,k} sums with per-cell exclusion
        gsum = defaultdict(lambda: [np.zeros(768, dtype=np.float64), 0])
        for (w, c), r in resid.items():
            if c not in clook: continue
            g = gsum[(w, clook[c])]; g[0] += r; g[1] += 1
        sse_base = sse_add = sse_int = n = 0.0
        for (w, c), yy in y.items():
            if c not in clook: continue
            k = clook[c]
            a = alpha[(w, c)]
            sw, cnt = sums[w]
            b_w = (sw - resid[(w, c)]) / max(cnt - 1, 1)
            pred_base = a
            pred_add = a + b_w
            gg = gsum[(w, k)]
            g = (gg[0] - resid[(w, c)]) / max(gg[1] - 1, 1)
            pred_int = pred_add + g
            sse_base += float(((yy - pred_base) ** 2).sum())
            sse_add += float(((yy - pred_add) ** 2).sum())
            sse_int += float(((yy - pred_int) ** 2).sum())
            n += 1
        res_int[K] = dict(n_cells=int(n),
                          delta_add_vs_base=float((sse_base - sse_add) / sse_base),
                          delta_int_vs_add=float((sse_add - sse_int) / sse_add),
                          delta_int_of_global=float((sse_add - sse_int) / max(sse_base - sse_add, 1e-9)))
        print('interaction K', K, res_int[K], flush=True)

    # S3 spectrum + logloss
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    F = np.stack([f for (_w, _c, f) in imgs])
    Ws = np.array([w for (w, _c, _f) in imgs])
    Cs = np.array([c for (_w, c, _f) in imgs])
    chars_uniq = np.array(sorted(set(Cs)))
    rank_acc = defaultdict(list); ll_lda = []; ll_qda = []
    for sp in range(5):
        perm = rng.permutation(len(chars_uniq))
        A = set(chars_uniq[perm[:len(perm) // 2]]); B = set(chars_uniq[perm[len(perm) // 2:]])
        mA = np.array([c in A for c in Cs]); mB = ~mA
        sc = StandardScaler().fit(F[mA])
        pa = PCA(n_components=64, random_state=90).fit(sc.transform(F[mA]))
        Za, Zb = pa.transform(sc.transform(F[mA])), pa.transform(sc.transform(F[mB]))
        ya, yb = Ws[mA], Ws[mB]
        clf = LogisticRegression(max_iter=1500, C=1.).fit(Za, ya)
        U, S, Vt = np.linalg.svd(clf.coef_, full_matrices=False)
        full_acc = float((clf.predict(Zb) == yb).mean())
        for k in range(1, 10):
            Wk = U[:, :k] @ np.diag(S[:k]) @ Vt[:k]
            scores = Zb @ Wk.T + clf.intercept_
            pred = clf.classes_[scores.argmax(1)]
            rank_acc[k].append(float((pred == yb).mean()))
        lda = LinearDiscriminantAnalysis().fit(Za, ya)
        qda = QuadraticDiscriminantAnalysis(reg_param=.5).fit(Za[:, :20], ya)
        from sklearn.metrics import log_loss
        ll_lda.append(log_loss(yb, lda.predict_proba(Zb), labels=lda.classes_))
        ll_qda.append(log_loss(yb, qda.predict_proba(Zb[:, :20]), labels=qda.classes_))
    spec = {k: float(np.mean(v)) for k, v in rank_acc.items()}
    d95 = next((k for k in sorted(spec) if spec[k] >= 0.95 * spec[9]), 9)

    results = dict(
        k_shot=dict(curve=curve, K90=k90,
                    stability={K: float(np.mean(v)) for K, v in stab.items()},
                    cluster_restricted=None),
        interaction=res_int,
        spectrum=dict(rank_acc=spec, d_eff95=d95),
        logloss=dict(lda=float(np.mean(ll_lda)), qda20=float(np.mean(ll_qda)),
                     delta=float(np.mean(ll_lda) - np.mean(ll_qda))))
    out.write_text(json.dumps(results, indent=1))
    print('STRUCTURE_DONE', json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
