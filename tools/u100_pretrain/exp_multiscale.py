"""Multi-scale writer-style separation (G / C / S / L), no generative training.

Scales (per-image features):
  G  DINOv2-B CLS of the whole character
  C  glyph-cluster factor: STD-render CLS K-means (K=12) one-hot + cluster-
     centered CLS residual (content removed within cluster)
  S  stroke statistics: direction x width joint histogram, width percentiles,
     edge softness, ink fraction (Quill-style)
  L  local patches: DINO patch tokens pooled over ink-covered patches
     (mean + std), PCA-controlled like every other block

Measurements per scale combination (G, G+C, G+S, G+L, G+all):
  1. aggregation: energy distance same-writer cross-char (A-half vs B-half of
     the same writer) vs matched cross-writer (same chars, different writer)
  2. writer separability with strict char holdout: 10 random char-half splits;
     linear probe (logistic, PCA-64 capacity control) trained on A chars,
     evaluated on B chars: Top-1, macro OVR-AUC, same-writer retrieval top-1
  3. bootstrap over B images for CIs
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def quill(a256, std_mode=False):
    a = np.asarray(a256, dtype=np.float32) / 255.0
    ink = (a > 0.5) if std_mode else (a < 0.5)
    if not ink.any(): return None
    from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    edt = distance_transform_edt(crop)
    ridge = crop & (edt >= maximum_filter(edt, size=5))
    width = (2.0 * edt - 1.0)[ridge] / 128.0
    sm = gaussian_filter(crop.astype(np.float32), 3.0)
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)[ridge] + 1e-9
    ang = np.arctan2(gy[ridge], gx[ridge]) % np.pi
    dirh, _ = np.histogram(ang, bins=8, range=(0, np.pi)); dirh = dirh / (dirh.sum() + 1e-9)
    di = (np.digitize(ang, np.linspace(0, np.pi, 9)) - 1).clip(0, 7)
    widx = np.digitize(width, [0.015, 0.03, 0.05, 0.08, 1.0])
    joint = np.zeros((8, 6)); np.add.at(joint, (di, widx), 1.0)
    joint = (joint / (joint.sum() + 1e-9)).flatten()
    return np.concatenate([dirh, joint, [float(ink.mean()), float(np.mean(mag)),
                        float(np.percentile(width, 10)), float(np.percentile(width, 50)),
                        float(np.percentile(width, 90)), float(crop.shape[0] / crop.shape[1])]])


def energy_dist(X, Y, rng, n=40, draws=20):
    vals = []
    for _ in range(draws):
        x = X[rng.choice(len(X), min(n, len(X)), replace=False)]
        y = Y[rng.choice(len(Y), min(n, len(Y)), replace=False)]
        dxx = np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1)).mean()
        dyy = np.sqrt(((y[:, None] - y[None]) ** 2).sum(-1)).mean()
        dxy = np.sqrt(((x[:, None] - y[None]) ** 2).sum(-1)).mean()
        vals.append(2 * dxy - dxx - dyy)
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--min-writers', type=int, default=4)
    parser.add_argument('--cap', type=int, default=2)
    parser.add_argument('--clusters', type=int, default=12)
    parser.add_argument('--pca-dim', type=int, default=64)
    parser.add_argument('--splits', type=int, default=10)
    parser.add_argument('--limit', type=int, default=0, help='smoke: cap image count')
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_multiscale.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    cell = defaultdict(list)
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        cell[(w, ord(r['character']))].append((int(r['img_id']), r['image_path']))
    by_char = defaultdict(set)
    for (w, c) in cell: by_char[c].add(w)
    panel = sorted(c for c, ws in by_char.items() if len(ws) >= args.min_writers)
    print('panel chars', len(panel), flush=True)

    images = []   # (writer, char, scale-feature dict)
    bank = np.load(args.bank)
    blook = {int(v): i for i, v in enumerate(bank['ids'])}
    for (w, c), ps in cell.items():
        if c not in by_char or len(by_char[c]) < args.min_writers: continue
        for img_id, p in ps[:args.cap]:
            if img_id not in blook: continue
            images.append(dict(w=w, c=c, img_id=img_id, path=p))
    if args.limit:
        keep = rng.choice(len(images), min(args.limit, len(images)), replace=False)
        images = [images[i] for i in sorted(keep)]
    print('images', len(images), flush=True)
    print('len check', len(images), flush=True)

    from transformers import AutoModel
    from sklearn.cluster import KMeans
    dino = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    dmean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    dstd = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()

    def dino_fwd(imgs, patches=False):
        outs = []
        with torch.inference_mode():
            for s in range(0, len(imgs), 48):
                x = torch.from_numpy(np.stack(imgs[s:s + 48])).cuda().float() / 255
                x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
                x = (x - dmean) / dstd
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    t = dino(pixel_values=x).last_hidden_state
                outs.append(t.float().cpu().numpy())
        return np.concatenate(outs)

    # G from cache; STD cls + clusters; S quill; L patches
    print('assembling G', flush=True)
    FALL = bank['features']
    Gf = np.stack([FALL[blook[im['img_id']]] for im in images])
    print('G ready', flush=True)
    std_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    uniq_chars = sorted({im['c'] for im in images})
    stdp = [std_rows.get((0, chr(c))) or next(v for k, v in std_rows.items() if k[1] == chr(c)) for c in uniq_chars]
    print('std fwd begin', len(stdp), flush=True)
    std_cls = dino_fwd([np.asarray(Image.open(Path(args.std_root) / p).convert('L').resize((224, 224), Image.BICUBIC))[None] for p in stdp])[:, 0]
    print('std fwd done', flush=True)
    km = KMeans(n_clusters=args.clusters, n_init=4, random_state=90).fit(std_cls)
    cluster_of = {c: int(km.labels_[i]) for i, c in enumerate(uniq_chars)}
    print('clusters ready', flush=True)

    print('quill+patch begin', flush=True)
    Sf = np.zeros((len(images), 62), dtype=np.float32)
    Lf = np.zeros((len(images), 1536), dtype=np.float32)
    B = 48
    for s in range(0, len(images), B):
        chunk = images[s:s + B]
        imgs = [np.asarray(Image.open(im['path']).convert('L').resize((224, 224), Image.BICUBIC))[None] for im in chunk]
        toks = dino_fwd(imgs, patches=True)             # [b,257,768]
        for k, (im, im224) in enumerate(zip(chunk, imgs)):
            Sf[s + k] = quill(im224[0], std_mode=False)
            t = toks[k, 1:]                              # [256,768] patches
            ink = np.asarray(Image.fromarray(im224[0]).resize((16, 16), Image.BILINEAR), dtype=np.float32) / 255
            mask = (ink > 0.08).flatten()
            sel = t[mask] if mask.sum() >= 4 else t
            Lf[s + k] = np.concatenate([sel.mean(0), sel.std(0)])
        if (s // B) % 20 == 0: print('feat', s, flush=True)
    print('S/L ready', flush=True)

    cluster_ids = np.array([cluster_of[im['c']] for im in images])
    char_ids = np.array([im['c'] for im in images])
    writer_ids = np.array([im['w'] for im in images])
    Gc_mean = np.stack([Gf[cluster_ids == k].mean(0) for k in range(args.clusters)])
    Cres = Gf - Gc_mean[cluster_ids]
    blocks = dict(G=(Gf,), C=(np.eye(args.clusters)[cluster_ids] * 3.0, Cres), S=(Sf,), L=(Lf,))
    combos = {'G': ['G'], 'G+C': ['G', 'C'], 'G+S': ['G', 'S'], 'G+L': ['G', 'L'], 'G+all': ['G', 'C', 'S', 'L']}

    from sklearn.linear_model import LogisticRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    table = {}
    chars_uniq = np.array(sorted(set(char_ids)))
    for combo, names in combos.items():
        F = np.concatenate([b for n in names for b in blocks[n]], axis=1)
        top1s, aucs, rets, aggs = [], [], [], []
        for sp in range(args.splits):
            perm = rng.permutation(len(chars_uniq))
            A_chars = set(chars_uniq[perm[:len(perm) // 2]]); B_chars = set(chars_uniq[perm[len(perm) // 2:]])
            mA = np.array([im['c'] in A_chars for im in images]); mB = ~mA
            Xa, Xb = F[mA], F[mB]; ya, yb = writer_ids[mA], writer_ids[mB]
            sc = StandardScaler().fit(Xa)
            pa = PCA(n_components=args.pca_dim, random_state=90).fit(sc.transform(Xa))
            Za, Zb = pa.transform(sc.transform(Xa)), pa.transform(sc.transform(Xb))
            clf = LogisticRegression(max_iter=1500, C=1.0).fit(Za, ya)
            pred = clf.predict(Zb)
            top1s.append(float((pred == yb).mean()))
            aucs.append(float(roc_auc_score(yb, clf.predict_proba(Zb), multi_class='ovr', average='macro')))
            cent = np.stack([Za[ya == w].mean(0) for w in range(10)])
            d = np.sqrt(((Zb[:, None] - cent[None]) ** 2).sum(-1))
            rets.append(float((d.argmin(1) == yb).mean()))
            # aggregation on PCA features
            same, cross = [], []
            for w in range(10):
                A_w, B_w = Za[(ya == w)], Zb[(yb == w)]
                if len(A_w) >= 8 and len(B_w) >= 8: same.append(energy_dist(A_w, B_w, rng))
            # matched cross-writer: chars present for both writers in opposite halves
            if len(writer_ids) != len(images) or len(char_ids) != len(images) or len(mA) != len(images):
                raise ValueError(f'length mismatch w={len(writer_ids)} c={len(char_ids)} mA={len(mA)} images={len(images)}')
            ca = defaultdict(dict); cb = defaultdict(dict)
            for k, i in enumerate(np.where(mA)[0]): ca[writer_ids[i]][char_ids[i]] = Za[k]
            for k, i in enumerate(np.where(mB)[0]): cb[writer_ids[i]][char_ids[i]] = Zb[k]
            for w in range(10):
                for v in range(w + 1, 10):
                    cc = sorted(set(ca[w]) & set(cb[v]))
                    if len(cc) >= 8:
                        X1 = np.stack([ca[w][c] for c in cc]); X2 = np.stack([cb[v][c] for c in cc])
                        cross.append(energy_dist(X1, X2, rng))
            aggs.append(float(np.mean(cross) - np.mean(same)))
        bs = []
        for _ in range(200):
            idx = rng.choice(len(yb), len(yb), replace=True)
            bs.append(float((pred[idx] == yb[idx]).mean()))
        table[combo] = dict(top1=round(float(np.mean(top1s)), 4), top1_ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)],
                            auc=round(float(np.mean(aucs)), 4), retrieval=round(float(np.mean(rets)), 4),
                            aggregation=round(float(np.mean(aggs)), 4))
        print(combo, table[combo], flush=True)
    out.write_text(json.dumps(dict(panel_chars=len(panel), images=len(images), table=table), indent=1))
    print('MULTISCALE_DONE')


if __name__ == '__main__':
    main()
