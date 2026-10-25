"""P0+P1: existence / separability / learnability of writer style, measured
properly.

Data: balanced panel (chars covered by the 6 dense writers, >=2 images each).
Targets y[w,c] = mean feature over that cell's images; per-cell scatter gives
the irreducible noise floor eps.

Feature spaces:
  dino  — DINOv2-B CLS of real images (target) and of the STD render (input)
  quill — stroke width x direction joint histogram + edge softness + width
          percentiles of real images; structural stats of the STD render as
          input (skeleton direction histogram, grid density, aspect)

Models (per 5-fold char split, identical training for all):
  M0  y_hat = W0 @ x_c                        content only
  M1  y_hat = W0 @ x_c + b_w                  + additive writer bias
  M2K y_hat = M1 + U @ (a_w * (V @ x_c))      + rank-K writer-char modulation
Trained with full-batch Adam on train chars only; test chars strictly held
out (a test char never contributes to any parameter).

Outputs: SSE0/SSE1/SSE2K on test cells, noise floor SSE_eps (for predicting
the mean of n images), and noise-corrected explanation rates
R_global=(SSE0-SSE1)/(SSE0-SSE_eps), R_interaction(K)=(SSE1-SSE2K)/(SSE0-SSE_eps).
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
W = 6


def quill(img, std_mode):
    """Stroke statistics of one 256px grayscale image. std_mode: STD renders
    have ink=white on black; real images have ink=black on white."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    ink = (a > 0.5) if std_mode else (a < 0.5)
    if not ink.any(): return None
    from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h = crop.shape[0]
    edt = distance_transform_edt(crop)
    ridge = crop & (edt >= maximum_filter(edt, size=5))
    width = (2.0 * edt - 1.0)[ridge] / 128.0
    sm = gaussian_filter(crop.astype(np.float32), 3.0)
    gy, gx = np.gradient(sm)
    mag = np.hypot(gx, gy)[ridge] + 1e-9
    ang = (np.arctan2(gy[ridge], gx[ridge])) % np.pi     # stroke direction perp to gradient
    dirh, _ = np.histogram(ang, bins=8, range=(0, np.pi))
    dirh = dirh / (dirh.sum() + 1e-9)
    wb = np.array([0.015, 0.03, 0.05, 0.08, 1.0])
    widx = np.digitize(width, wb)
    joint = np.zeros((8, 6))
    np.add.at(joint, (np.digitize(ang, np.linspace(0, np.pi, 9)) - 1).clip(0, 7), widx * 0 + 1)
    # joint: direction bin x width bin counts, normalized
    joint2 = np.zeros((8, 6))
    di = (np.digitize(ang, np.linspace(0, np.pi, 9)) - 1).clip(0, 7)
    np.add.at(joint2, (di, widx), 1.0)
    joint2 = (joint2 / (joint2.sum() + 1e-9)).flatten()
    edge = float(np.mean(mag))                            # edge softness proxy
    feats = np.concatenate([dirh, joint2,
                            [ink.mean(), edge,
                             float(np.percentile(width, 10)), float(np.percentile(width, 50)),
                             float(np.percentile(width, 90)), float(h / crop.shape[1])]])
    return feats


def std_struct(img):
    """Structural input for the STD render: direction histogram + 4x4 density."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    ink = a > 0.5
    if not ink.any(): return None
    from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
    ys, xs = np.where(ink)
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    edt = distance_transform_edt(crop)
    ridge = crop & (edt >= maximum_filter(edt, size=5))
    sm = gaussian_filter(crop.astype(np.float32), 3.0)
    gy, gx = np.gradient(sm)
    ang = (np.arctan2(gy[ridge], gx[ridge])) % np.pi
    dirh, _ = np.histogram(ang, bins=8, range=(0, np.pi))
    dirh = dirh / (dirh.sum() + 1e-9)
    dens = crop.reshape(4, crop.shape[0] // 4, 4, crop.shape[1] // 4).mean((1, 3)) if \
        crop.shape[0] % 4 == 0 and crop.shape[1] % 4 == 0 else np.zeros((4, 4))
    return np.concatenate([dirh, dens.flatten(), [crop.mean(), crop.shape[0] / crop.shape[1]]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--ks', type=int, nargs='+', default=[4, 8, 16, 32])
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_m1_predict.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    cell = defaultdict(list)
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        if w < W: cell[(w, ord(r['character']))].append(r['image_path'])
    panel = sorted(c for c in {c for (w, c) in cell}
                   if all(len(cell.get((w, c), [])) >= 2 for w in range(W)))
    if len(panel) < 30: raise ValueError(f'panel {len(panel)}')

    from transformers import AutoModel
    dino = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    dmean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    dstd = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()

    def dino_cls(imgs):
        outs = []
        with torch.inference_mode():
            for s in range(0, len(imgs), 32):
                x = torch.from_numpy(np.stack(imgs[s:s + 32])).cuda().float() / 255
                x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
                x = (x - dmean) / dstd
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    t = dino(pixel_values=x).last_hidden_state[:, 0]
                outs.append(t.float().cpu().numpy())
        return np.concatenate(outs)

    bank = np.load(args.bank)
    bank_lookup = {int(v): i for i, v in enumerate(bank['ids'])}
    img_ids_by_cell = {}
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        if w < W: img_ids_by_cell.setdefault((w, ord(r['character'])), []).append(int(r['img_id']))

    std_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}

    spaces = {}
    # dino space
    Yd = np.zeros((len(panel), W, 768)); Xd = np.zeros((len(panel), 768))
    eps_d = np.zeros((len(panel), W))
    for ci, c in enumerate(panel):
        imgs = []
        for w in range(W):
            ids = img_ids_by_cell[(w, c)][:2]
            vecs = np.stack([bank['features'][bank_lookup[i]] for i in ids])
            Yd[ci, w] = vecs.mean(0)
            if len(ids) > 1: eps_d[ci, w] = float((vecs - vecs.mean(0) ** 0).var(0).sum()) if False else float(((vecs - vecs.mean(0)) ** 2).sum() / len(ids))
            stdp = std_rows.get((0, chr(c))) or next(v for k, v in std_rows.items() if k[1] == chr(c))
            imgs.append(np.asarray(Image.open(Path(args.std_root) / stdp).convert('L').resize((224, 224), Image.BICUBIC))[None])
        Xd[ci] = dino_cls(imgs)[0]
    spaces['dino'] = (Yd, Xd, eps_d, 768)

    # quill space
    for ci, c in enumerate(panel[:1]):  # warm polarity check
        pass
    Yq = None; Xq = None; eps_q = None
    try:
        qY = np.zeros((len(panel), W, 8 * 6 + 6)); qX = np.zeros((len(panel), 8 + 16 + 2))
        qE = np.zeros((len(panel), W))
        ok = True
        for ci, c in enumerate(panel):
            for w in range(W):
                qv = []
                for p in cell[(w, c)][:2]:
                    f = quill(Image.open(p).convert('L'), std_mode=False)
                    if f is None: ok = False; break
                    qv.append(f)
                if not ok or not qv: ok = False; break
                qY[ci, w] = np.mean(qv, 0)
                qE[ci, w] = float(np.mean([((v - np.mean(qv, 0)) ** 2).sum() for v in qv]) / len(qv))
            if not ok: break
            stdp = std_rows.get((0, chr(c))) or next(v for k, v in std_rows.items() if k[1] == chr(c))
            sx = std_struct(Image.open(Path(args.std_root) / stdp).convert('L'))
            if sx is None: ok = False; break
            qX[ci] = sx
        if ok: spaces['quill'] = (qY, qX, qE, qY.shape[2])
    except Exception as e:
        print('quill space skipped:', repr(e))

    results = {}
    for space, (Y, X, eps, dy) in spaces.items():
        folds = []
        idx = np.arange(len(panel))
        folds = [idx[k::args.folds] for k in range(args.folds)]
        agg = defaultdict(list)
        for f, test_idx in enumerate(folds):
            train_idx = np.setdiff1d(idx, test_idx)
            Xt = X[train_idx]; Yt = Y[train_idx]          # [n_train_c, W, d]
            Xs = Xt.reshape(-1, X.shape[1]); Ys = Yt.reshape(-1, Y.shape[2])
            wtr = np.tile(np.arange(W), len(train_idx))
            mu_x, sd_x = Xs.mean(0), Xs.std(0) + 1e-9
            mu_y, sd_y = Ys.mean(0), Ys.std(0) + 1e-9
            xs = (Xs - mu_x) / sd_x; ys = (Ys - mu_y) / sd_y
            xte = (X[test_idx] - mu_x) / sd_x
            yte = (Y[test_idx] - mu_y) / sd_y
            n_tr = len(train_idx)

            def train_model(kind, K=0):
                torch.manual_seed(90)
                x_t = torch.tensor(np.repeat(xs, W, axis=0), dtype=torch.float32).cuda()
                y_t = torch.tensor(ys, dtype=torch.float32).cuda()
                w_t = torch.tensor(np.tile(np.arange(W), n_tr), dtype=torch.long).cuda()
                W0 = torch.zeros(X.shape[1], dy, device='cuda', requires_grad=True)
                params = [W0]
                if kind >= 1:
                    bw = torch.zeros(W, dy, device='cuda', requires_grad=True); params.append(bw)
                if kind == 2:
                    V = torch.randn(K, X.shape[1], device='cuda') * 0.01; V.requires_grad_(True)
                    aw = torch.zeros(W, K, device='cuda'); aw.requires_grad_(True)
                    U = torch.zeros(dy, K, device='cuda'); U.requires_grad_(True)
                    params += [V, aw, U]
                opt = torch.optim.Adam(params, lr=5e-3, weight_decay=1e-4)
                for _ in range(args.steps):
                    pred = x_t @ W0
                    if kind >= 1: pred = pred + bw[w_t]
                    if kind == 2: pred = pred + (U @ (aw[w_t].T * (V @ x_t.T))).T
                    loss = ((pred - y_t) ** 2).mean()
                    opt.zero_grad(); loss.backward(); opt.step()
                return dict(W0=W0.detach().cpu().numpy(), **({} if kind == 0 else dict(
                    bw=bw.detach().cpu().numpy(), **({} if kind == 1 else dict(
                        V=V.detach().cpu().numpy(), aw=aw.detach().cpu().numpy(), U=U.detach().cpu().numpy())))))

            sses = {}
            for kind, K in ((0, 0), (1, 0)) + tuple((2, k) for k in args.ks):
                pr = train_model(kind, K)
                sse = 0.0
                for ti, c in enumerate(test_idx):
                    x_c = xte[ti]
                    for w in range(W):
                        p = x_c @ pr['W0']
                        if kind >= 1: p = p + pr['bw'][w]
                        if kind == 2: p = p + pr['U'] @ (pr['aw'][w] * (pr['V'] @ x_c))
                        sse += float(((p - yte[ti, w]) ** 2).sum())
                sses[f'M{kind}' if kind < 2 else f'M2_K{K}'] = sse
            sse_eps = 0.0
            for ti, c in enumerate(test_idx):
                for w in range(W):
                    n = len(img_ids_by_cell[(w, c)][:2])
                    sse_eps += eps[test_idx[ti], w] * (n - 1) / max(n, 1) * Y.shape[2] / Y.shape[2]
            sse_eps *= Y.shape[2] / 768 if space == 'dino' else 1.0
            denom = max(sses['M0'] - sse_eps, 1e-9)
            rec = dict(fold=f, sse0=sses['M0'], sse1=sses['M1'], sse_eps=sse_eps,
                       R_global=(sses['M0'] - sses['M1']) / denom,
                       **{f'R_int_K{k}': (sses['M1'] - sses[f'M2_K{k}']) / denom for k in args.ks},
                       **{f'sse2_K{k}': sses[f'M2_K{k}'] for k in args.ks})
            folds.append(rec)
            for k, v in rec.items():
                if k != 'fold': agg[k].append(v)
        results[space] = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)))
                          for k, v in agg.items() if k != 'fold'}
        results[space]['folds'] = folds
    out.write_text(json.dumps(dict(panel_chars=len(panel), results=results), indent=1))
    for space, r in results.items():
        print(f'== {space} ==')
        for k, v in r.items():
            if k != 'folds': print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
