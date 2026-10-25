"""M1 vs M2 on char-centered residuals: global writer bias vs writer x STD
interaction, strict char holdout.

Target (leave-writer-out centering, no self leakage):
    r[w,c] = y[w,c] - mean_{v!=w} y[v,c]
M1: r ~ b_w                      (per-writer constant)
M2: r ~ b_w + U (a_w * (V g_c))  (rank-K bilinear with STD stats g_c)
K in {2,4,8,16}; 5 char folds; full-batch Adam.
Outputs test MSE, train MSE, Delta R^2 (test), train-test gap (memorization).
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def std_feat(im):
    a = np.asarray(im, dtype=np.float32) / 255.0
    ink = a > .5
    ys, xs = np.where(ink)
    if not len(ys): return None
    h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
    from scipy.ndimage import distance_transform_edt, maximum_filter, gaussian_filter
    crop = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    edt = distance_transform_edt(crop)
    ridge = crop & (edt >= maximum_filter(edt, size=5))
    sm = gaussian_filter(crop.astype(np.float32), 3.0)
    gy, gx = np.gradient(sm)
    ang = np.arctan2(gy[ridge], gx[ridge]) % np.pi if ridge.any() else np.zeros(0)
    dirh, _ = np.histogram(ang, bins=8, range=(0, np.pi))
    dirh = dirh / (dirh.sum() + 1e-9)
    wid = (2 * edt - 1)[ridge] / 64.0 if ridge.any() else np.zeros(1)
    g8 = crop.reshape(8, crop.shape[0] // 8, 8, crop.shape[1] // 8).mean((1, 3)).flatten() \
        if crop.shape[0] % 8 == 0 and crop.shape[1] % 8 == 0 else np.zeros(64)
    return np.concatenate([[h / w, ink.mean()], dirh, g8,
                           [np.percentile(wid, 10), np.percentile(wid, 50), np.percentile(wid, 90)]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--bank', default='assets/u100_pretrain_20261024/style_bank_top10.npz')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--steps', type=int, default=2500)
    parser.add_argument('--ks', type=int, nargs='+', default=[2, 4, 8, 16])
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_m1m2.json')
    parser.add_argument('--g-dino', action='store_true', help='use DINO CLS of STD render as g_c')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    from PIL import Image
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    bank = np.load(args.bank)
    FALL = bank['features']
    blook = {int(v): i for i, v in enumerate(bank['ids'])}
    by_wc = defaultdict(list)
    for r in rows:
        w = TOP10_RAW[int(r['calligrapher_id'])]
        iid = int(r['img_id'])
        if iid in blook and w < 10:
            by_wc[(w, ord(r['character']))].append(FALL[blook[iid]])
    by_char = defaultdict(set)
    for (w, c) in by_wc: by_char[c].add(w)
    panel = {c for c, ws in by_char.items() if len(ws) >= 4}
    y = {k: np.mean(v, 0) for k, v in by_wc.items() if k[1] in panel}
    idx_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    g = {}
    use_dino = '--g-dino' in sys.argv
    if use_dino:
        from transformers import AutoModel
        import torch as _t
        dino = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
        dm = _t.tensor([.485, .456, .406])[None, :, None, None].cuda()
        ds = _t.tensor([.229, .224, .225])[None, :, None, None].cuda()
        clist = sorted(panel)
        paths = []
        for c in clist:
            ch = chr(c)
            key = next((k for k in idx_rows if k[1] == ch), None)
            paths.append(None if key is None else idx_rows[key])
        with _t.inference_mode():
            for i0 in range(0, len(clist), 32):
                batch = []
                keep = []
                for i, pth in enumerate(paths[i0:i0 + 32]):
                    if pth is None: continue
                    im = np.asarray(Image.open(Path(args.std_root) / pth).convert('L').resize((224, 224), Image.BICUBIC))
                    batch.append(im[None]); keep.append(i)
                if not batch: continue
                x = _t.from_numpy(np.stack(batch)).cuda().float() / 255
                x = _t.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
                x = (x - dm) / ds
                with _t.autocast('cuda', dtype=_t.bfloat16):
                    t = dino(pixel_values=x).last_hidden_state[:, 0]
                t = _t.nn.functional.normalize(t.float(), dim=-1).cpu().numpy()
                for j, i in enumerate(keep): g[clist[i0 + i]] = t[j].astype(np.float32)
        print('g via DINO CLS', len(g), flush=True)
    else:
        for c in panel:
            ch = chr(c)
            key = next((k for k in idx_rows if k[1] == ch), None)
            if key is None: continue
            f = std_feat(Image.open(Path(args.std_root) / idx_rows[key]).convert('L'))
            if f is not None: g[c] = f
    cells = [(w, c) for (w, c) in y if c in g]
    r = {}
    for (w, c) in cells:
        others = [y[(v, c)] for v in by_char[c] if v != w and (v, c) in y]
        if others: r[(w, c)] = y[(w, c)] - np.mean(others, 0)
    cells = [k for k in cells if k in r]
    chars = np.array(sorted({c for (_w, c) in cells}))
    Gm = np.stack([g[c] for c in chars])
    Gm = (Gm - Gm.mean(0)) / (Gm.std(0) + 1e-9)
    gidx = {c: i for i, c in enumerate(chars)}
    R = np.stack([r[k] for k in cells])
    Wv = np.array([k[0] for k in cells])
    Cv = np.array([k[1] for k in cells])
    print('cells', len(cells), 'chars', len(chars), 'g_dim', Gm.shape[1], flush=True)

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    res = defaultdict(lambda: defaultdict(list))
    for fold in range(args.folds):
        te = set(chars[rng.permutation(len(chars))][: max(len(chars) // args.folds, 1)].tolist()) \
            if fold < 4 else None
        perm = rng.permutation(len(chars))
        n_te = len(chars) // args.folds
        te_set = set(chars[perm[fold * n_te:(fold + 1) * n_te]].tolist())
        mte = np.array([c in te_set for c in Cv]); mtr = ~mte
        # M1 closed form
        b = np.stack([R[mtr & (Wv == w)].mean(0) for w in range(10)])
        mse1_te = float(((R[mte] - b[Wv[mte]]) ** 2).mean())
        mse1_tr = float(((R[mtr] - b[Wv[mtr]]) ** 2).mean())
        res['M1']['test'].append(mse1_te); res['M1']['train'].append(mse1_tr)
        for K in args.ks:
            torch.manual_seed(90)
            bw = torch.tensor(b, dtype=torch.float32, device=dev)
            U = (torch.randn(768, K, device=dev) * .01).requires_grad_(True)
            V = (torch.randn(K, Gm.shape[1], device=dev) * .01).requires_grad_(True)
            A = (torch.randn(10, K, device=dev) * .01).requires_grad_(True)
            Rt = torch.tensor(R, dtype=torch.float32, device=dev)
            Gt = torch.tensor(Gm, dtype=torch.float32, device=dev)
            Wt = torch.tensor(Wv, dtype=torch.long, device=dev)
            Ct = torch.tensor(np.array([gidx[c] for c in Cv]), dtype=torch.long, device=dev)
            Mtr = torch.tensor(mtr, device=dev)
            opt = torch.optim.Adam([U, V, A], lr=5e-3, weight_decay=1e-4)
            for _ in range(args.steps):
                pred = bw[Wt] + (A[Wt] * (Gt[Ct] @ V.T)) @ U.T
                loss = ((pred - Rt) ** 2)[Mtr].mean()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                contrib = (A[Wt] * (Gt[Ct] @ V.T)) @ U.T
                pred = bw[Wt] + contrib
                mse2_tr = float(((pred - Rt) ** 2)[Mtr].mean())
                mse2_te = float(((pred - Rt) ** 2)[~Mtr].mean())
                cn = float(contrib.norm() / Rt.norm())
            res[f'M2_K{K}']['test'].append(mse2_te); res[f'M2_K{K}']['train'].append(mse2_tr)
        res[f'M2_K{K}']['contrib_rel_norm'].append(cn)
        print('fold', fold, 'M1_te', round(mse1_te, 5), flush=True)
    summary = {}
    m1 = np.mean(res['M1']['test'])
    for k, v in res.items():
        te, tr = float(np.mean(v['test'])), float(np.mean(v['train']))
        summary[k] = dict(test_mse=te, train_mse=tr,
                          dR2_test=float((m1 - te) / m1), dR2_train=float((m1 - tr) / m1) if 'M2' in k else 0.0)
    out.write_text(json.dumps(dict(n_cells=len(cells), folds=args.folds, summary=summary), indent=1))
    print('M1M2_DONE', json.dumps(summary))


if __name__ == '__main__':
    main()
