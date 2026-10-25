"""P0b (lean): patch-token interaction structure and held-out prediction.

Spectrum: with only C~105 rows, all nonzero singular values of the
interaction matrix come from its small Gram G = I I^T [C,C]; eigenvalues of G
give the exact spectrum instantly (no full SVD). Permutation nulls likewise.

Prediction: STD patch features are PCA-reduced (fitted on the panel via the
same small-Gram eigendecomposition) to --pca-dim; ridge from PCA features to
the held-out char's writer-difference map (residual over the pair-mean B0),
solved per LOO fold on the small side.
"""
import argparse
from PIL import Image
import csv
import json
from collections import defaultdict
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
TOP10_RAW = {601: 0, 703: 1, 479: 2, 956: 3, 9001: 4, 760: 5, 672: 6, 365: 7, 461: 8, 53: 9}


def dino(model, imgs, mean, std):
    outs = []
    with torch.inference_mode():
        for s in range(0, len(imgs), 32):
            x = torch.from_numpy(np.stack(imgs[s:s + 32])).cuda().float() / 255
            x = torch.nn.functional.interpolate(x, (224, 224), mode='bicubic', align_corners=False, antialias=True).clamp_(0, 1).expand(-1, 3, -1, -1)
            x = (x - mean) / std
            with torch.autocast('cuda', dtype=torch.bfloat16):
                t = model(pixel_values=x).last_hidden_state[:, 1:]
            outs.append(t.float().cpu().numpy())
    return np.concatenate(outs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-csv', default='assets/generic48_kxl_top10_20261022/top10_train.csv')
    parser.add_argument('--std-root', default='assets/generic48_kxl_top10_20261022/std_bbox64')
    parser.add_argument('--writers', type=int, default=6)
    parser.add_argument('--min-imgs', type=int, default=2)
    parser.add_argument('--cap', type=int, default=2)
    parser.add_argument('--pca-dim', type=int, default=128)
    parser.add_argument('--ridge-lambda', type=float, default=1.0)
    parser.add_argument('--perms', type=int, default=50)
    parser.add_argument('--out', default='assets/u100_pretrain_20261024/exp_a3_patch.json')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists(): raise FileExistsError(out)
    rng = np.random.default_rng(90)
    W = args.writers
    rows = [r for r in csv.DictReader(open(args.images_csv, encoding='utf-8-sig')) if r['generic'] == '0']
    wc = defaultdict(list)
    for r in rows: wc[(TOP10_RAW[int(r['calligrapher_id'])], ord(r['character']))].append(r['image_path'])
    panel = sorted(c for c in {c for (w, c) in wc}
                   if all(len(wc.get((w, c), [])) >= args.min_imgs for w in range(W)))
    if len(panel) < 30: raise ValueError(f'panel too small {len(panel)}')

    from transformers import AutoModel
    model = AutoModel.from_pretrained('facebook/dinov2-base', local_files_only=True).cuda().eval().requires_grad_(False)
    mean = torch.tensor([.485, .456, .406])[None, :, None, None].cuda()
    std = torch.tensor([.229, .224, .225])[None, :, None, None].cuda()
    P = np.zeros((len(panel), W, 256, 768), dtype=np.float32)
    for ci, c in enumerate(panel):
        for w in range(W):
            imgs = [np.asarray(Image.open(p).convert('L').resize((224, 224), Image.BICUBIC))[None]
                    for p in wc[(w, c)][:args.cap]]
            P[ci, w] = dino(model, imgs, mean, std).mean(0)
    std_rows = {(int(r['script_id']), r['character']): r['path']
                for r in csv.DictReader(open(Path(args.std_root) / 'index.csv', encoding='utf-8-sig'))}
    std_imgs = []
    for c in panel:
        p = std_rows.get((0, chr(c))) or next(v for k, v in std_rows.items() if k[1] == chr(c))
        std_imgs.append(np.asarray(Image.open(Path(args.std_root) / p).convert('L').resize((224, 224), Image.BICUBIC))[None])
    S = dino(model, std_imgs, mean, std)                    # [C,256,768]

    g = P - P.mean(1)[:, None]                              # remove char content
    a = g.mean(0)                                           # writer main effect
    I = g - a[None]                                         # [C,W,256,768]
    e_a = float((a ** 2).mean()); e_i = float((I ** 2).mean())
    I_mat = I.reshape(len(panel), -1)                       # [C, W*256*768]

    def spectrum(mat):
        Gv = np.linalg.eigvalsh(mat @ mat.T)
        sv = np.sqrt(np.clip(Gv, 0, None))[::-1]
        return sv / (sv.sum() + 1e-12)
    spec_obs = spectrum(I_mat)
    top10_obs = float(spec_obs[:10].sum())
    top10_null = []
    for _ in range(args.perms):
        perm = np.empty_like(P)
        for w in range(W): perm[:, w] = P[rng.permutation(len(panel)), w]
        gp = perm - perm.mean(1)[:, None]
        Ip = (gp - gp.mean(0)[None]).reshape(len(panel), -1)
        top10_null.append(float(spectrum(Ip)[:10].sum()))
    top10_null = np.array(top10_null)

    # PCA of STD patches via small Gram
    Xf = S.reshape(len(panel), -1)
    Xc = Xf - Xf.mean(0)
    Gx = Xc @ Xc.T
    evals, evecs = np.linalg.eigh(Gx)
    order = np.argsort(-evals)[:args.pca_dim]
    comp = evecs[:, order] * np.sqrt(np.clip(evals[order], 0, None))[None, :]
    Xp = comp / (np.linalg.norm(comp, axis=0, keepdims=True) + 1e-9)   # [C,pca]

    per_pair = []
    for A in range(W):
        for B in range(A + 1, W):
            Y = (P[:, A] - P[:, B]).reshape(len(panel), -1)
            cos0, cos1, cos2, r0, r2 = [], [], [], [], []
            for j in range(len(panel)):
                others = [i for i in range(len(panel)) if i != j]
                Yt = Y[others]; b0 = Yt.mean(0)
                sims = (S.mean(1)[others] @ S.mean(1)[j]) / .15
                sims -= sims.max(); w1 = np.exp(sims); w1 /= w1.sum()
                b1 = (w1[:, None] * Yt).sum(0)
                Xt = Xp[others]
                coef = np.linalg.solve(Xt.T @ Xt + args.ridge_lambda * np.eye(Xt.shape[1]),
                                       Xt.T @ (Yt - b0))
                b2 = b0 + Xp[j] @ coef
                d = Y[j]
                for b, cs in ((b0, cos0), (b1, cos1), (b2, cos2)):
                    cs.append(float(b @ d / (np.linalg.norm(b) * np.linalg.norm(d) + 1e-9)))
                r0.append(float(((b0 - d) ** 2).mean())); r2.append(float(((b2 - d) ** 2).mean()))
            per_pair.append(dict(pair=f'{A}-{B}', cos_B0=float(np.mean(cos0)),
                                 cos_B1=float(np.mean(cos1)), cos_B2=float(np.mean(cos2)),
                                 relmse_B0=float(np.mean(r0) / (Y ** 2).mean()),
                                 relmse_B2=float(np.mean(r2) / (Y ** 2).mean())))
    summary = dict(panel_chars=len(panel),
                   patch_energy_share=dict(writer_main_a=round(e_a / (e_a + e_i), 4),
                                           interaction_i=round(e_i / (e_a + e_i), 4)),
                   interaction_spectrum=dict(top10_frac_observed=round(top10_obs, 4),
                                             top10_frac_null_mean=round(float(top10_null.mean()), 4),
                                             p_value=round(float((top10_null >= top10_obs).mean()), 4)),
                   heldout_prediction=dict(
                       mean_cos_B0=round(float(np.mean([r['cos_B0'] for r in per_pair])), 4),
                       mean_cos_B1=round(float(np.mean([r['cos_B1'] for r in per_pair])), 4),
                       mean_cos_B2=round(float(np.mean([r['cos_B2'] for r in per_pair])), 4),
                       mean_relmse_B0=round(float(np.mean([r['relmse_B0'] for r in per_pair])), 4),
                       mean_relmse_B2=round(float(np.mean([r['relmse_B2'] for r in per_pair])), 4),
                       n_pairs=len(per_pair)),
                   note='spectrum from the [C,C] Gram (exact); B2 = ridge(PCA128(STD patch)) on residual over B0, LOO chars')
    out.write_text(json.dumps(dict(summary=summary, per_pair=per_pair), indent=1))
    print('EXP_A3_DONE', json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
