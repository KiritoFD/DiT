#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_loss_race.py — 各种损失实际训一遍, 看图空间结果谁真的产出了骨架。

判据全部在**图空间**(VAE 解码后, 只用 no_grad, 不进损失): 墨量 / 连通分量 / 骨架IoU。
这是唯一有意义的判据 —— latent MSE 会把"输出模糊平均骨架"评为最优。

对比:
  A. 原始 latent MSE            (v1 的损失)
  B. 原始 latent MSE + 梯度L1   (抗模糊, 零额外显存)
  C. probe 空间 MSE             (我上一版)
  D. 图空间 BCE(pos_weight)+Dice (VAE 解码, 但只在极小 sub-batch 上算梯度)
"""
import os
import sys
import glob
import time
import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'
N, STEPS, BATCH = 1024, 1500, 128


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def main():
    from src.model.deform_skel import DeformSkel
    from src.train.latent_structure import LatentSkelProbe
    from diffusers.models import AutoencoderKL
    from skimage.morphology import skeletonize
    from scipy.ndimage import label, binary_dilation, generate_binary_structure

    G, gid = load_all('data/top10_style23/shards_std')
    T, _ = load_all('data/top10_style23/shards_aux_skel3')
    import csv
    rows = list(csv.DictReader(open('assets/train_top10_style23.csv', encoding='utf-8')))
    slots = sorted({r['slot_name'] for r in rows})
    s2i = {s: i for i, s in enumerate(slots)}
    id2s = {}
    for r in rows:
        try:
            id2s[int(r['img_id'])] = s2i[r['slot_name']]
        except Exception:
            pass
    rng = np.random.RandomState(0)
    sel = rng.choice(len(G), N, replace=False)
    Gs = torch.from_numpy(G[sel]).to(DEV)
    Ts = torch.from_numpy(T[sel]).to(DEV)
    sid = torch.from_numpy(np.array([id2s.get(int(i), 0) for i in np.array(gid)[sel]])).to(DEV)
    Y = torch.load('assets/callig_script_emb_top10.pt', map_location='cpu', weights_only=False)
    if isinstance(Y, dict):
        Y = Y.get('emb', Y.get('weight', list(Y.values())[0]))
    Y = Y.float().to(DEV)

    ck = torch.load('assets/structure_probes/latent_skel_probe_v1_mse/best.pt',
                    map_location='cpu', weights_only=False)
    pa = ck['args']
    probe = LatentSkelProbe(pa['in_channels'], pa['out_channels'], pa['width'], pa['depth'])
    probe.load_state_dict(ck['model'], strict=False)
    probe = probe.to(DEV).eval()
    for p in probe.parameters():
        p.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    for p in vae.parameters():
        p.requires_grad_(False)
    sc = float(getattr(vae.config, 'scaling_factor', 0.18215))

    @torch.no_grad()
    def dec(z):
        outs = []
        for i in range(0, z.shape[0], 16):
            im = vae.decode(z[i:i + 16].to(DEV).float() / sc).sample
            outs.append((((im.clamp(-1, 1) + 1) / 2).mean(1).float().cpu()))
        return torch.cat(outs, 0)

    ST = generate_binary_structure(2, 2)

    def px_metrics(g2):
        """g2: (N,4,32,32) -> 图空间指标。"""
        g = dec(g2)
        ink = (g < 0.5)
        sk = torch.stack([torch.from_numpy(binary_dilation(
            skeletonize(a.numpy() < 0.5), ST, iterations=1)) for a in g])
        return dict(ink=float(ink.float().mean()), nc=float(np.mean(
            [label(a.numpy(), ST)[1] for a in sk])))

    with torch.no_grad():
        gt_g = dec(Ts)
        gt_ink = (gt_g < 0.5)
        gt_sk = torch.stack([torch.from_numpy(binary_dilation(
            skeletonize(a.numpy() < 0.5), ST, iterations=1)) for a in gt_g])
        gt_m = dict(ink=float(gt_ink.float().mean()),
                    nc=float(np.mean([label(a.numpy(), ST)[1] for a in gt_sk])))

    def gl1(x):
        gx = (x[..., :, 1:] - x[..., :, :-1]).abs().mean()
        gy = (x[..., 1:, :] - x[..., :-1, :]).abs().mean()
        tx = (Ts[bi_g[0]][..., :, 1:] - Ts[bi_g[0]][..., :, :-1]).abs().mean()
        ty = (Ts[bi_g[0]][..., 1:, :] - Ts[bi_g[0]][..., :-1, :]).abs().mean()
        return (gx - tx).abs() + (gy - ty).abs()

    print(f"参考 GT: 墨量 {gt_m['ink']:.4f}  连通分量 {gt_m['nc']:.1f}")
    print(f"\n{'损失':<34}{'latentMSE':>11}{'墨量':>9}{'分量':>8}{'骨架IoU':>10}{'sec':>7}")
    print("  " + "-" * 78)

    losses = ['A. 原始 latent MSE', 'B. latent MSE + 1.0*梯度L1',
              'C. probe 空间 MSE', 'D. 图空间 BCE+Dice (sub=8)']
    for tag in losses:
        torch.manual_seed(0)
        m = DeformSkel(cond_dim=128, ch=4, grid=32, style_ch=32, width=128, max_off=6.0,
                       coarse=8, residual=0, res_cap=1.0, blur=0, dt_ch=1, affine=1, ckpt=0,
                       stroke_mod=1, stroke_cap=1.0, gate_radius=0.25, topo_mode=0,
                       warp_iters=1, film_mode='film', style_tokens=0, attn_heads=4,
                       preserve_amp=0).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS, eta_min=1e-4)
        t0 = time.time()
        for st in range(STEPS):
            bi = torch.randint(0, N, (BATCH,), device=DEV)
            bi_g = (bi,)
            g2 = m(Gs[bi], Y[sid[bi]])
            if tag.startswith('A'):
                loss = F.mse_loss(g2, Ts[bi])
            elif tag.startswith('B'):
                loss = F.mse_loss(g2, Ts[bi]) + 1.0 * gl1(g2)
            elif tag.startswith('C'):
                loss = F.mse_loss(probe(g2).float(), Ts[bi])
            else:
                k = bi[:8]
                img = vae.decode(m(Gs[k], Y[sid[k]])[:8] / sc).sample
                pr = ((img.clamp(-1, 1) + 1) / 2).mean(1, keepdim=True)
                tg = dec(Ts[k]).to(DEV).unsqueeze(1)
                tg = F.interpolate(tg, size=pr.shape[-2:], mode='nearest')
                pos = tg.sum().clamp_min(1.0)
                pw = ((tg.numel() - pos) / pos).clamp(1.0, 10.0)
                bce = F.binary_cross_entropy_with_logits(
                    (pr - 0.5) * 12.0, tg.expand_as(pr), pos_weight=pw)
                p = torch.sigmoid((pr - 0.5) * 12.0)
                dice = 1 - (2 * (p * tg).sum() + 1e-6) / (p.sum() + tg.sum() + 1e-6)
                # 用同样的图给整批算 latent MSE (保证其它样本也有梯度)
                loss = F.mse_loss(g2, Ts[bi]) + 2.0 * (bce + dice)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sch.step()
        m.eval()
        with torch.no_grad():
            g2 = m(Gs, Y[sid])
            lmse = float(F.mse_loss(g2, Ts))
            pm = px_metrics(g2)
            g = dec(g2)
            sk = torch.stack([torch.from_numpy(binary_dilation(
                skeletonize(a.numpy() < 0.5), ST, iterations=1)) for a in g])
            iou = float((sk & gt_sk).sum().float() / (sk | gt_sk).sum().clamp_min(1).float())
        print(f"  {tag:<32}{lmse:>11.5f}{pm['ink']:>9.4f}{pm['nc']:>8.1f}"
              f"{iou:>10.4f}{time.time()-t0:>7.0f}")
        del m, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
