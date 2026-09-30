#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/diag_grad_loss.py — 不挂 VAE 的替代: latent 空间"抗模糊"损失能不能治退化吸引子?

退化吸引子的本质: MSE 的最优解是「数据集平均骨架 latent」, 它解码出来是一团**模糊的灰**。
模糊 => **空间梯度小**。所以只要把空间梯度加进损失, "平均骨架"就会被重罚。

本脚本对几种候选预测算:
   MSE(latent) / L1(latent) / 空间梯度L1 / 组合, 看**排序**是否正确
   (好的预测必须全项最低, 平均骨架必须被重罚)。
零额外显存、精确可微、不挂 VAE。
"""
import os
import sys
import glob
import numpy as np
import torch
import torch.nn.functional as F

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
sys.stdout.reconfigure(encoding="utf-8")
DEV = 'cuda'


def load_all(d):
    L, I = [], []
    for sp in sorted(glob.glob(os.path.join(d, 'shard_*.npz'))):
        with np.load(sp) as z:
            L.append(z['latents'].astype(np.float32))
            I.append(z['img_ids'])
    return np.concatenate(L), np.concatenate(I)


def grad_terms(x):
    """x: (N,4,32,32) -> (gx_l1, gy_l1, gx_l2, gy_l2) 空间梯度(每个样本)。"""
    gx = (x[..., :, 1:] - x[..., :, :-1]).abs().flatten(1).mean(1)
    gy = (x[..., 1:, :] - x[..., :-1, :]).abs().flatten(1).mean(1)
    return gx, gy


def main():
    G, gid = load_all('data/top10_style23/shards_std')
    T, _ = load_all('data/top10_style23/shards_aux_skel3')
    rng = np.random.RandomState(0)
    sel = rng.choice(len(G), 512, replace=False)
    Gt = torch.from_numpy(G[sel]).to(DEV)
    Tt = torch.from_numpy(T[sel]).to(DEV)

    cands = {
        '真值 g_gt (好预测)': Tt,
        '平均骨架 mean(g_gt)': Tt.mean(0, keepdim=True).expand_as(Tt),
        'g_std (无形变)': Gt,
        '空白 z_bg': torch.zeros_like(Tt),
    }
    # 空白用真实白底 latent
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('data/pretrained/pretrained_models/sd-vae-ft-ema',
                                        local_files_only=True).eval().to(DEV)
    with torch.no_grad():
        zb = (vae.encode(torch.ones(1, 3, 256, 256, device=DEV)).latent_dist.mean
              * float(getattr(vae.config, 'scaling_factor', 0.18215))).float()
    cands['空白 z_bg'] = zb.expand_as(Tt)
    del vae
    torch.cuda.empty_cache()

    gt_gx, gt_gy = grad_terms(Tt)

    print("=" * 92)
    print("各种候选预测的损失值 —— 好预测必须**全项最低**")
    print("=" * 92)
    hdr = (f"  {'预测':<22}{'MSE':>10}{'L1':>10}{'梯度L1':>10}{'梯度L2':>10}"
           f"{'MSE+梯度':>11}{'排序':>8}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    rows = []
    for name, P in cands.items():
        mse = F.mse_loss(P, Tt).item()
        l1 = F.l1_loss(P, Tt).item()
        gx, gy = grad_terms(P)
        gl1 = float(((gx - gt_gx).abs() + (gy - gt_gy).abs()).mean())
        gl2 = float(((gx - gt_gx).pow(2) + (gy - gt_gy).pow(2)).mean())
        rows.append((name, mse, l1, gl1, gl2))
    for name, mse, l1, gl1, gl2 in rows:
        print(f"  {name:<22}{mse:>10.5f}{l1:>10.5f}{gl1:>10.5f}{gl2:>10.5f}"
              f"{mse+gl1:>11.5f}")
    print("\n  判读:")
    for col, nm in [(1, 'MSE'), (2, 'L1'), (3, '梯度L1'), (4, '梯度L2')]:
        v = {r[0]: r[col] for r in rows}
        good = v['真值 g_gt (好预测)']
        mean = v['平均骨架 mean(g_gt)']
        blank = v['空白 z_bg']
        ok = good < mean and good < blank
        print(f"    {nm:<8} 好={good:.5f}  平均骨架={mean:.5f}  空白={blank:.5f}  "
              f"-> {'✓ 排序正确' if ok else '✗ 退化吸引子还在'}")

    # 组合损失
    print("\n  组合损失 MSE + w*梯度L1 的排序 (w 扫描):")
    v = {r[0]: (r[1], r[3]) for r in rows}
    print(f"    {'w':>6}{'好':>12}{'平均骨架':>12}{'空白':>12}{'排序':>8}")
    for w in (0, 0.5, 1, 2, 5, 10):
        g = v['真值 g_gt (好预测)'][0] + w * v['真值 g_gt (好预测)'][1]
        m = v['平均骨架 mean(g_gt)'][0] + w * v['平均骨架 mean(g_gt)'][1]
        b = v['空白 z_bg'][0] + w * v['空白 z_bg'][1]
        s = v['g_std (无形变)'][0] + w * v['g_std (无形变)'][1]
        ok = '✓' if (g < m and g < b and g < s) else '✗'
        print(f"    {w:>6}{g:>12.5f}{m:>12.5f}{b:>12.5f}{ok:>8}   (g_std={s:.5f})")


if __name__ == "__main__":
    main()
