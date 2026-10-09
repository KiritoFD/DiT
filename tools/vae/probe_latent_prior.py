#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_latent_prior.py — 判定"冻结 VAE 的潜空间能否直接给扩散模型当目标"
(回答: 冻结 VAE 训练 vs REPA-E 端到端)

三个轴 (都不是 PSNR):
  A. 先验对齐: 潜变量逐维统计 vs N(0,1)。若逐维 std 偏离 1 很多, 扩散的噪声日程/prior
     采样就与训练目标错位 (SD-VAE 用 sf=0.18215 做全局缩放, 我们沿用同一约定)。
  B. 先验采样可解码性 (最关键): z~N(0,I) 直接解码 -> 是否落在真实书法流形上?
     用 DINO 特征到真实图最近邻距离 + 墨迹覆盖率量化, 并落盘网格图供人眼判。
     若 N(0,1) 解码是噪声/垃圾, 说明潜空间与先验有域隙, 冻结训练会更难 (或需要重标定)。
  C. 潜空间敏感度: decode(z + eps*noise) 的损伤曲线。扩散预测永远不完美, 若轻微潜误差
     就导致画面崩坏, 则对 DiT 精度要求极高 (冻结训练风险大)。给出 Calli vs base 同协议对照。

用法 (4090):
  PYTHONPATH=. python tools/vae/probe_latent_prior.py \
      --vae experiments/calli_vae_dino_stdconv_kl1e6/calli_vae_step_12500 \
      --baseline data/pretrained/pretrained_models/sd-vae-ft-ema \
      --csv /root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv \
      --data-root /root/Workspace/xy/UNIFIED_RAW --n 48 --device cuda
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)

from eval_calli_vae import load_dino, load_images, dino_feats, ssim, _gauss_win  # noqa: E402
from diffusers import AutoencoderKL  # noqa: E402


def save_grid(imgs, path, nrow=8):
    """imgs: [-1,1] (N,C,H,W) -> 网格 PNG。"""
    x = ((imgs.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()
    N, C, H, W = x.shape
    ncol = int(np.ceil(N / nrow))
    canvas = torch.full((C, ncol * H, nrow * W), 255, dtype=torch.uint8)
    for i in range(N):
        r, c = i // nrow, i % nrow
        canvas[:, r * H:(r + 1) * H, c * W:(c + 1) * W] = x[i]
    arr = canvas.permute(1, 2, 0).numpy()
    Image.fromarray(arr).save(path)
    print(f"  [grid] {path}", flush=True)


@torch.no_grad()
def probe(name, vae_path, imgs, dev, dino, outdir, n_prior=16):
    vae = AutoencoderKL.from_pretrained(vae_path).to(dev).eval()
    sf = vae.config.scaling_factor
    x = imgs.to(dev)

    def enc_chunked(t, bs=8):
        mus, sds, zs = [], [], []
        for i in range(0, len(t), bs):
            p = vae.encode(t[i:i + bs]).latent_dist
            mus.append(p.mean)
            sds.append(p.std)
            zs.append(p.sample())
        return torch.cat(mus), torch.cat(sds), torch.cat(zs)

    def dec_chunked(zs, bs=8):
        outs = []
        for i in range(0, len(zs), bs):
            outs.append(vae.decode(zs[i:i + bs]).sample.clamp(-1, 1))
        return torch.cat(outs)

    mu, sd, z = enc_chunked(x)
    torch.cuda.empty_cache()

    # ---- A. 逐维统计 (在 z 空间; DiT 目标是 z*sf) ----
    m_flat = mu.flatten(1)                            # (N, D)
    d_mean = m_flat.mean(0)                            # 逐维均值
    d_var = m_flat.var(0) + sd.flatten(1).pow(2).mean(0)
    d_all = z.flatten(1)
    print(f"\n=== [{name}] {vae_path}")
    print(f"  [A 先验对齐] 逐维 std: 均值 {d_var.sqrt().mean():.3f} 中位 {d_var.sqrt().median():.3f} "
          f"p5 {d_var.sqrt().quantile(.05):.3f} p95 {d_var.sqrt().quantile(.95):.3f} | "
          f"逐维 |均值| 中位 {d_mean.abs().median():.3f}")
    sub = m_flat[:, ::8]                               # 512 维子采样, 估维间相关
    cm = torch.corrcoef(sub.T)
    off = ~torch.eye(sub.shape[1], dtype=torch.bool, device=dev)
    offcorr = cm[off].abs().mean().item()
    print(f"      整体: z.std={d_all.std():.3f} z.mean={d_all.mean():.3f} "
          f"| 维间相关(off-diag |r| 均值)={offcorr:.3f}")
    sfr = sf
    print(f"      DiT 目标 (z*sf) 逐维 std 中位 = {(d_var.sqrt() * sfr).median():.4f} "
          f"(扩散里噪声日程需按此标定)")

    # ---- B. 先验采样可解码性 ----
    real_feat = dino_feats(dino, x[:n_prior], dev)
    stats = {}
    for tag, zs in (("N(0,1)", torch.randn(n_prior, *z.shape[1:], device=dev)),
                    ("N(mu,std)", torch.randn_like(z[:n_prior]) * sd[:n_prior] + mu[:n_prior])):
        dec = dec_chunked(zs)
        g = dec.mean(1) * 0.5 + 0.5
        ink = (g < 0.5).float().mean().item()
        # DINO 特征到真实图的最近邻余弦 (值越高越像真迹)
        df = dino_feats(dino, dec, dev)
        cos = F.cosine_similarity(df[:, None], real_feat[None], dim=-1).max(dim=1).values
        stats[tag] = (ink, cos.mean().item())
        save_grid(dec, os.path.join(outdir, f"{name}_prior_{tag.replace('(','').replace(')','').replace(',','')}.png"))
        print(f"  [B 先验解码] {tag:10s}: 墨迹覆盖 {ink:.4f} | 到真实图最近邻 DINO cos "
              f"{cos.mean().item():.4f}")
    print(f"      (真实图墨迹覆盖参考 {((x[:n_prior].mean(1)*0.5+0.5)<0.5).float().mean():.4f})")

    # ---- C. 潜空间敏感度 ----
    win = _gauss_win(dev)
    base_std = d_var.sqrt().mean()
    print(f"  [C 敏感度] decode(z + eps*noise), noise~N(0, base_std²)  [base_std={base_std:.3f}]")
    for eps in (0.0, 0.25, 0.5, 1.0):
        zn = z + eps * base_std * torch.randn_like(z)
        dec = dec_chunked(zn)
        torch.cuda.empty_cache()
        l1 = F.l1_loss(dec, x).item()
        ss = float(np.mean([ssim(x[i:i + 1], dec[i:i + 1], win) for i in range(min(8, len(x)))]))
        gx = x[:8].mean(1, keepdim=True) * .5 + .5
        gd = dec[:8].mean(1, keepdim=True) * .5 + .5
        mx_, md_ = (gx < .5).float(), (gd < .5).float()
        iou = ((mx_ * md_).sum() / ((mx_ + md_ - mx_ * md_).sum() + 1e-6)).item()
        print(f"      eps={eps:<4} L1={l1:.4f} SSIM={ss:.4f} ink_IoU={iou:.4f}")
    del vae
    torch.cuda.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dino-ckpt", default=None)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--stride", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default="experiments/_probe_latent_prior")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    imgs = load_images(a.csv, a.data_root, a.stride, a.n)
    dino = load_dino(a.dino_ckpt, dev) if a.dino_ckpt else None
    probe("calli", a.vae, imgs, dev, dino, a.outdir)
    if a.baseline and os.path.exists(a.baseline):
        probe("sdvae", a.baseline, imgs, dev, dino, a.outdir)


if __name__ == "__main__":
    main()
