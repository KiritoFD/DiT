#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_calli_vae_trend.py — Calli-VAE 多 ckpt 趋势评估 (回答"还需不需要跑完")

对同一批抽样图, 逐个 ckpt 跑 `eval_calli_vae.eval_one`, 输出重建质量 + latent 诊断随 step 的走向。

用法 (4090, 需在仓库根目录):
  PYTHONPATH=. N=48 python tools/vae/eval_calli_vae_trend.py \
      --root experiments/calli_vae_dino_stdconv_kl1e6 \
      --steps 2500,5000,7500,10000,12500 \
      --csv /root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv \
      --data-root /root/Workspace/xy/UNIFIED_RAW \
      --dino-ckpt data/pretrained/pretrained_models/dinov2_vits14_pretrain.safetensors

2026-10-09 实测结论 (kl1e6 版, 48 图):
  像素/结构单调改善但强边际递减 (PSNR 32.87→34.42dB, 末段 2.5k 步仅 +0.13dB);
  DINO 感知指标 7.5k 触底后微退 (struct 0.523→0.554); latent 诊断 (std≈0.40,
  有效秩, 往返 cos≈0.72) 从 2.5k 起完全平台 -> 下游可用性早在 7.5k 即达标。
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_calli_vae import eval_one, load_dino, load_images  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="ckpt 根目录 (含 calli_vae_step_XXXXX/)")
    ap.add_argument("--steps", default="2500,5000,7500,10000,12500")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dino-ckpt", required=True)
    ap.add_argument("--stride", type=int, default=8000)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    print(f"[dev] {dev}", flush=True)
    imgs = load_images(a.csv, a.data_root, a.stride, a.n)
    dino = load_dino(a.dino_ckpt, dev)

    rows = []
    for s in [int(x) for x in a.steps.split(",")]:
        p = os.path.join(a.root, f"calli_vae_step_{s:05d}")
        if not os.path.isdir(p):
            print(f"[skip] {p} 不存在", flush=True)
            continue
        out, best = eval_one(f"step{s}", p, imgs, dev, a.batch, dino, None)
        rows.append((s, out))

    hdr = ["L1", "MSE", "PSNR", "SSIM", "ink_IoU", "ink_SSIM", "hf_ratio",
           "dino_struct_mse", "dino_style_mse", "dino_cos",
           "latent_std_mean", "eff_rank_90", "roundtrip_cos"]
    print("\n" + "=" * 110)
    print("step  | " + " | ".join(hdr))
    for s, o in rows:
        print(f"{s:5d} | " + " | ".join(f"{o[h]:.4f}" for h in hdr))


if __name__ == "__main__":
    main()
