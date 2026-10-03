"""FLUX AE 探针: ①键结构 ②能否直接当 diffusers AutoencoderKL ③与 SD VAE 的**往返保真度对比**。

核心问题(本项目的病根候选): std 条件是 7px 宽的细黑线, 真迹是细笔画。
  SD VAE(4ch) 对"细高对比线"的编码能力有限 -> 条件在 encode 后就丢了形;
  FLUX AE(16ch) 通道多 4 倍, 若往返明显更好地保住细线, 那"换 latent 空间"就是根上的解法。
指标: PSNR / SSIM / ink-IoU(0.5 二值) / 墨量比 / frag(碎片率)。样本: GT 真迹 + std 条件图各 8 张。
用法: python tools/probe_flux_vae.py
"""
import os
import sys
from collections import Counter

import numpy as np
import torch as th
from PIL import Image
from safetensors import safe_open

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

FLUX_CKPT = "data/pretrained/flux_vae/ae.safetensors"
FLUX_CFG = dict(in_channels=3, out_channels=3, latent_channels=16,
                down_block_types=("DownEncoderBlock2D",) * 4,
                up_block_types=("UpDecoderBlock2D",) * 4,
                block_out_channels=(128, 256, 512, 512), layers_per_block=2,
                norm_num_groups=32, sample_size=1024,
                scaling_factor=0.3611, shift_factor=0.1159)

# ── ① 键结构 ──────────────────────────────────────────────────────────
print("=" * 70)
print("① FLUX AE 键结构")
with safe_open(FLUX_CKPT, framework="pt") as f:
    keys = list(f.keys())
    print(f"  总键数 = {len(keys)}")
    print("  顶层前缀:", dict(Counter(k.split('.')[0] for k in keys)))
    for want in ("encoder.conv_out.weight", "decoder.conv_in.weight",
                 "decoder.conv_out.weight", "encoder.conv_in.weight"):
        if want in keys:
            print(f"  {want:32s} -> {tuple(f.get_slice(want).get_shape())}")
    print("  含 quant_conv :", any("quant_conv" in k for k in keys))
    print("  含 post_quant_conv:", any("post_quant_conv" in k for k in keys))
    print("  前 10 键:", keys[:10])

# ── ② 建 VAE ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("② 构建 diffusers.AutoencoderKL (Flux 参数)")
from diffusers import AutoencoderKL  # noqa: E402

flux_vae = AutoencoderKL(**FLUX_CFG)
sd = {}
with safe_open(FLUX_CKPT, framework="pt") as f:
    for k in f.keys():
        sd[k] = f.get_tensor(k)
missing, unexpected = flux_vae.load_state_dict(sd, strict=False)
miss_w = [k for k in missing if k.endswith("weight") or k.endswith("bias")]
print(f"  missing={len(missing)} (其中权重类 {len(miss_w)}): {miss_w[:6]}")
print(f"  unexpected={len(unexpected)}: {unexpected[:6]}")
print("  ⚠ 若 missing 里出现 quant_conv/post_quant_conv -> 那两个是随机初始化, 往返会坏,"
      " 需要为 Flux AE 走 bypass 路径")

dev = th.device("cuda" if th.cuda.is_available() else "cpu")
flux_vae = flux_vae.to(dev).eval()

# SD VAE (对照组)
from src.eval.in_mem_eval import _get_vae  # noqa: E402
sd_vae = _get_vae(dev, "data/pretrained/pretrained_models/sd-vae-ft-ema").eval()
SD_SF = 0.18215


# ── 指标 ─────────────────────────────────────────────────────────────
def to_t(p):
    im = Image.open(p).convert("RGB").resize((256, 256))
    a = np.asarray(im, np.float32) / 255.0
    return th.from_numpy(a).permute(2, 0, 1)[None] * 2 - 1          # [-1,1]


def psnr(a, b):
    m = float(((a - b) ** 2).mean())
    return 99.0 if m <= 1e-9 else 10 * np.log10(1.0 / m)


def ssim_gray(a, b):
    a = a.mean(axis=2); b = b.mean(axis=2)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)))


def ink_iou(a, b, thr=0.5):
    ia, ib = (a.mean(axis=2) < thr), (b.mean(axis=2) < thr)
    u = (ia | ib).sum()
    return float((ia & ib).sum() / u) if u else 1.0


def frag(a, thr=0.5):
    from scipy.ndimage import label
    m = (a.mean(axis=2) < thr)
    _, n = label(m)
    return int(n)


@th.no_grad()
def roundtrip(x, vae, sf, shift=0.0):
    z = vae.encode(x.to(dev)).latent_dist.mode()
    z = (z - shift) * sf
    with th.autocast("cuda", dtype=th.bfloat16):
        dec = vae.decode((z / sf + shift).to(th.bfloat16)).sample
    return ((dec.clamp(-1, 1) + 1) / 2).float()


import csv  # noqa: E402
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8")))[:8]
samples = [("GT", os.path.join("data/top10_style23/imgs", os.path.basename(r["image_path"])))
           for r in rows]
samples += [("std条件", os.path.join("exp-std/data/std_fixed_eval200",
                                      os.path.basename(r["image_path"]))) for r in rows]

print("\n" + "=" * 70)
print("③ 往返保真度 (n=%d 张, 4ch SD VAE vs 16ch FLUX AE)" % (len(samples) // 2))
hdr = (f"{'类型':<8} {'VAE':<10} {'PSNR':>7} {'SSIM':>7} {'ink-IoU':>8} "
       f"{'墨量比':>8} {'frag_gt->rt':>12}")
print(hdr); print("-" * len(hdr))
agg = {}
for kind in ("GT", "std条件"):
    for name, vae, sf, sh in (("SD-4ch", sd_vae, SD_SF, 0.0),
                              ("FLUX-16ch", flux_vae, FLUX_CFG["scaling_factor"],
                               FLUX_CFG["shift_factor"])):
        P, S, I, M, F = [], [], [], [], []
        for k, p in samples:
            if k != kind or not os.path.exists(p):
                continue
            x = to_t(p)
            rt = roundtrip(x, vae, sf, sh)
            xa = ((x + 1) / 2)[0].permute(1, 2, 0).cpu().numpy()
            ra = rt[0].permute(1, 2, 0).cpu().numpy()
            P.append(psnr(xa, ra)); S.append(ssim_gray(xa, ra))
            I.append(ink_iou(xa, ra))
            M.append(float(ra.mean() / max(xa.mean(), 1e-6)))
            F.append(frag(ra) - frag(xa))
        if not P:
            continue
        agg[(kind, name)] = (np.mean(P), np.mean(S), np.mean(I), np.mean(M), np.mean(F))
        print(f"{kind:<8} {name:<10} {np.mean(P):>7.2f} {np.mean(S):>7.4f} "
              f"{np.mean(I):>8.4f} {np.mean(M):>8.3f} {np.mean(F):>12.2f}")

print("\n判读: 看 'std条件' 两行 —— ink-IoU 更高 / 墨量比更接近 1 / frag 增量更小 者, "
      "对细笔画更保真")
