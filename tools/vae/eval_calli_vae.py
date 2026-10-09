#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_calli_vae.py — Calli-VAE 重建质量 + latent 表征质量一体评估器 (2026-10-09)

设计要点 (为什么不能用旧 eval_vae.py):
  1. 旧脚本硬编码 `decode(z/sf)` —— 对约定被训反的 Calli 权重是错的; 本脚本**先实测探针**
     四种解码输入 (sample / sample/sf / mode / mode/sf) 的 L1, 自动选正确约定。
  2. 新增书法专门指标: 墨迹 IoU / 墨迹区 SSIM / 高频能量比 (飞白-枯笔保留度)。
  3. 新增 latent 表征质量诊断: 后验 std、有效秩 (SVD 90% 能量)、往返自洽性、
     以及 DINO patch 特征一致性 (即训练目标本身在 held-out 上的实测值)。
  4. 支持 --baseline 同协议对照 (如 sd-vae-ft-ema)。

用法 (4090):
  python tools/vae/eval_calli_vae.py \
    --vae experiments/calli_vae_dino_stdconv_kl1e6/calli_vae_step_12500 \
    --baseline data/pretrained/pretrained_models/sd-vae-ft-ema \
    --csv /root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv \
    --data-root /root/Workspace/xy/UNIFIED_RAW \
    --stride 3000 --n 128 --device cuda --batch 8
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))

from diffusers import AutoencoderKL  # noqa: E402
from safetensors import safe_open  # noqa: E402
from transformers import Dinov2Config, Dinov2Model  # noqa: E402


# ─────────────────────────── 数据 ───────────────────────────
def load_images(csv_path, data_root, stride, n, size=256):
    items = []
    with open(csv_path, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i % stride == 0:
                rel = row.get("image_path") or row.get("path") or ""
                if rel:
                    items.append(rel)
            if len(items) >= n:
                break
    out = []
    for rel in items:
        p = rel if os.path.isabs(rel) else os.path.join(data_root, rel)
        try:
            img = Image.open(p).convert("RGB").resize((size, size))
        except Exception:
            continue
        a = np.asarray(img, np.float32) / 127.5 - 1.0
        out.append(torch.from_numpy(a).permute(2, 0, 1))
    print(f"[data] 载入 {len(out)} 张 (stride={stride}, csv={os.path.basename(csv_path)})", flush=True)
    return torch.stack(out)


# ─────────────────────────── 指标 ───────────────────────────
def _gauss_win(dev, ws=11, sigma=1.5):
    g = torch.arange(ws, dtype=torch.float32, device=dev) - ws // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g.reshape(1, 1, ws, 1) @ g.reshape(1, 1, 1, ws))


def ssim(x, y, win, dr=2.0):
    if x.shape[1] == 3:
        return float(np.mean([ssim(x[:, i:i + 1], y[:, i:i + 1], win, dr) for i in range(3)]))
    C1, C2 = (0.01 * dr) ** 2, (0.03 * dr) ** 2
    mx, my = F.conv2d(x, win, padding=5), F.conv2d(y, win, padding=5)
    sxx = F.conv2d(x * x, win, padding=5) - mx * mx
    syy = F.conv2d(y * y, win, padding=5) - my * my
    sxy = F.conv2d(x * y, win, padding=5) - mx * my
    m = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
    return m.mean().item()


@torch.no_grad()
def recon_metrics(x, dec, win):
    """x/dec: [-1,1] (B,3,H,W)。返回像素 + 墨迹 + 高频三类指标。"""
    l1 = F.l1_loss(dec, x).item()
    mse = F.mse_loss(dec, x).item()
    psnr = 10 * np.log10(4.0 / max(mse, 1e-12))          # data_range=2
    ss = float(np.mean([ssim(x[i:i + 1], dec[i:i + 1], win) for i in range(x.shape[0])]))
    # 墨迹 (灰度 < 0.5 视为墨)
    gx = x.mean(1, keepdim=True) * 0.5 + 0.5              # [-1,1] -> [0,1]
    gd = dec.mean(1, keepdim=True) * 0.5 + 0.5
    mx_, md_ = (gx < 0.5).float(), (gd < 0.5).float()
    inter = (mx_ * md_).sum()
    ink_iou = (inter / ((mx_ + md_ - mx_ * md_).sum() + 1e-6)).item()
    ink_ssim = float(np.mean([
        ssim(gx[i:i + 1] * mx_[i:i + 1] * 2 - 1, gd[i:i + 1] * md_[i:i + 1] * 2 - 1, win)
        for i in range(x.shape[0])]))
    # 高频能量比 (Sobel 梯度均值比, 反映飞白/枯笔保留)
    k = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]], device=x.device).view(1, 1, 3, 3)
    def hf(t):
        return F.conv2d(t, k, padding=1).abs().mean().item()
    hf_gt, hf_rc = hf(gx), hf(gd)
    return dict(L1=l1, MSE=mse, PSNR=psnr, SSIM=ss, ink_IoU=ink_iou, ink_SSIM=ink_ssim,
                hf_gt=hf_gt, hf_recon=hf_rc, hf_ratio=hf_rc / max(hf_gt, 1e-9))


# ─────────────────────────── DINO (可选) ───────────────────────────
def load_dino(ckpt, device):
    with safe_open(ckpt, framework="pt") as f:
        keys = list(f.keys())
        hidden = f.get_tensor("encoder.layer.0.attention.attention.query.weight").shape[0]
    depth = max(int(k.split(".")[2]) for k in keys if k.startswith("encoder.layer.")) + 1
    cfg = Dinov2Config(image_size=224, patch_size=14, num_channels=3, hidden_size=hidden,
                       num_hidden_layers=depth,
                       num_attention_heads={384: 6, 768: 12, 1024: 16, 1536: 24}.get(hidden, 6),
                       intermediate_size=4 * hidden, hidden_act="gelu", layer_norm_eps=1e-6,
                       layer_scale_init_value=1.0,
                       num_register_tokens=1 if any("register_tokens" in k for k in keys) else 0)
    try:
        m = Dinov2Model(cfg, attn_implementation="sdpa")
    except Exception:
        m = Dinov2Model(cfg)
    new_names = any(".attention.q_proj." in k for k in m.state_dict())
    sd = {}
    with safe_open(ckpt, framework="pt") as f:
        for k in f.keys():
            nk = k
            if new_names:
                nk = (nk.replace("attention.attention.query", "attention.q_proj")
                        .replace("attention.attention.key", "attention.k_proj")
                        .replace("attention.attention.value", "attention.v_proj")
                        .replace("attention.output.dense", "attention.o_proj"))
            sd[nk] = f.get_tensor(k)
    pe = sd.get("embeddings.position_embeddings")
    tp = (224 // 14) ** 2
    if pe is not None and pe.shape[1] != 1 + tp:
        cls, pat = pe[:, :1], pe[:, 1:]
        h = w = int(round(pat.shape[1] ** 0.5))
        pat = pat.reshape(1, h, w, -1).permute(0, 3, 1, 2)
        pat = F.interpolate(pat, size=(16, 16), mode="bicubic", align_corners=False)
        sd["embeddings.position_embeddings"] = torch.cat([cls, pat.permute(0, 2, 3, 1).reshape(1, tp, -1)], 1)
    miss, _ = m.load_state_dict(sd, strict=False)
    if miss:
        raise RuntimeError(f"DINO 权重缺失 (命名不匹配!): {miss[:3]}")
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad = False
    return m


DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def dino_feats(dino, x, dev):
    x01 = (x + 1) / 2
    x224 = F.interpolate(x01, size=(224, 224), mode="bicubic", align_corners=False)
    x224 = (x224 - DINO_MEAN.to(dev)) / DINO_STD.to(dev)
    return dino(x224).last_hidden_state[:, 1:, :]        # (B,256,384)


# ─────────────────────────── 主流程 ───────────────────────────
@torch.no_grad()
def probe_convention(vae, x, sf, dev):
    """实测四种解码输入的 L1, 判定正确约定 (自检即契约)。"""
    post = vae.encode(x).latent_dist
    s, m = post.sample(), post.mode()
    res = {}
    for name, z in (("sample", s), ("sample/sf", s / sf), ("mode", m), ("mode/sf", m / sf)):
        res[name] = F.l1_loss(vae.decode(z).sample, x).item()
    best = min(res, key=res.get)
    return best, res, float(post.std.mean().item())


@torch.no_grad()
def eval_one(name, vae_path, images, dev, batch, dino=None, dino_ckpt=None):
    vae = AutoencoderKL.from_pretrained(vae_path).to(dev).eval()
    sf = vae.config.scaling_factor
    best, probe, pstd = probe_convention(vae, images[:8].to(dev), sf, dev)
    print(f"\n=== [{name}] {vae_path}")
    print(f"  [探针] 四种解码 L1: " + "  ".join(f"{k}={v:.4f}" for k, v in probe.items())
          + f"   -> 正确约定 = decode({best})")
    print(f"  [latent] posterior.std mean = {pstd:.4f}")

    win = _gauss_win(dev)
    acc = {}
    lat_mu, lat_sd, feats_gt, feats_rc, z_all, z_re = [], [], [], [], [], []
    for i in range(0, len(images), batch):
        x = images[i:i + batch].to(dev)
        post = vae.encode(x).latent_dist
        z = post.sample()
        dec = vae.decode(z if best == "sample" else
                         (z / sf if best == "sample/sf" else
                          (post.mode() if best == "mode" else post.mode() / sf))).sample
        dec = dec.clamp(-1, 1)
        m = recon_metrics(x, dec, win)
        for k, v in m.items():
            acc[k] = acc.get(k, 0.0) + v * x.shape[0]
        lat_mu.append(post.mean.flatten(1))
        lat_sd.append(post.std.flatten(1))
        z_all.append(z.flatten(1))
        z_re.append(vae.encode(dec).latent_dist.mean.flatten(1))
        if dino is not None:
            feats_gt.append(dino_feats(dino, x, dev).cpu())
            feats_rc.append(dino_feats(dino, dec, dev).cpu())
    n = len(images)
    out = {k: v / n for k, v in acc.items()}

    Z = torch.cat(z_all)                                  # (N, 4*32*32)
    Zc = Z - Z.mean(0, keepdim=True)
    sv = torch.linalg.svdvals(Zc)
    energy = torch.cumsum(sv ** 2, 0) / (sv ** 2).sum()
    eff_rank = int((energy < 0.90).sum().item()) + 1
    mu = torch.cat(lat_mu)
    sd = torch.cat(lat_sd)
    z2 = torch.cat(z_re)
    rt_cos = F.cosine_similarity(Z, z2, dim=1).mean().item()
    out.update(latent_std_mean=sd.mean().item(), latent_std_p95=sd.flatten().kthvalue(
        int(0.95 * sd.numel())).values.item(), mu_abs_mean=mu.abs().mean().item(),
        latent_dim=Z.shape[1], eff_rank_90=eff_rank, roundtrip_cos=rt_cos)
    if dino is not None:
        fg, fr = torch.cat(feats_gt), torch.cat(feats_rc)
        out["dino_struct_mse"] = F.mse_loss(fr, fg).item()
        out["dino_style_mse"] = F.mse_loss(fr.std(1), fg.std(1)).item()
        out["dino_cos"] = F.cosine_similarity(fr, fg, dim=-1).mean().item()

    print(f"  重建: L1={out['L1']:.4f} MSE={out['MSE']:.4f} PSNR={out['PSNR']:.2f}dB "
          f"SSIM={out['SSIM']:.4f}")
    print(f"  书法: ink_IoU={out['ink_IoU']:.4f} ink_SSIM={out['ink_SSIM']:.4f} "
          f"高频比={out['hf_ratio']:.3f} (recon/GT)")
    print(f"  latent: std={out['latent_std_mean']:.3f}(p95 {out['latent_std_p95']:.3f}) "
          f"|mu|={out['mu_abs_mean']:.3f} dim={out['latent_dim']} 有效秩90%={out['eff_rank_90']} "
          f"往返cos={out['roundtrip_cos']:.4f}")
    if dino is not None:
        print(f"  DINO(held-out): struct={out['dino_struct_mse']:.4f} "
              f"style={out['dino_style_mse']:.4f} patch_cos={out['dino_cos']:.4f}")
    del vae
    torch.cuda.empty_cache()
    return out, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--stride", type=int, default=3000)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino-ckpt", default=None)
    ap.add_argument("--skip-dino", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    images = load_images(a.csv, a.data_root, a.stride, a.n)
    dino = None
    if a.dino_ckpt and not a.skip_dino:
        dino = load_dino(a.dino_ckpt, dev)
        print("[dino] 权重校验通过 (无 missing key)", flush=True)
    res = {}
    res["calli"] = eval_one("Calli-VAE", a.vae, images, dev, a.batch, dino, a.dino_ckpt)[0]
    if a.baseline and os.path.exists(a.baseline):
        res["baseline"] = eval_one("baseline", a.baseline, images, dev, a.batch, dino, a.dino_ckpt)[0]
    print("\n" + "=" * 78)
    keys = [k for k in res["calli"]]
    for k in keys:
        line = f"{k:>16} | " + " | ".join(
            f"{res[n][k]:.4f}" if isinstance(res[n].get(k), float) else str(res[n].get(k))
            for n in res)
        print(line)
    print(f"{'模型':>16} | " + " | ".join(res))


if __name__ == "__main__":
    main()
