#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""accept_vae.py — VAE 验收器（配方文档 §4 的 I1–I6），对任何 VAE 目录出表格+结论

用法: python tools/vae/accept_vae.py --vae <dir> [--vae-b <dir2> 对照] --n 187 --device cuda:0
判据（配方文档 docs/vae_training_recipe.md §4）:
  I1 往返: L1<=0.005, PSNR>=30, 顶带高频 >=0.85x GT (目标 0.90)
  I2 raw 尺度: 报告 std/mean（存盘分布由 encode --scaling-factor auto 保证）
  I3 后验宽度: 记录 sigma/|z|
  I6 往返 LPIPS 不退化（防 GAN 用噪点换顶带份额）
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools", "vae"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from hf_losses import band_fraction  # noqa: E402


def band_fraction_np(a_uint8, r_cut=0.5):
    """numpy 版顶带份额（与 census 同口径），用于逐图统计。"""
    a = a_uint8.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    b = a / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
    return float(F[r > r_cut].sum() / max(F.sum(), 1e-9))


def radial_profile(a_uint8, nbins=6):
    """径向功率谱**剖面**（按半径分箱的归一化能量，从低频到高频）。

    ★ 为什么需要它: "顶带份额"只是一个数字, 噪声也能把它刷高;
      而"GAN 没做好"的典型指纹是**中高频整段能量不足**（图像发"塑料"）。
      逐带对比才能区分『真锐』与『用噪点换份额』。
    """
    a = a_uint8.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(axis=-1)
    b = a / 255.0
    F = np.fft.fftshift(np.abs(np.fft.fft2(b - b.mean())) ** 2)
    h, w = b.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2) / np.sqrt(2.0)
    tot = F.sum() + 1e-12
    out = []
    for i in range(nbins):
        lo, hi = i / nbins, (i + 1) / nbins
        out.append(float(F[(r >= lo) & (r < hi)].sum() / tot))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--vae-b", default="", help="对照 VAE（可选）")
    ap.add_argument("--n", type=int, default=187)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = torch.device(a.device)

    rows = [r for r in csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8"))
            if os.path.exists(r["image_path"])][:a.n]
    gt = np.stack([np.asarray(Image.open(r["image_path"]).convert("RGB").resize((256, 256), Image.BICUBIC)) for r in rows])
    xt = torch.from_numpy(gt).float().permute(0, 3, 1, 2).to(dev) / 127.5 - 1.0

    from diffusers.models import AutoencoderKL
    try:
        import lpips as _lp
        lp = _lp.LPIPS(net="vgg", verbose=False).to(dev).eval()
        for p in lp.parameters():
            p.requires_grad_(False)
    except Exception as e:
        lp = None
        print(f"  ⚠ LPIPS 不可用: {e}")

    res = {}
    for tag, path in (("new", a.vae), ("ctrl", a.vae_b)):
        if not path:
            continue
        if not os.path.isdir(path):
            print(f"  ✗ {tag} 缺: {path}"); continue
        vae = AutoencoderKL.from_pretrained(path).to(dev).eval()
        for p in vae.parameters():
            p.requires_grad_(False)
        l1s, ps, ss, hfs, lps, inks, sig = [], [], [], [], [], [], []
        rps, gps = [], []
        from skimage.metrics import structural_similarity as ssim_fn
        for i in range(0, len(xt), 8):
            xb = xt[i:i + 8]
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                post = vae.encode(xb).latent_dist
                sd = post.std.float() if hasattr(post, "std") else torch.exp(0.5 * post.logvar).float()
                z = post.sample()
                rec = vae.decode(z).sample.float()
            sig.append(float(sd.mean() / z.std()))
            l1s.append(float(F.l1_loss(rec, xb)))
            mse = float(F.mse_loss(rec, xb))
            ps.append(99.0 if mse <= 1e-12 else 10 * np.log10(4.0 / mse))
            r8 = rec.cpu(); x8 = xb.cpu()
            for k in range(r8.shape[0]):
                rn = ((r8[k].clamp(-1, 1) + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
                gn = ((x8[k].clamp(-1, 1) + 1) / 2 * 255).numpy().transpose(1, 2, 0).astype(np.uint8)
                ss.append(ssim_fn(rn, gn, channel_axis=2, data_range=255))
                hfs.append(float(band_fraction(torch.from_numpy(rn).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).item()))
                inks.append(float((rn.mean(axis=2) < 128).mean()))
                rps.append(radial_profile(rn))
                gps.append(radial_profile(gn))
            if lp is not None:
                lps.append(float(lp(rec, xb).mean().item()))
        gt_hf = []
        for k in range(len(gt)):
            gt_hf.append(float(band_fraction(torch.from_numpy(gt[k]).float().permute(2, 0, 1).unsqueeze(0) / 127.5 - 1).item()))
        r = {
            "vae": path, "n": len(gt),
            "L1": round(float(np.mean(l1s)), 5),
            "PSNR": round(float(np.mean(ps)), 2),
            "SSIM": round(float(np.mean(ss)), 4),
            "topband_x_gt": round(float(np.mean(hfs) / np.mean(gt_hf)), 3),
            "LPIPS": round(float(np.mean(lps)), 4) if lps else None,
            "ink_x_gt": round(float(np.mean(inks) / np.mean([ (np.asarray(Image.fromarray(g).convert("L")) < 128).mean() for g in gt])), 3),
            "posterior_rel_width": round(float(np.mean(sig)), 4),
            "radial_rec": [round(v, 4) for v in np.mean(np.array(rps), axis=0)],
            "radial_gt": [round(v, 4) for v in np.mean(np.array(gps), axis=0)],
        }
        r["radial_ratio"] = [round(a / b, 3) if b > 1e-9 else None
                             for a, b in zip(r["radial_rec"], r["radial_gt"])]
        # I2: raw 尺度（用于定 sf/shift）
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            post = vae.encode(xt[:64]).latent_dist
            zraw = post.sample().float().double()
        r["raw_std"] = round(float(zraw.std()), 4)
        r["raw_mean"] = round(float(zraw.mean()), 5)
        r["sf_fit"] = round(1.0 / float(zraw.std()), 6)
        res[tag] = r
        del vae
        torch.cuda.empty_cache()

    print(f"\n  {'项':<24}{'新 VAE':>14}{'对照':>14}   判据")
    keys = ["L1", "PSNR", "SSIM", "topband_x_gt", "LPIPS", "ink_x_gt", "posterior_rel_width", "raw_std", "raw_mean", "sf_fit"]
    for k in keys:
        vn = res.get("new", {}).get(k); vc = res.get("ctrl", {}).get(k)
        print(f"  {k:<24}{str(vn):>14}{str(vc):>14}")
    n = res.get("new", {})
    ok1 = (n.get("L1", 9) <= 0.005) and (n.get("PSNR", 0) >= 30) and (n.get("topband_x_gt", 0) >= 0.85)
    print(f"\n  I1 往返保真: L1 {n.get('L1')}<=0.005 | PSNR {n.get('PSNR')}>=30 | 顶带 {n.get('topband_x_gt')}>=0.85 "
          f"-> {'✓' if ok1 else '✗'}")
    if n.get("topband_x_gt"):
        print(f"  目标档(>=0.90): {'✓ 达标' if n['topband_x_gt'] >= 0.90 else '未达 0.90'}")
    if "ctrl" in res and n.get("LPIPS") and res["ctrl"].get("LPIPS"):
        print(f"  I6 往返 LPIPS: 新 {n['LPIPS']} vs 对照 {res['ctrl']['LPIPS']} -> "
              f"{'✓ 不退化' if n['LPIPS'] <= res['ctrl']['LPIPS'] + 0.005 else '⚠ 退化'}")
    print(f"  I2 拟合参数: sf={n.get('sf_fit')}  shift={n.get('raw_mean')}  (交给 encode --scaling-factor auto)")
    # ★ 径向谱剖面: 区分『真锐』与『用噪点换份额/发塑料』
    if n.get("radial_gt"):
        nb = len(n["radial_gt"])
        print(f"\n  径向谱剖面（低→高 {nb} 带, 重建/GT 的能量比）:")
        print("    GT  : " + "  ".join(f"{v:.4f}" for v in n["radial_gt"]))
        for tag in ("new", "ctrl"):
            if tag in res and res[tag].get("radial_rec"):
                lab = "新  " if tag == "new" else "对照"
                print(f"    {lab}: " + "  ".join(f"{v:.4f}" for v in res[tag]["radial_rec"]))
                print(f"    比值: " + "  ".join(f"{v:.3f}" for v in res[tag]["radial_ratio"]))
        print("    → 判读: 高频带比值 <<1 且顶带份额却很高 = 用噪点/伪影换份额; 全带比值≈1 = 频谱保真良好")
    if a.out:
        json.dump(res, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
