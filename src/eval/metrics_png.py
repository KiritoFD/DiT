# -*- coding: utf-8 -*-
"""
metrics_png.py — 从已落盘的 PNG 计算扩展视觉质量指标（含噪点/清晰度类），CPU。

新增指标（在 MSE/SSIM/skel_iou/LPIPS 基础上）：
  * psnr            : PSNR (dB), 需 GT, skimage
  * tv              : 平均总变差 (mean abs 梯度), 越大越"碎/噪"
  * lap_var         : Laplacian 方差 (清晰度), 越小越糊
  * hf_energy       : 高频能量占比 (FFT 高通), 越大噪点越多
  * saltpepper      : 中值滤波残差中的孤立像素占比 (椒盐噪点代理)
  * edge_clean      : 笔画内梯度平滑度 (边缘连续性代理)

用法:
  python src/eval/metrics_png.py --dir <dir> --tag ctrl --n 100 --out metrics.json
  约定: dir 内含 {tag}{i}.png 与 gt{i}.png。
"""
import os
import sys
import json
import argparse

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))


def _load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def _mse(pred, gt):
    return float(np.mean((pred - gt) ** 2)) * 4.0


def _psnr(pred, gt, data_range=1.0):
    m = np.mean((pred - gt) ** 2)
    if m <= 1e-12:
        return 99.0
    return float(10 * np.log10(data_range ** 2 / m))


def _ssim_sk(pred, gt):
    from skimage.metrics import structural_similarity as ssim
    return float(ssim(pred, gt, channel_axis=2, data_range=1.0))


def _tv(img):
    """平均总变差: mean(|dI/dx| + |dI/dy|) across channels."""
    dx = np.abs(np.diff(img, axis=1)).mean()
    dy = np.abs(np.diff(img, axis=0)).mean()
    return float((dx + dy) / 2)


def _lap_var(img):
    """Laplacian 方差: 越糊越低, 噪点越高."""
    from scipy.ndimage import laplace
    gray = img.mean(axis=2)
    return float(laplace(gray).var())


def _hf_energy(img):
    """高频能量占比: FFT 后高于 1/8 奈奎斯特的频带能量占比."""
    gray = img.mean(axis=2)
    F = np.fft.fft2(gray)
    Fs = np.fft.fftshift(F)
    energy = np.abs(Fs) ** 2
    H, W = energy.shape
    cy, cx = H // 2, W // 2
    # 中心半径 = 1/8 * min(H,W)
    R = max(1, min(H, W) // 16)
    yy, xx = np.ogrid[:H, :W]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= R ** 2
    low = energy[mask_low].sum()
    total = energy.sum()
    return float((total - low) / total if total > 0 else 0.0)


def _salt_pepper(img, med_size=3):
    """中值滤波残差: |img - median| > 0.15 的像素占比 (椒盐噪点代理)."""
    from scipy.ndimage import median_filter
    gray = img.mean(axis=2)
    med = median_filter(gray, size=med_size)
    res = np.abs(gray - med)
    return float((res > 0.15).mean())


def _edge_clean(img, thresh=0.2):
    """笔画边缘平滑度: 梯度幅值>阈值 的像素的梯度一致性 (1 - 梯度方向标准差归一化)."""
    from scipy.ndimage import sobel
    gray = img.mean(axis=2)
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    edge = mag > thresh
    if edge.sum() < 10:
        return 1.0
    # 方向熵的简化代理: 梯度幅值在边缘上的离散度 (越小越干净)
    em = mag[edge]
    return float(em.std() / (em.mean() + 1e-6))


def _bg_uniformity(img):
    """背景均匀度: 灰度>0.9 (纸面) 像素的标准差, 越小背景越干净 (噪点/水渍越少)."""
    gray = img.mean(axis=2)
    bg = gray > 0.9
    if bg.sum() <= 10:
        return 0.0
    return float(gray[bg].std())


def _ink_purity(img):
    """墨色纯度 (Otsu 二分类类内方差): 用简单全局阈值把图分为墨/纸两类,
    两类的各自方差加权平均越低 = 墨色越纯(无杂斑/半灰噪). 返回 [0,1] 反向 (1=最纯)."""
    gray = img.mean(axis=2)
    t = 0.5
    # 简单自适应: Otsu 近似
    hist, _be = np.histogram(gray, bins=64, range=(0, 1))
    be = (_be[:-1] + _be[1:]) / 2
    total = hist.sum()
    if total == 0:
        return 1.0
    w0 = hist.cumsum()
    w1 = total - w0
    mu0 = (be * hist).cumsum() / np.maximum(w0, 1)
    mu1 = (total * (be * hist).sum() - (be * hist).cumsum()) / np.maximum(w1, 1)
    between = w0 * w1 * (mu0 - mu1) ** 2
    t_idx = int(np.argmax(between))
    t = be[t_idx]
    mask_ink = gray <= t
    mask_paper = gray > t
    if mask_ink.sum() <= 10 or mask_paper.sum() <= 10:
        return 0.0
    var_ink = gray[mask_ink].var()
    var_paper = gray[mask_paper].var()
    # 类内方差越小越好; 归一化到 [0,1], 1=纯墨纯纸
    pur = 1.0 - min(1.0, (var_ink + var_paper) / 0.05)
    return float(pur)


def _ringing(img, gt, grad_thresh=0.25):
    """振铃/过冲 (ringing & overshoot): 沿边缘法线方向检查预测是否过度过冲。
    简化代理: |pred - gt| 在边缘邻域 (梯度>阈值±2px 内) 的平均绝对差,
    归一化到 [0,1] 反向 (1=无振铃)."""
    from scipy.ndimage import sobel
    g = img.mean(axis=2)
    gw = gt.mean(axis=2)
    gx = sobel(gw, axis=1)
    gy = sobel(gw, axis=0)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    # 边缘掩码膨胀 ±2px
    from scipy.ndimage import binary_dilation, generate_binary_structure
    edge = mag > grad_thresh
    if edge.sum() < 10:
        return 1.0
    edge_d = binary_dilation(edge, structure=generate_binary_structure(2, 1), iterations=2)
    err = np.abs(g - gw)
    ring_mae = float(err[edge_d].mean())
    # 归一化: mae 0.05 视为完全干净 (1.0), 0.3 视为严重振铃 (0.0)
    return float(np.clip(1.0 - ring_mae / 0.3, 0.0, 1.0))


def compute_one(pred, gt):
    m = {}
    m["mse"] = _mse(pred, gt)
    m["psnr"] = _psnr(pred, gt)
    m["ssim"] = _ssim_sk(pred, gt)
    m["tv"] = _tv(pred)
    m["lap_var"] = _lap_var(pred)
    m["hf_energy"] = _hf_energy(pred)
    m["saltpepper"] = _salt_pepper(pred)
    m["edge_clean"] = _edge_clean(pred)
    m["bg_uniformity"] = _bg_uniformity(pred)
    m["ink_purity"] = _ink_purity(pred)
    m["ringing"] = _ringing(pred, gt)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tag", default="ctrl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="")
    ap.add_argument("--use-lpips", action="store_true", default=True)
    args = ap.parse_args()

    keys = ["mse", "psnr", "ssim", "tv", "lap_var", "hf_energy", "saltpepper", "edge_clean",
            "bg_uniformity", "ink_purity", "ringing"]
    agg = {k: [] for k in keys}
    lpips_ = []
    cnt = 0
    for i in range(args.n):
        p = os.path.join(args.dir, f"{args.tag}{i}.png")
        g = os.path.join(args.dir, f"gt{i}.png")
        if not (os.path.exists(p) and os.path.exists(g)):
            continue
        pred = _load(p)
        gt = _load(g)
        r = compute_one(pred, gt)
        for k in keys:
            agg[k].append(r[k])
        cnt += 1
    if cnt == 0:
        print(f"NO images found in {args.dir} (tag={args.tag})")
        sys.exit(1)

    def _stat(a):
        a = np.array(a)
        return {"mean": float(a.mean()), "std": float(a.std()),
                "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75))}

    result = {"n": cnt, "tag": args.tag}
    for k in keys:
        result[k] = _stat(agg[k])

    # LPIPS (需 GT, 与 compute_metrics 同约定)
    try:
        import torch
        from src.eval.inference import _get_lpips
        fn = _get_lpips()
        if fn is not None:
            vals = []
            for i in range(cnt):
                p = os.path.join(args.dir, f"{args.tag}{i}.png")
                g = os.path.join(args.dir, f"gt{i}.png")
                pp = torch.from_numpy(_load(p).transpose(2, 0, 1)[None] * 2 - 1)
                gg = torch.from_numpy(_load(g).transpose(2, 0, 1)[None] * 2 - 1)
                with torch.no_grad():
                    vals.append(float(fn(pp, gg).mean().item()))
            result["lpips"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    except Exception as e:
        print(f"[warn] lpips failed: {e}")

    out_path = args.out or os.path.join(args.dir, "metrics_ext.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[metrics] n={cnt} tag={args.tag}")
    for k in keys:
        s = result[k]
        print(f"  {k:>12}: mean={s['mean']:.4f}  std={s['std']:.4f}  "
              f"q25={s['q25']:.4f} q75={s['q75']:.4f}")
    if "lpips" in result:
        print(f"  {'lpips':>12}: mean={result['lpips']['mean']:.4f}")
    print(f"[metrics] -> {out_path}")


if __name__ == "__main__":
    main()