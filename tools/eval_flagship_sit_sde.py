#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_flagship_sit_sde.py — 旗舰模型 (DiT-B/2 @ 40k) 在黄金测试集上的 SiT SDE 随机纠偏全量评测

评测协议与官方基准完全同频 (eval200_fixed, N=187 严格未见真迹):
  - VAE: sd-vae-ft-ema
  - LPIPS: VGG (官方口径)
  - SSIM: 11x11 高斯窗口 (官方口径)
  - Skel-IoU: 骨架中心线交并比 (官方口径)
对比模式:
  1. Baseline Euler ODE (gamma=0.0, 50步, 官方历史读数 0.6039)
  2. Heun-RK2 二阶 ODE (gamma=0.0, 50步)
  3. SiT SDE gamma=0.10
  4. SiT SDE gamma=0.20
  5. SiT SDE gamma=0.35 (黄金平衡推荐)
  6. SiT SDE gamma=0.50 (极致自纠偏)
"""

import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch as th
from PIL import Image
from scipy.ndimage import correlate1d
from skimage.morphology import skeletonize

_root = "/home/ds/Workspace/DiT"
sys.path.insert(0, _root)

from diffusers.models import AutoencoderKL
import lpips
from src.model import DiT_2Cond_models


def _g(img, k1d):
    return correlate1d(correlate1d(img, k1d, axis=0, mode="reflect"), k1d, axis=1, mode="reflect")


def ssim_np(pred, gt, win=11, sigma=1.5, dr=1.0):
    r = win // 2
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-(x**2) / (2 * sigma**2))
    k = k / k.sum()
    c1 = (0.01 * dr) ** 2
    c2 = (0.03 * dr) ** 2
    out = []
    for ch in range(pred.shape[2]):
        xx = pred[:, :, ch].astype(np.float64)
        yy = gt[:, :, ch].astype(np.float64)
        ux, uy = _g(xx, k), _g(yy, k)
        ux2, uy2, uxy = ux**2, uy**2, ux * uy
        sx2 = _g(xx * xx, k) - ux2
        sy2 = _g(yy * yy, k) - uy2
        sxy = _g(xx * yy, k) - uxy
        m = ((2 * uxy + c1) * (2 * sxy + c2)) / ((ux2 + uy2 + c1) * (sx2 + sy2 + c2))
        out.append(float(m.mean()))
    return float(np.mean(out))


def calc_skel_iou(pred_np, gt_np, thresh=0.5):
    b1 = pred_np.mean(axis=2) < thresh
    b2 = gt_np.mean(axis=2) < thresh
    if not b1.any() and not b2.any():
        return 1.0
    if not b1.any() or not b2.any():
        return 0.0
    s1, s2 = skeletonize(b1), skeletonize(b2)
    inter = float((s1 & s2).sum())
    union = float((s1 | s2).sum())
    return inter / union if union > 0 else 1.0


def sample_with_sit_sde(model, z_init, steps, y_callig, y_script, y_char, gamma=0.0, sampler="euler", device="cuda"):
    """执行含 SiT SDE 纠偏项的潜变量反向积分。"""
    z = z_init.clone()
    ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

    for i in range(steps):
        t_i = ts[i]
        t_next = ts[i + 1]
        dt = t_next - t_i  # 负步长
        h = -dt            # 正步长
        t_curr = t_i.expand(z.shape[0]) * 1000.0

        v1 = model(z, t_curr, y_callig=y_callig, y_script=y_script, y_char=y_char)
        if isinstance(v1, tuple):
            v1 = v1[0]
        v1 = v1[:, :4]

        if sampler == "heun" and i < steps - 1:
            z_euler = z + dt * v1
            t_next_curr = t_next.expand(z.shape[0]) * 1000.0
            v2 = model(z_euler, t_next_curr, y_callig=y_callig, y_script=y_script, y_char=y_char)
            if isinstance(v2, tuple):
                v2 = v2[0]
            v_eff = 0.5 * (v1 + v2[:, :4])
        else:
            v_eff = v1

        if gamma > 0.0:
            # SiT SDE 动力学校正项
            drift = v_eff * (1.0 + (gamma ** 2) * t_i)
            noise_scale = gamma * torch.sqrt(torch.clamp(2.0 * t_i * (1.0 - t_i) * h, min=0.0))
            z = z + dt * drift + noise_scale * torch.randn_like(z)
        else:
            z = z + dt * v_eff

    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/checkpoints/0040000.pt",
                        help="旗舰模型权重路径")
    parser.add_argument("--out-dir", type=str,
                        default="/home/ds/Workspace/DiT/experiments/flagship_sit_sde_results",
                        help="输出目录")
    parser.add_argument("--batch-size", type=int, default=16, help="评测批次大小")
    parser.add_argument("--steps", type=int, default=50, help="采样步数")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    CSV_STRICT = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
    SHARDS_DIR = "/home/ds/Workspace/moyi/data/top10_style23/shards_img"
    VAE_PATH = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
    ASSETS_DIR = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal"

    print("=" * 85)
    print("【启动旗舰模型 (Tier 3 B/2 @ 40k) SiT SDE 随机纠偏全量评测】")
    print(f"  模型权重: {args.ckpt}")
    print(f"  评测集:   {CSV_STRICT} (eval200_fixed 官方口径, N=187)")
    print(f"  批次大小: {args.batch_size} (极速批量模式)")
    print(f"  输出目录: {args.out_dir}")
    print("=" * 85)

    # 1. 载入 VAE 与 LPIPS VGG
    print("[1/4] 载入 VAE 与 LPIPS VGG 官方评测模型 ...", flush=True)
    vae = AutoencoderKL.from_pretrained(VAE_PATH).to(device).eval()
    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()

    # 2. 载入三表映射
    char_remap = {int(k): int(v) for k, v in json.load(open(f"{ASSETS_DIR}/char_remap.json", encoding="utf-8")).items()}
    callig_remap = {int(k): int(v) for k, v in json.load(open(f"{ASSETS_DIR}/callig_remap.json", encoding="utf-8")).items()}
    font_remap = {int(k): int(v) for k, v in json.load(open(f"{ASSETS_DIR}/font_remap.json", encoding="utf-8")).items()}

    # 3. 索引真迹 GT
    print("[2/4] 索引真迹 GT 潜变量 ...", flush=True)
    id_to_lat = {}
    for sf in sorted(glob.glob(f"{SHARDS_DIR}/shard_*.npz")):
        with np.load(sf) as d:
            lats = d["latents"]
            ids = d["img_ids"]
            for idx_i, iid in enumerate(ids):
                id_to_lat[str(iid)] = torch.from_numpy(lats[idx_i]).float()

    # 4. 构建模型并加载权重
    print("[3/4] 构建 DiT 骨干网络并加载权重 ...", flush=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg_args = ckpt["args"]

    model = DiT_2Cond_models[cfg_args.model](
        learn_sigma=False,
        norm_type=cfg_args.norm_type,
        mlp_type=cfg_args.mlp_type,
        qk_norm=cfg_args.qk_norm,
        rope=cfg_args.rope,
        condition_fusion=cfg_args.condition_fusion,
        cond_fusion_norm=cfg_args.cond_fusion_norm,
        num_calligraphers=cfg_args.num_calligraphers,
        num_characters=cfg_args.num_characters,
        num_script_classes=getattr(cfg_args, "num_script_classes", 3),
        use_script_cond=getattr(cfg_args, "use_script_cond", False),
        char_embed_dim=cfg_args.char_embed_dim,
        callig_embed_dim=cfg_args.callig_embed_dim,
        script_embed_dim=getattr(cfg_args, "script_embed_dim", None),
        glyph_inject_layers=cfg_args.glyph_inject_layers,
        cond_inject_at=getattr(cfg_args, 'cond_inject_at', ''),
        cond_inject_scale=getattr(cfg_args, 'cond_inject_scale', True),
    ).to(device)

    state = ckpt.get("ema", ckpt.get("model", ckpt.get("delta", ckpt)))
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    model.load_state_dict(state, strict=False)
    model.eval()

    rows = list(csv.DictReader(open(CSV_STRICT, encoding="utf-8")))
    print(f"  成功载入测试行数: {len(rows)} 条", flush=True)

    # 预加载与预对齐条件
    c_list = [callig_remap.get(int(r["calligrapher_id"]), 0) for r in rows]
    s_list = [font_remap.get(int(r["script_id"]), 0) for r in rows]
    ch_list = [char_remap.get(int(r["character_id"]), 0) for r in rows]
    img_ids = [str(r["img_id"]) for r in rows]

    # 预先解码真迹图像 GT 并缓存
    print("[3.5/4] 预解码真迹图像 GT ...", flush=True)
    gt_imgs = []
    with torch.no_grad():
        for b_start in range(0, len(rows), args.batch_size):
            b_end = min(b_start + args.batch_size, len(rows))
            b_lats = torch.stack([id_to_lat[img_ids[k]] for k in range(b_start, b_end)]).to(device)
            dec_gt = vae.decode(b_lats / 0.18215).sample
            gt_imgs.append(torch.clamp((dec_gt + 1.0) / 2.0, 0.0, 1.0).cpu())
    all_gt = torch.cat(gt_imgs, dim=0).permute(0, 2, 3, 1).numpy()

    # 固定全量初始噪声
    torch.manual_seed(0)
    all_init_noise = torch.randn(len(rows), 4, 32, 32, device=device)

    # 评测模式配置
    modes = [
        {"name": "Euler ODE Baseline (官方历史基线)", "sampler": "euler", "gamma": 0.0},
        {"name": "Heun-RK2 二阶 ODE", "sampler": "heun", "gamma": 0.0},
        {"name": "SiT SDE (gamma=0.10)", "sampler": "heun", "gamma": 0.10},
        {"name": "SiT SDE (gamma=0.20)", "sampler": "heun", "gamma": 0.20},
        {"name": "SiT SDE (gamma=0.35, 黄金推荐)", "sampler": "heun", "gamma": 0.35},
        {"name": "SiT SDE (gamma=0.50, 极致自纠偏)", "sampler": "heun", "gamma": 0.50},
    ]

    print("\n[4/4] 开始各模式批量评测 ...", flush=True)
    all_results = []

    for m in modes:
        m_name = m["name"]
        sampler = m["sampler"]
        gamma = m["gamma"]
        print(f"\n>>> 正在运行 [{m_name}] (N={len(rows)}) ...", flush=True)
        t0 = time.time()

        pred_imgs = []
        with torch.no_grad():
            for b_start in range(0, len(rows), args.batch_size):
                b_end = min(b_start + args.batch_size, len(rows))
                b_size = b_end - b_start

                y_cal = torch.tensor(c_list[b_start:b_end], device=device, dtype=torch.long)
                y_sc = torch.tensor(s_list[b_start:b_end], device=device, dtype=torch.long)
                y_ch = torch.tensor(ch_list[b_start:b_end], device=device, dtype=torch.long)
                z_init = all_init_noise[b_start:b_end]

                z_out = sample_with_sit_sde(
                    model, z_init, args.steps, y_cal, y_sc, y_ch,
                    gamma=gamma, sampler=sampler, device=device
                )

                dec = vae.decode(z_out / 0.18215).sample
                pred = torch.clamp((dec + 1.0) / 2.0, 0.0, 1.0)
                pred_imgs.append(pred)

        all_pred_t = torch.cat(pred_imgs, dim=0)  # (N, 3, 256, 256)
        all_pred_np = all_pred_t.permute(0, 2, 3, 1).cpu().numpy()

        # 批量评估各项指标
        ssims, mses, lpip_list, skel_list = [], [], [], []
        with torch.no_grad():
            for idx in range(len(rows)):
                p_np = all_pred_np[idx]
                g_np = all_gt[idx]
                ssims.append(ssim_np(p_np, g_np))
                mses.append(float(np.mean((p_np - g_np) ** 2)) * 4.0)
                skel_list.append(calc_skel_iou(p_np, g_np))

            # 批量 LPIPS
            for b_start in range(0, len(rows), args.batch_size):
                b_end = min(b_start + args.batch_size, len(rows))
                p_lp = all_pred_t[b_start:b_end].to(device) * 2.0 - 1.0
                g_lp = torch.from_numpy(all_gt[b_start:b_end]).permute(0, 3, 1, 2).to(device) * 2.0 - 1.0
                lp_vals = lpips_fn(p_lp, g_lp).squeeze().cpu().tolist()
                if isinstance(lp_vals, float):
                    lpip_list.append(lp_vals)
                else:
                    lpip_list.extend(lp_vals)

        dt = time.time() - t0
        s_mean = float(np.mean(ssims))
        s_med = float(np.median(ssims))
        lp_mean = float(np.mean(lpip_list))
        sk_mean = float(np.mean(skel_list))
        mse_mean = float(np.mean(mses))

        res = {
            "mode": m_name,
            "sampler": sampler,
            "gamma": gamma,
            "strict_ssim": round(s_mean, 4),
            "ssim_med": round(s_med, 4),
            "strict_lpips": round(lp_mean, 4),
            "skel_iou": round(sk_mean, 4),
            "mse": round(mse_mean, 4),
            "time_sec": round(dt, 1),
        }
        all_results.append(res)
        # 立即增量落盘 (防止进程中断丢失前序数据)
        with open(f"{args.out_dir}/flagship_sit_sde_results.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        with open(f"{args.out_dir}/flagship_sit_sde_results.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            w.writeheader()
            w.writerows(all_results)
        print(f"  ✓ [{m_name}] 耗时 {dt:.1f}s | "
              f"Strict SSIM: {s_mean:.4f} (med {s_med:.4f}) | "
              f"LPIPS: {lp_mean:.4f} | "
              f"Skel-IoU: {sk_mean:.4f} | "
              f"MSE: {mse_mean:.4f}", flush=True)

    # 打印全量对比总表
    print("\n" + "=" * 95)
    print("【旗舰模型 (Tier 3 B/2 @ 40k) SiT SDE 全模式对比评测总表】")
    print("=" * 95)
    print(f"{'评测模式':<35} | {'Strict SSIM':<12} | {'Strict LPIPS':<14} | {'Skel IoU':<10} | {'MSE':<8}")
    print("-" * 95)
    base_ssim = all_results[0]["strict_ssim"]
    base_lp = all_results[0]["strict_lpips"]
    base_skel = all_results[0]["skel_iou"]

    for r in all_results:
        d_ssim = r["strict_ssim"] - base_ssim
        d_lp = r["strict_lpips"] - base_lp
        d_skel = r["skel_iou"] - base_skel
        gain_str = f" (SSIM {d_ssim:+.4f}, LP {d_lp:+.4f}, Skel {d_skel:+.4f})" if r["gamma"] > 0 or r["sampler"] == "heun" else " (历史基准)"
        print(f"{r['mode']:<35} | {r['strict_ssim']:<12.4f} | {r['strict_lpips']:<14.4f} | {r['skel_iou']:<10.4f} | {r['mse']:<8.4f}{gain_str}")
    print("=" * 95)

    # 保存文件
    with open(f"{args.out_dir}/flagship_sit_sde_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    with open(f"{args.out_dir}/flagship_sit_sde_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"\n[完成] 评测数据已保存至: {args.out_dir}/flagship_sit_sde_results.csv")


if __name__ == "__main__":
    main()
