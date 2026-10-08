#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_sit_sde_sweep.py — SiT SDE 随机纠偏采样器参数扫描与自纠错能力评估

测试核心假说:
  在确定性流匹配 (Flow Matching Heun/Euler ODE) 中，由于缺乏随机探索与自纠错机制，
  前序步骤的微小误差会单向累积，导致书法飞白发糊、笔画边缘晕染。
  引入 SiT SDE 随机纠偏采样器 (带布朗运动扩散项与得分兰芝文纠偏):
    X_{t-h} = X_t - h * [ v + gamma^2 * (X_t - x_0) ] + gamma * sqrt(2*t*(1-t)*h) * xi
  是否能在 gamma in [0.05, 0.20] 区间内提升真实质感与感知锐度 (LPIPS/SSIM/Skel-IoU)？
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch as th

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from src.model import DiT_2Cond_models
from src.loss import create_flow_matching
from src.eval.inference import _mse, _ssim, load_eval_vae, sample_latents
from skimage.morphology import skeletonize


def compute_skel_iou(pred_img_np, gt_img_np):
    """计算生成图骨架与真迹骨架的 IoU。"""
    pred_ink = pred_img_np.mean(axis=-1) < 0.5
    gt_ink = gt_img_np.mean(axis=-1) < 0.5
    if not pred_ink.any() or not gt_ink.any():
        return 0.0
    pred_skel = skeletonize(pred_ink)
    gt_skel = skeletonize(gt_ink)
    inter = np.logical_and(pred_skel, gt_skel).sum()
    union = np.logical_or(pred_skel, gt_skel).sum()
    return float(inter / max(union, 1))


def parse_args():
    parser = argparse.ArgumentParser(description="SiT SDE 随机纠偏采样器 Gamma 调参扫描")
    parser.add_argument("--ckpt", type=str,
                        default="/home/ds/Workspace/DiT/experiments/recurrent_skel_refine/checkpoints/v46_5050_40k.pt",
                        help="用于评估的模型权重路径")
    parser.add_argument("--cache", type=str,
                        default="/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt",
                        help="eval200 测试集缓存文件")
    parser.add_argument("--vae-path", type=str,
                        default="/home/ds/Workspace/moyi/models/sd-vae-ft-ema",
                        help="SD-VAE 权重目录")
    parser.add_argument("--gammas", type=str,
                        default="0.0,0.05,0.10,0.15,0.20,0.30",
                        help="要扫描的 gamma 列表 (逗号分隔)")
    parser.add_argument("--out-dir", type=str,
                        default="/home/ds/Workspace/DiT/experiments/sit_sde_sweep",
                        help="评估输出目录")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="评估样本数量 (默认 200)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="批次大小 (8 仅占 ~1.5G 显存，极度安全)")
    parser.add_argument("--steps", type=int, default=50,
                        help="采样步数")
    parser.add_argument("--sampler", type=str, default="heun", choices=["euler", "heun"],
                        help="基础求解器")
    parser.add_argument("--cfg", type=float, default=0.7,
                        help="CFG 强度")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = th.device(args.device if th.cuda.is_available() else "cpu")

    gamma_list = [float(g.strip()) for g in args.gammas.split(",") if g.strip()]

    print("=" * 85)
    print("【启动 SiT SDE 随机纠偏采样器调参评估流水线】")
    print(f"  模型权重: {args.ckpt}")
    print(f"  测试样本: {args.num_samples} (来自 {args.cache})")
    print(f"  扫描 Gammas: {gamma_list}")
    print(f"  求解器: {args.sampler} (步数: {args.steps}, CFG: {args.cfg})")
    print(f"  批次大小: {args.batch_size} (低显存模式，安全并发)")
    print(f"  输出目录: {args.out_dir}")
    print("=" * 85)

    # 1. 载入测试数据缓存
    print("\n[1/4] 读取真实书法测试集缓存 ...", flush=True)
    cache = th.load(args.cache, map_location="cpu", weights_only=False)
    N = min(args.num_samples, cache["noise"].shape[0])

    fixed_noise = cache["noise"][:N]
    std_lats = cache["std_lats"][:N]
    gt_pngs = cache["gt_pngs"][:N]  # (N, 3, 256, 256), [0, 1]
    conds_raw = cache["conds"][:N]
    gt_np = gt_pngs.numpy().transpose(0, 2, 3, 1)

    # 2. 载入 VAE 与 LPIPS
    print("[2/4] 加载 SD-VAE 与 LPIPS 模型 ...", flush=True)
    vae = load_eval_vae(device, args.vae_path)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:
        print(f"  [warn] LPIPS 模块不可用: {e}")
        lpips_fn = None

    # 3. 构建模型并加载权重
    print("[3/4] 初始化 DiT-2Cond-S/2 ...", flush=True)
    ckpt = th.load(args.ckpt, map_location="cpu", weights_only=False)
    model = DiT_2Cond_models["DiT-2Cond-S/2"](
        num_calligraphers=23,
        callig_embed_dim=128,
        condition_fusion="factorized_cat",
        cond_fusion_norm="split",
        glyph_inject_mode="adaln",
        glyph_inject_layers=4,
        glyph_embedder_depth=2,
        skel_as_glyph_cond=True,
        no_char_cond=True,
        use_char_cond=False,
        use_glyph_cond=True,
        norm_type="rms",
        mlp_type="swiglu",
        qk_norm=True,
        rope=True,
        rope_theta=100.0,
        attn_impl="sdpa",
        use_checkpoint=False,
        learn_sigma=False,
    ).to(device).eval()

    sd = ckpt.get("ema") or ckpt.get("model") or ckpt
    clean_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(clean_sd, strict=False)

    # 4. 逐个 Gamma 扫描
    print("\n[4/4] 启动 Gamma 调参扫描循环 ...", flush=True)
    results = []

    for gamma in gamma_list:
        tag = "ODE Baseline" if gamma == 0.0 else f"SDE gamma={gamma:.2f}"
        print(f"\n>>> 正在运行 [{tag}] 采样评估 (N={N}) ...", flush=True)
        t0 = time.time()

        # 构建含特定 gamma 的 FlowMatching
        fm = create_flow_matching(
            timestep_respacing=str(args.steps),
            sampler=args.sampler,
            t_sampler="logit_normal",
            shift=1.0,
            sde_gamma=gamma,
        )

        # 采样潜变量
        lat = sample_latents(
            model, fm, fixed_noise, conds_raw,
            cfg_scale=args.cfg, batch=args.batch_size, device=device,
            skel=std_lats, seed=0
        )

        # 解码像素
        dec_list = []
        with th.no_grad():
            for s in range(0, N, args.batch_size):
                e = min(s + args.batch_size, N)
                batch_lat = lat[s:e].to(device)
                with th.autocast("cuda", dtype=th.bfloat16):
                    dec = vae.decode(batch_lat / 0.18215).sample
                dec_list.append(dec.clamp(-1.0, 1.0))
        all_gen_t = th.cat(dec_list, dim=0).float()
        pred_01 = ((all_gen_t + 1.0) / 2.0).cpu().numpy().transpose(0, 2, 3, 1)

        # 计算指标
        ssims, lpips_list, skel_ious, mses = [], [], [], []
        for i in range(N):
            p_img = pred_01[i]
            g_img = gt_np[i]
            ssims.append(_ssim(p_img, g_img))
            mses.append(_mse(p_img, g_img))
            skel_ious.append(compute_skel_iou(p_img, g_img))
            if lpips_fn is not None:
                p_th = th.from_numpy(p_img.transpose(2, 0, 1)[None] * 2.0 - 1.0).float().to(device)
                g_th = th.from_numpy(g_img.transpose(2, 0, 1)[None] * 2.0 - 1.0).float().to(device)
                with th.no_grad():
                    lp = float(lpips_fn(p_th, g_th).mean().item())
                lpips_list.append(lp)

        ssim_mean = float(np.mean(ssims))
        ssim_med = float(np.median(ssims))
        lpips_mean = float(np.mean(lpips_list)) if lpips_list else 0.0
        skel_mean = float(np.mean(skel_ious))
        mse_mean = float(np.mean(mses))
        dt = time.time() - t0

        res_entry = {
            "gamma": gamma,
            "tag": tag,
            "strict_ssim": round(ssim_mean, 4),
            "ssim_med": round(ssim_med, 4),
            "strict_lpips": round(lpips_mean, 4),
            "skel_iou": round(skel_mean, 4),
            "mse": round(mse_mean, 4),
            "time_sec": round(dt, 1),
        }
        results.append(res_entry)

        print(f"  ✓ [{tag}] 耗时 {dt:.1f}s | "
              f"SSIM: {ssim_mean:.4f} (med {ssim_med:.4f}) | "
              f"LPIPS: {lpips_mean:.4f} | "
              f"Skel-IoU: {skel_mean:.4f} | "
              f"MSE: {mse_mean:.4f}", flush=True)

    # 5. 打印对比总结表
    print("\n" + "=" * 90)
    print("【SiT SDE 随机纠偏采样器参数扫描全量对比总表】")
    print("=" * 90)
    print(f"{'配置模式':<22} | {'Strict SSIM':<12} | {'Strict LPIPS':<14} | {'Skel IoU':<10} | {'MSE':<8}")
    print("-" * 90)
    for r in results:
        diff_ssim = r["strict_ssim"] - results[0]["strict_ssim"]
        diff_lp = r["strict_lpips"] - results[0]["strict_lpips"]
        diff_skel = r["skel_iou"] - results[0]["skel_iou"]
        note = f" (SSIM {diff_ssim:+.4f}, LP {diff_lp:+.4f}, Skel {diff_skel:+.4f})" if r["gamma"] > 0 else " (基准线)"
        print(f"{r['tag']:<22} | {r['strict_ssim']:<12.4f} | {r['strict_lpips']:<14.4f} | {r['skel_iou']:<10.4f} | {r['mse']:<8.4f}{note}")
    print("=" * 90)

    # 保存 JSON 与 CSV
    json_path = os.path.join(args.out_dir, "sit_sde_sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(args.out_dir, "sit_sde_sweep_results.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    print(f"\n[完成] 扫描数据已保存至: {csv_path} 与 {json_path}")


if __name__ == "__main__":
    main()
