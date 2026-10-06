#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""recurrent_skel_refine.py — 骨架自循环重采样提精 (Recurrent Skeleton Refinement)

实验假说与核心机制:
  1. Round 0: 初始输入僵硬的印刷宋体/标准骨架 g^(0)，模型输出书法图像 x^(0)。
  2. Round 1: 从 x^(0) 提取书法形态的中心线骨架 g^(1) = Encode(Skeletonize(x^(0)))。
     此时 g^(1) 已蕴含书家风格形变（牵丝、外拓、倾斜），比 g^(0) 更贴近真迹骨架！
  3. 将 g^(1) 循环输入回模型，重新采样生成 x^(1)。
  4. 重复至 Round K，观察 SSIM、LPIPS、Skel-IoU 是否产生单调正向增益。
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
from PIL import Image
from scipy.ndimage import binary_dilation
from skimage.morphology import skeletonize

# 路径自适应
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from src.model import DiT_2Cond_models
from src.loss import create_diffusion_or_flow
from src.eval.inference import _mse, _ssim, load_eval_vae, sample_latents


def parse_args():
    parser = argparse.ArgumentParser(description="骨架自循环重采样提精实验")
    parser.add_argument("--ckpt", type=str,
                        default="/home/ds/Workspace/DiT/experiments/recurrent_skel_refine/checkpoints/v46_5050_40k.pt",
                        help="用于循环生成的模型权重路径")
    parser.add_argument("--cache", type=str,
                        default="/home/ds/Workspace/moyi/data/top10_style23/eval_real200_cache.pt",
                        help="预构建的 eval200 测试集缓存文件")
    parser.add_argument("--vae-path", type=str,
                        default="/home/ds/Workspace/moyi/models/sd-vae-ft-ema",
                        help="SD-VAE 权重目录")
    parser.add_argument("--out-dir", type=str,
                        default="/home/ds/Workspace/DiT/experiments/recurrent_skel_refine/results",
                        help="评测指标与海报输出目录")
    parser.add_argument("--rounds", type=int, default=3,
                        help="自循环提精轮数 (0=仅基线, 1=1次循环, 2=2次循环, 3=3次循环)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="评测样本数量 (最大 200)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="生成 Batch Size")
    parser.add_argument("--steps", type=int, default=50,
                        help="ODE 采样步数")
    parser.add_argument("--cfg", type=float, default=0.7,
                        help="CFG 引导强度")
    parser.add_argument("--blend-alpha", type=float, default=1.0,
                        help="骨架潜变量融合系数: g = (1-alpha)*g_std + alpha*g_extract (1.0=纯提取, 0.5=平滑半融合)")
    parser.add_argument("--alpha-sweep", action="store_true",
                        help="是否运行 alpha 遍历实验 (测试 alpha 在 [0.1, 0.2, 0.3, 0.5, 0.7, 1.0] 的表现)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备")
    return parser.parse_args()


def extract_and_encode_skeleton(imgs_tensor, vae, device):
    """从生成图像批量提取 1px 中心线骨架，经 3px 膨胀后编码为 VAE 潜变量条件 g。
    
    Args:
        imgs_tensor: (B, 3, 256, 256) float tensor, 值域 [-1, 1], 白底黑字
        vae: AutoencoderKL
    Returns:
        g_latent: (B, 4, 32, 32) float tensor, VAE 潜变量
        skel_preview: (B, 1, 256, 256) uint8 numpy, 骨架可视化图 (白底黑线 0/255)
    """
    B, C, H, W = imgs_tensor.shape
    # 灰度化: 白底黑字，墨迹通常像素值较小 (< 0)
    gray = imgs_tensor.mean(dim=1)  # (B, 256, 256)
    ink = (gray < 0.0).cpu().numpy()

    skel_3px = np.zeros((B, H, W), dtype=np.float32)
    struct3 = np.ones((3, 3), bool)

    for i in range(B):
        if ink[i].any():
            sk1 = skeletonize(ink[i])
            sk3 = binary_dilation(sk1, structure=struct3, iterations=1)
            skel_3px[i] = sk3.astype(np.float32)
        else:
            skel_3px[i] = 0.0

    # 构造白底黑线标准输入格式 (背景=+1.0, 墨迹=-1.0)
    skel_img = (1.0 - skel_3px) * 2.0 - 1.0
    skel_t = th.from_numpy(skel_img).unsqueeze(1).repeat(1, 3, 1, 1).to(device)

    with th.no_grad():
        g_latent = vae.encode(skel_t).latent_dist.mode() * 0.18215

    skel_preview = ((1.0 - skel_3px) * 255.0).astype(np.uint8)
    return g_latent, skel_preview


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


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = th.device(args.device if th.cuda.is_available() else "cpu")

    print("=" * 80)
    print("【启动骨架自循环重采样提精 (Recurrent Skeleton Refinement) 评测】")
    print(f"  模型权重: {args.ckpt}")
    print(f"  测试缓存: {args.cache}")
    print(f"  循环轮数: 0 ~ {args.rounds}")
    print(f"  评估样本: {args.num_samples}")
    print(f"  输出目录: {args.out_dir}")
    print("=" * 80)

    # 1. 加载测试集缓存
    print("\n[1/4] 读取真实书法测试集缓存 ...", flush=True)
    cache = th.load(args.cache, map_location="cpu", weights_only=False)
    N = min(args.num_samples, cache["noise"].shape[0])
    
    fixed_noise = cache["noise"][:N].to(device)
    std_lats = cache["std_lats"][:N].to(device)  # Round 0 初始条件
    gt_pngs = cache["gt_pngs"][:N]               # (N, 3, 256, 256), [0, 1]
    conds_raw = cache["conds"][:N]               # 风格条件元组列表
    rows_meta = cache["rows"][:N] if "rows" in cache else [{}] * N

    # 2. 加载 VAE 与评测感知器 (LPIPS)
    print("[2/4] 加载 SD-VAE 与 LPIPS 感知模型 ...", flush=True)
    vae = load_eval_vae(device, args.vae_path)
    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    except Exception as e:
        print(f"  [warn] LPIPS 不可用: {e}")
        lpips_fn = None

    # 3. 构建 DiT-2Cond-S/2 模型
    print("[3/4] 初始化 DiT-2Cond-S/2 并加载 Checkpoint ...", flush=True)
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
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"  权重加载成功: missing={len(missing)}, unexpected={len(unexpected)}")

    # 构建 Flow ODE 采样器
    diffusion = create_diffusion_or_flow(
        str(args.steps),
        diffusion_type="flow",
        t_sampler="logit_normal",
        sampler="heun",
        shift=1.0,
    )

    # 4. 逐轮运行自循环重采样
    print("\n[4/4] 启动循环提精流水线 ...", flush=True)
    round_metrics = []
    
    # 保存各轮生成的样本图像，供海报与跨轮对比使用
    # round_samples[round_idx] = (N, 3, 256, 256) float32 [0, 1]
    round_samples = {}
    round_skels = {}

    current_g = std_lats  # 初始条件 = 标准字形骨架

    for r in range(args.rounds + 1):
        r_name = "Round 0 (Std Baseline)" if r == 0 else f"Round {r} (Recurrent Refined)"
        print(f"\n>>> 正在运行 {r_name} 生成与评测 (N={N}) ...", flush=True)
        t0 = time.time()

        # DiT 潜变量采样
        lat = sample_latents(
            model, diffusion, fixed_noise, conds_raw,
            cfg_scale=args.cfg, batch=args.batch_size, device=device,
            skel=current_g, seed=0
        )

        # VAE 解码为像素图像 [-1, 1]
        dec_list = []
        with th.no_grad():
            for s in range(0, N, args.batch_size):
                e = min(s + args.batch_size, N)
                batch_lat = lat[s:e].to(device)
                if device.type == "cuda":
                    with th.autocast("cuda", dtype=th.bfloat16):
                        dec = vae.decode(batch_lat / 0.18215).sample
                else:
                    dec = vae.decode(batch_lat / 0.18215).sample
                dec_list.append(dec.clamp(-1.0, 1.0))
        all_gen_t = th.cat(dec_list, dim=0).float()
        # 转为 [0, 1] 用于评估
        all_gen_01 = (all_gen_t + 1.0) / 2.0
        round_samples[r] = all_gen_01.cpu().numpy()

        # 计算本轮指标
        ssims, lpips_list, skel_ious, mses = [], [], [], []
        gt_np = gt_pngs.numpy().transpose(0, 2, 3, 1)  # (N, 256, 256, 3)
        pred_np = round_samples[r].transpose(0, 2, 3, 1)

        for i in range(N):
            p_img = pred_np[i]
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

        stat = {
            "round": r,
            "name": r_name,
            "ssim_mean": round(ssim_mean, 4),
            "ssim_med": round(ssim_med, 4),
            "lpips_mean": round(lpips_mean, 4),
            "skel_iou": round(skel_mean, 4),
            "mse_mean": round(mse_mean, 4),
            "time_sec": round(dt, 1),
        }
        round_metrics.append(stat)

        print(f"  ✓ {r_name} 耗时: {dt:.1f}s | "
              f"SSIM: {ssim_mean:.4f} (med {ssim_med:.4f}) | "
              f"LPIPS: {lpips_mean:.4f} | "
              f"Skel-IoU: {skel_mean:.4f} | "
              f"MSE: {mse_mean:.4f}")

        # 若还有下一轮，执行骨架提取并编码为下一轮的条件 g
        if r < args.rounds and not args.alpha_sweep:
            print(f"  -> 提取 Round {r} 生成图骨架并重新编码为潜变量 ...", flush=True)
            g_extracted, skel_prev = extract_and_encode_skeleton(all_gen_t, vae, device)
            round_skels[r + 1] = skel_prev
            current_g = (1.0 - args.blend_alpha) * std_lats + args.blend_alpha * g_extracted

    # ── 如果开启了 Alpha 遍历实验模式 ───────────────────────────────────────────
    if args.alpha_sweep:
        print("\n" + "=" * 90)
        print("【启动 Alpha 软融合插值系数扫描实验 (Alpha Sweep)】")
        print("  公式: g_cond = (1 - alpha) * g_std + alpha * g_extracted")
        print("=" * 90)
        # 提取 Round 0 的骨架
        print("  -> 提取 Round 0 生成图骨架并重新编码 ...", flush=True)
        g_extracted, skel_r0 = extract_and_encode_skeleton(all_gen_t, vae, device)

        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
        alpha_metrics = [round_metrics[0]]  # 以 alpha=0.0 (Round 0) 作为起点

        for alpha in alphas:
            alpha_name = f"Blend alpha={alpha:.2f}"
            print(f"\n>>> 正在测试 {alpha_name} (N={N}) ...", flush=True)
            t0 = time.time()
            g_blend = (1.0 - alpha) * std_lats + alpha * g_extracted

            # 采样
            lat = sample_latents(
                model, diffusion, fixed_noise, conds_raw,
                cfg_scale=args.cfg, batch=args.batch_size, device=device,
                skel=g_blend, seed=0
            )

            # 解码
            dec_list = []
            with th.no_grad():
                for s in range(0, N, args.batch_size):
                    e = min(s + args.batch_size, N)
                    batch_lat = lat[s:e].to(device)
                    if device.type == "cuda":
                        with th.autocast("cuda", dtype=th.bfloat16):
                            dec = vae.decode(batch_lat / 0.18215).sample
                    else:
                        dec = vae.decode(batch_lat / 0.18215).sample
                    dec_list.append(dec.clamp(-1.0, 1.0))
            all_gen_t = th.cat(dec_list, dim=0).float()
            all_gen_01 = (all_gen_t + 1.0) / 2.0
            pred_np = all_gen_01.cpu().numpy().transpose(0, 2, 3, 1)

            # 评估
            ssims, lpips_list, skel_ious, mses = [], [], [], []
            for i in range(N):
                p_img = pred_np[i]
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

            stat = {
                "round": f"alpha_{alpha}",
                "name": alpha_name,
                "ssim_mean": round(ssim_mean, 4),
                "ssim_med": round(ssim_med, 4),
                "lpips_mean": round(lpips_mean, 4),
                "skel_iou": round(skel_mean, 4),
                "mse_mean": round(mse_mean, 4),
                "time_sec": round(dt, 1),
            }
            alpha_metrics.append(stat)
            print(f"  ✓ {alpha_name} 耗时: {dt:.1f}s | "
                  f"SSIM: {ssim_mean:.4f} | "
                  f"LPIPS: {lpips_mean:.4f} | "
                  f"Skel-IoU: {skel_mean:.4f} | "
                  f"MSE: {mse_mean:.4f}")

        round_metrics = alpha_metrics

    # 5. 输出汇总表
    print("\n" + "=" * 90)
    print("【骨架自循环重采样全周期实验结果对比总表】")
    print("=" * 90)
    print(f"{'轮次 (Round)':<26} | {'Strict SSIM':<12} | {'Strict LPIPS':<14} | {'Skel IoU':<10} | {'MSE':<8}")
    print("-" * 90)
    for m in round_metrics:
        print(f"{m['name']:<26} | {m['ssim_mean']:<12.4f} | {m['lpips_mean']:<14.4f} | {m['skel_iou']:<10.4f} | {m['mse_mean']:<8.4f}")
    print("=" * 90)

    # 保存 JSON 与 CSV
    res_json_path = os.path.join(args.out_dir, "recurrent_refine_metrics.json")
    with open(res_json_path, "w", encoding="utf-8") as f:
        json.dump(round_metrics, f, indent=2, ensure_ascii=False)

    res_csv_path = os.path.join(args.out_dir, "recurrent_refine_metrics.csv")
    with open(res_csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(round_metrics[0].keys()))
        w.writeheader()
        w.writerows(round_metrics)

    # 6. 生成代表性样本横向演化海报
    print("\n[海报生成] 正在绘制多轮次演化对比海报 ...", flush=True)
    num_show = min(10, N)
    cols = 1 + 1 + (args.rounds + 1)  # GT + Std + Round 0..R
    canvas_w = cols * 256
    canvas_h = num_show * 256
    poster = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

    std_pngs_np = cache["std_pngs"][:num_show].numpy().transpose(0, 2, 3, 1)

    for row_idx in range(num_show):
        y_off = row_idx * 256
        # Col 0: GT
        gt_im = Image.fromarray((gt_np[row_idx] * 255).astype(np.uint8))
        poster.paste(gt_im, (0, y_off))
        # Col 1: Std Skel
        std_im = Image.fromarray((std_pngs_np[row_idx] * 255).astype(np.uint8))
        poster.paste(std_im, (256, y_off))
        # Col 2..: Round 0..K
        for r in range(args.rounds + 1):
            x_off = (2 + r) * 256
            r_im = Image.fromarray((round_samples[r][row_idx].transpose(1, 2, 0) * 255).astype(np.uint8))
            poster.paste(r_im, (x_off, y_off))

    poster_path = os.path.join(args.out_dir, "recurrent_refine_poster.png")
    poster.save(poster_path)
    print(f"  ✓ 演化对比海报已保存至: {poster_path}")
    print("\n[完成] 骨架自循环重采样实验执行完毕！")


if __name__ == "__main__":
    main()
