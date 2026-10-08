#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v68_official_sde_sweep.py — 历史巅峰模型 v68 @ 200k 步官方同频 SiT SDE 随机纠偏评测

调用官方 run_in_mem_eval 核心，完全对齐 eval200fix (N=187 严格真迹) 评测协议：
  - Checkpoint: v68_aug_sp_c2ot @ 200,000 steps (0200000.pt, DiT-Sp/2, 59.2M)
  - 官方历史读数 (ODE gamma=0.0): SSIM=0.6149, LPIPS=0.3249, Skel-IoU=0.0334, MSE=0.7496
  - 扫描 SiT SDE 纠偏系数: gamma in [0.0, 0.10, 0.20, 0.35, 0.50]
"""

import argparse
import copy
import csv
import json
import os
import sys
import time

import torch

_root = "/home/ds/Workspace/DiT"
sys.path.insert(0, _root)

from src.model import DiT_2Cond_models
from src.eval.in_mem_eval import run_in_mem_eval


class ArgsObject:
    """透传配置对象。"""
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str,
                        default="/home/ds/Workspace/DiT/experiments/v68_aug_sp_c2ot/checkpoints/0200000.pt")
    parser.add_argument("--config", type=str,
                        default="/home/ds/Workspace/DiT/experiments/v68_aug_sp_c2ot/resolved_config.json")
    parser.add_argument("--out-dir", type=str,
                        default="/home/ds/Workspace/DiT/experiments/v68_sit_sde_sweep_results")
    parser.add_argument("--gammas", type=str, default="0.0,0.10,0.20,0.35,0.50")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--cfg", type=float, default=0.7)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cli_args.out_dir, exist_ok=True)

    gamma_list = [float(g.strip()) for g in cli_args.gammas.split(",") if g.strip()]

    print("=" * 90)
    print("【启动历史巅峰模型 v68 @ 200k 步 SiT SDE 官方同频评测】")
    print(f"  模型权重: {cli_args.ckpt}")
    print(f"  配置文件: {cli_args.config}")
    print(f"  扫描 Gammas: {gamma_list}")
    print(f"  采样步数: {cli_args.steps} | CFG: {cli_args.cfg}")
    print(f"  输出目录: {cli_args.out_dir}")
    print("=" * 90)

    # 1. 读取基础配置
    raw_cfg = json.load(open(cli_args.config, encoding="utf-8"))
    
    # 路径适配 48 机器
    raw_cfg["eval_csv"] = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
    raw_cfg["in_mem_eval_sets"] = "eval200fix:/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv:187"
    raw_cfg["triple_table_prefix"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/"
    raw_cfg["char_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/char_remap.json"
    raw_cfg["callig_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/callig_remap.json"
    raw_cfg["font_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/font_remap.json"
    raw_cfg["eval_vae_path"] = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
    raw_cfg["vae_path"] = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
    raw_cfg["eval_steps"] = cli_args.steps
    raw_cfg["eval_cfg"] = cli_args.cfg
    raw_cfg["in_mem_eval_batch"] = 16
    raw_cfg["in_mem_eval_vae_batch"] = 16
    raw_cfg["in_mem_eval_save_samples"] = False  # 调参扫描不落盘冗余 PNG

    # 2. 构建 DiT-2Cond-Sp/2 模型
    print("[1/3] 正在构建 DiT-2Cond-Sp/2 并加载 200k 步 EMA 权重 ...", flush=True)
    model = DiT_2Cond_models[raw_cfg.get("model", "DiT-2Cond-Sp/2")](
        learn_sigma=False,
        norm_type=raw_cfg.get("norm_type", "layer"),
        mlp_type=raw_cfg.get("mlp_type", "gelu"),
        qk_norm=raw_cfg.get("qk_norm", 0),
        rope=raw_cfg.get("rope", 0),
        condition_fusion=raw_cfg.get("condition_fusion", "factorized_cat"),
        cond_fusion_norm=raw_cfg.get("cond_fusion_norm", "split"),
        num_calligraphers=raw_cfg.get("num_calligraphers", 10),
        num_characters=raw_cfg.get("num_characters", 4690),
        num_script_classes=raw_cfg.get("num_script_classes", 3),
        use_script_cond=raw_cfg.get("use_script_cond", True),
        char_embed_dim=raw_cfg.get("char_embed_dim", 256),
        callig_embed_dim=raw_cfg.get("callig_embed_dim", 32),
        script_embed_dim=raw_cfg.get("script_embed_dim", 16),
        glyph_inject_layers=raw_cfg.get("glyph_inject_layers", 0),
        cond_inject_at=raw_cfg.get("cond_inject_at", "2,4,5,6"),
        cond_inject_scale=raw_cfg.get("cond_inject_scale", True),
    ).to(device)

    ckpt = torch.load(cli_args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("ema") or ckpt.get("model") or ckpt
    if hasattr(state, "state_dict"):
        state = state.state_dict()
    clean_sd = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"  ✓ 权重加载成功 (missing={len(missing)}, unexpected={len(unexpected)})")
    model.eval()

    # 3. 逐个 Gamma 运行官方评测
    print("\n[2/3] 开始官方同频评测扫描 ...", flush=True)
    sweep_results = []

    for gamma in gamma_list:
        tag = "ODE Baseline" if gamma == 0.0 else f"SDE gamma={gamma:.2f}"
        print(f"\n>>> 正在运行 [{tag}] 官方评测 (N=187) ...", flush=True)
        t0 = time.time()

        cfg_copy = copy.deepcopy(raw_cfg)
        cfg_copy["sde_gamma"] = gamma
        args_obj = ArgsObject(cfg_copy)

        mode_out_dir = os.path.join(cli_args.out_dir, f"gamma_{gamma:.2f}")
        os.makedirs(mode_out_dir, exist_ok=True)

        res = run_in_mem_eval(
            model, args_obj, step=200000, device=device,
            results_dir=mode_out_dir, logger=print
        )
        dt = time.time() - t0

        e200_res = res.get("eval200fix", {})
        ssim = float(e200_res.get("ssim", 0.0))
        lpips_val = float(e200_res.get("lpips", 0.0))
        skel_iou = float(e200_res.get("skel_iou", 0.0))
        mse_val = float(e200_res.get("mse", 0.0))
        ink_ssim = float(e200_res.get("ink_ssim", 0.0))
        ink_iou = float(e200_res.get("ink_iou", 0.0))
        frag = float(e200_res.get("frag", 0.0))

        entry = {
            "gamma": gamma,
            "tag": tag,
            "strict_ssim": round(ssim, 4),
            "strict_lpips": round(lpips_val, 4),
            "skel_iou": round(skel_iou, 4),
            "mse": round(mse_val, 4),
            "ink_ssim": round(ink_ssim, 4),
            "ink_iou": round(ink_iou, 4),
            "frag": round(frag, 4),
            "time_sec": round(dt, 1),
        }
        sweep_results.append(entry)

        # 实时落盘
        json_path = os.path.join(cli_args.out_dir, "v68_sde_sweep_results.json")
        csv_path = os.path.join(cli_args.out_dir, "v68_sde_sweep_results.csv")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(sweep_results, f, indent=2, ensure_ascii=False)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_results[0].keys()))
            w.writeheader()
            w.writerows(sweep_results)

        print(f"  ✓ [{tag}] 耗时 {dt:.1f}s | "
              f"Strict SSIM: {ssim:.4f} | "
              f"LPIPS: {lpips_val:.4f} | "
              f"Skel-IoU: {skel_iou:.4f} | "
              f"MSE: {mse_val:.4f} | "
              f"Ink-IoU: {ink_iou:.4f}")

    # 4. 汇总总表
    print("\n" + "=" * 105)
    print("【历史巅峰模型 v68 @ 200k 步 SiT SDE 随机纠偏参数扫描全量对比总表】")
    print("=" * 105)
    print(f"{'配置模式':<22} | {'Strict SSIM':<12} | {'Strict LPIPS':<14} | {'Skel IoU':<10} | {'MSE':<8} | {'Ink-IoU':<8} | {'增益评价'}")
    print("-" * 105)
    base_ssim = sweep_results[0]["strict_ssim"]
    base_lp = sweep_results[0]["strict_lpips"]
    base_skel = sweep_results[0]["skel_iou"]

    for r in sweep_results:
        d_ssim = r["strict_ssim"] - base_ssim
        d_lp = r["strict_lpips"] - base_lp
        d_skel = r["skel_iou"] - base_skel
        gain_note = f"SSIM {d_ssim:+.4f}, LP {d_lp:+.4f}, Skel {d_skel:+.4f}" if r["gamma"] > 0 else "历史巅峰基准"
        print(f"{r['tag']:<22} | {r['strict_ssim']:<12.4f} | {r['strict_lpips']:<14.4f} | {r['skel_iou']:<10.4f} | {r['mse']:<8.4f} | {r['ink_iou']:<8.4f} | {gain_note}")
    print("=" * 105)
    print(f"\n[完成] 全量评估数据已保存至: {csv_path} 与 {json_path}")


if __name__ == "__main__":
    main()
