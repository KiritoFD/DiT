#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_v68_steps_sweep.py — 旗舰模型采样步数 (NFE / Steps) 扫描器

在确定的最优引导配置 (CFG=0.80, SiT SDE gamma=0.50) 下，扫描采样步数,
寻找「质量 vs 推理速度」的最优折衷点。

评测协议: eval200_fixed (N=187 严格古代真迹)。
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
                        default="/home/ds/Workspace/DiT/experiments/v68_steps_sweep")
    parser.add_argument("--steps-list", type=str, default="10,20,30,50,100,200")
    parser.add_argument("--cfg", type=float, default=0.8)
    parser.add_argument("--sde-gamma", type=float, default=0.5)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    cli_args = parse_args()
    device = torch.device(cli_args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(cli_args.out_dir, exist_ok=True)

    steps_list = [int(s.strip()) for s in cli_args.steps_list.split(",") if s.strip()]

    print("=" * 95)
    print("【启动旗舰模型 (v68 @ 200k) 采样步数扫描】")
    print(f"  模型权重:   {cli_args.ckpt}")
    print(f"  固定 CFG:   {cli_args.cfg}")
    print(f"  固定 gamma: {cli_args.sde_gamma}")
    print(f"  步数列表:   {steps_list}")
    print(f"  输出目录:   {cli_args.out_dir}")
    print("=" * 95)

    raw_cfg = json.load(open(cli_args.config, encoding="utf-8"))
    raw_cfg["eval_csv"] = "/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv"
    raw_cfg["in_mem_eval_sets"] = "eval200fix:/home/ds/Workspace/moyi/exp-std-csv/eval200_fixed.csv:187"
    raw_cfg["triple_table_prefix"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/"
    raw_cfg["char_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/char_remap.json"
    raw_cfg["callig_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/callig_remap.json"
    raw_cfg["font_remap_json"] = "/home/ds/Workspace/DiT/assets/triple_tables_best_minimal/font_remap.json"
    raw_cfg["eval_vae_path"] = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
    raw_cfg["vae_path"] = "/home/ds/Workspace/moyi/models/sd-vae-ft-ema"
    raw_cfg["in_mem_eval_batch"] = 16
    raw_cfg["in_mem_eval_vae_batch"] = 16
    raw_cfg["in_mem_eval_save_samples"] = False

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

    print("\n[2/3] 启动采样步数扫描 ...", flush=True)
    results = []
    json_path = os.path.join(cli_args.out_dir, "steps_sweep_results.json")
    csv_path = os.path.join(cli_args.out_dir, "steps_sweep_results.csv")

    for idx, steps in enumerate(steps_list, 1):
        print(f"\n>>> [{idx:02d}/{len(steps_list):02d}] 正在评估 steps={steps} (N=187) ...", flush=True)
        t0 = time.time()

        cfg_copy = copy.deepcopy(raw_cfg)
        cfg_copy["eval_cfg"] = cli_args.cfg
        cfg_copy["sde_gamma"] = cli_args.sde_gamma
        cfg_copy["eval_steps"] = steps
        args_obj = ArgsObject(cfg_copy)

        pt_dir = os.path.join(cli_args.out_dir, f"steps{steps}")
        os.makedirs(pt_dir, exist_ok=True)

        res = run_in_mem_eval(
            model, args_obj, step=200000, device=device,
            results_dir=pt_dir, logger=print
        )
        dt = time.time() - t0

        e200 = res.get("eval200fix", {})
        entry = {
            "steps": steps,
            "cfg": cli_args.cfg,
            "gamma": cli_args.sde_gamma,
            "strict_ssim": round(float(e200.get("ssim", 0.0)), 4),
            "strict_lpips": round(float(e200.get("lpips", 0.0)), 4),
            "skel_iou": round(float(e200.get("skel_iou", 0.0)), 4),
            "mse": round(float(e200.get("mse", 0.0)), 4),
            "frag": round(float(e200.get("frag", 0.0)), 4),
            "time_sec": round(dt, 1),
        }
        results.append(entry)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

        print(f"  ✓ steps={steps} 耗时 {dt:.1f}s | SSIM: {entry['strict_ssim']:.4f} | "
              f"LPIPS: {entry['strict_lpips']:.4f} | Frag: {entry['frag']:.3f} | MSE: {entry['mse']:.4f}",
              flush=True)

    print("\n" + "=" * 95)
    print("【采样步数扫描总表】(CFG=%.2f, gamma=%.2f)" % (cli_args.cfg, cli_args.sde_gamma))
    print("=" * 95)
    print(f"{'Steps':<8} | {'SSIM':<10} | {'LPIPS':<10} | {'Skel IoU':<10} | {'Frag':<10} | {'MSE':<10} | {'耗时(s)':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['steps']:<8} | {r['strict_ssim']:<10.4f} | {r['strict_lpips']:<10.4f} | "
              f"{r['skel_iou']:<10.4f} | {r['frag']:<10.3f} | {r['mse']:<10.4f} | {r['time_sec']:<8.1f}")
    print("=" * 95)
    print(f"[完成] 扫描数据已落盘至: {csv_path}")


if __name__ == "__main__":
    main()
