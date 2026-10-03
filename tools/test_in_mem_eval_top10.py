#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from src.eval.in_mem_eval import run_in_mem_eval
from src.train.cli import parse_args

def main():
    print("=== [Eval Test] 验证在内存评测流程 ===")
    from src.model.dit import DiT_models
    model = DiT_models["DiT-2Cond-S/2"](
        num_calligraphers=23,
        glyph_cond=True,
        deform_skel=1,
        cond_fusion_norm="split"
    ).cuda().eval()

    class DummyArgs:
        in_mem_eval_sets = "seen:assets/eval_top10_seen_20.csv:20,strict:assets/eval_top10_strict_subset84.csv:84"
        eval_skel_latent_shards_dir = "data/50k_v2_glyph15k/shards_std"
        skel_latent_shards_dir = "data/top10_style23/shards_std"
        data_csv = "assets/train_top10_style23.csv"
        diffusion_type = "flow"
        eval_cfg = 0.7
        eval_steps = 5  # 快速测 5 步
        in_mem_eval_batch = 16
        in_mem_eval_vae_batch = 16
        vae_scaling_factor = 0.18215
        eval_self_cond = False
        eval_blend_alpha = 0.0

    print("开始测试 run_in_mem_eval ...")
    res = run_in_mem_eval(
        model=model,
        args=DummyArgs(),
        step=0,
        results_dir="/tmp/test_eval_top10",
        n_eval=10
    )
    print("🎉 in_mem_eval 快速测试通过！返回结果:", res)

if __name__ == "__main__":
    main()
