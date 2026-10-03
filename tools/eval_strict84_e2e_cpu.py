#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""eval_strict84_e2e_cpu.py — 在 CPU (32线程) 上使用官方 shards_std 评估 JointSkel2Img 在 Step 4,000 的真实端到端生成表现"""
import os, sys, json, csv, time, re
import numpy as np
import torch as th

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

# 启用 CPU 多线程
th.set_num_threads(32)
dev = th.device("cpu")

from src.eval import model_io
from src.eval.in_mem_eval import run_in_mem_eval
from src.model.joint_skel2img import JointSkel2Img
from types import SimpleNamespace

ckpt_path = "exp/v35_union/20261002-003600-v35-union-1step/checkpoints/0004000.pt"
print(f"[1] 载入 Checkpoint: {ckpt_path}")
ck = th.load(ckpt_path, map_location="cpu")
gen_sd = ck["gen_ema"]
gen_ckpt = ck["gen_ckpt"]
bak_ckpt = ck["bak_ckpt"]

print(f"[2] 装载模型到 CPU...")
gen, ga = model_io.load_model_from_ckpt(gen_ckpt, device=dev, use_ema=True)
bak, ba = model_io.load_model_from_ckpt(bak_ckpt, device=dev, use_ema=True)

# 载入 Step 4,000 EMA 权重
gen.load_state_dict({(k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v for k, v in gen_sd.items()})
gen.eval()
bak.eval()

# 构造统一端到端生成模型 (Stage 1 生成 25 步)
wrapper = JointSkel2Img(gen, bak, gen_steps=25).to(dev).eval()

ev_args = SimpleNamespace(**{k: v for k, v in vars(ba).items() if not k.startswith("_")})
ev_args.eval_blend_alpha = 0.0
ev_args.eval_cfg = 1.0
ev_args.eval_steps = 50
ev_args.eval_self_cond = False
ev_args.img_root = None

# ★ 关键修正: 彻底抛弃历史反相 Bug 目录，直接挂载官方 100% 匹配的标准骨架分片
ev_args.eval_skel_latent_shards_dir = "data/50k_v2_glyph15k/shards_std"
ev_args.eval_skel_latent_shards_dir_pred = ""

run_dir = "exp/v35_union/eval_cpu_step4000"
os.makedirs(run_dir, exist_ok=True)

print("\n[3] 开始在 CPU 上运行 Strict84 端到端评测 (84 样本)...")
t0 = time.time()
res = run_in_mem_eval(
    wrapper, ev_args, 4000, dev, run_dir,
    sets=[("strict84_e2e_official", "assets/eval_top10_strict_subset84.csv", 84)]
)
dt = time.time() - t0
print(f"\n[4] CPU 评测完成! 耗时: {dt:.1f}s")

print("\n" + "="*50)
print("=== Strict84 Step 4,000 官方分片真实端到端生成成绩 ===")
print("="*50)
for k, v in res.items():
    if isinstance(v, dict):
        print(f"[{k}]:")
        for sub_k, sub_v in v.items():
            print(f"  {sub_k}: {sub_v}")
    else:
        print(f"[{k}]: {v}")
