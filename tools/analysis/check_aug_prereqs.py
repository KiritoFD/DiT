#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_aug_prereqs.py — 起增广实验前的前置检查（在 48 上跑）

必须确认：
  1) 4 个配置存在且 batch/steps/use_c2ot 与 arm4 一致
  2) 每族的数据 CSV 与 shards 存在，行数对得上
  3) ★ 每族指向的 repa_cache_dir **存在且不是"旧 sym 语料缓存"**（否则增强行拿到源图特征 = 实验作废）
  4) GPU / 磁盘 / 是否有别的任务在跑
"""
import glob
import json
import os
import sys

os.chdir("/home/ds/Workspace/DiT")
sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAM = ["morphE", "elastic", "grid", "affine"]
cfgs = []
for f in FAM:
    cfgs += glob.glob(f"src/train/configs/*{f}*noOT*.json") + glob.glob(f"src/train/configs/*noOT*{f}*.json")
cfgs = sorted(set(cfgs))
print(f"=== 1) 找到 {len(cfgs)} 个候选配置 ===")
for c in cfgs:
    print("   ", c)
if not cfgs:
    print("  ✗ 没有 *noOT* 配置 —— 他处报告说已生成 4 个，检查命名")
    sys.exit(0)

print("\n=== 2) 逐配置的关键字段与路径存在性 ===")
for c in cfgs:
    d = json.load(open(c, encoding="utf-8"))
    print(f"\n  ── {os.path.basename(c)}")
    for k in ("data_csv", "latent_shards_dir", "repa_cache_dir", "global_batch_size",
              "max_steps", "use_c2ot", "use_ot", "w_repa", "vae_path"):
        v = d.get(k)
        if v is None:
            print(f"     {k:20} = (未设)")
            continue
        p = str(v)
        if k in ("data_csv", "latent_shards_dir", "repa_cache_dir", "vae_path"):
            ok = "✓" if os.path.exists(p) else "✗ 不存在"
            extra = ""
            if os.path.isdir(p):
                extra = f" ({len(os.listdir(p))} 项)"
            elif os.path.isfile(p):
                extra = f" ({sum(1 for _ in open(p, encoding='utf-8'))-1} 行)"
            print(f"     {k:20} = {p}   {ok}{extra}")
        else:
            print(f"     {k:20} = {v!r}")

print("\n=== 3) ★ REPA 缓存审计（关键陷阱）===")
cands = sorted(set(glob.glob("data/*repa*") + glob.glob("data/*/*repa*")
                     + glob.glob("exp-std/data/*repa*") + glob.glob("exp-std/data/*/*repa*")))
if not cands:
    print("  ✗ 没找到任何 *_repa_* 缓存目录")
for d in cands:
    if not os.path.isdir(d):
        continue
    n = len(glob.glob(os.path.join(d, "**", "*.npz"), recursive=True)) + \
        len(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True)) + \
        len(os.listdir(d))
    print(f"  {d}: {n} 项  {'← 旧 sym 语料缓存?' if 'top10_aug_v1' in d else ''}")
print("""
  判据: 增强族的 repa_cache_dir 必须是**该族自己**的缓存（每族 78,006 行）。
  若仍指向 data/dino_cache/top10_aug_v1 -> 增强行会拿到**源图**的 DINO 特征 = REPA 监督错误 = 实验作废 ✗
""")

print("=== 4) 资源 ===")
os.system("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader")
os.system("df -h . | tail -1")
os.system("tmux ls 2>&1 | head -3")
os.system("uptime | sed 's/.*load/load/'")
