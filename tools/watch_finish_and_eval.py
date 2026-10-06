#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch_finish_and_eval.py — 48 机训练完训自动触发阶段性权重批量 GPU 评测守护进程

逻辑:
  1. 监听 48,000 步最终权重 0048000.pt 是否生成完成
  2. 权重落盘后，等待 train.py 进程完全退出并释放 47GB 显存 (显存降至 0G)
  3. 立即自动调用 batch_eval_aug_ladder.py 启动全阶段 GPU 高速评测 (10k, 20k, 30k, 40k, 48k)
  4. 7~8 分钟内产出全周期阶段性 Strict SSIM / LPIPS / Skel 收敛总表
"""

import os
import subprocess
import sys
import time

CKPT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/checkpoints"
FINAL_CKPT = os.path.join(CKPT_DIR, "0048000.pt")
FINAL_DONE = FINAL_CKPT + ".done"

PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
EVAL_BATCH_SCRIPT = "/home/ds/Workspace/DiT/tools/batch_eval_aug_ladder.py"
LOG_FILE = "/home/ds/Workspace/DiT/experiments/capacity_ladder/logs/batch_eval_final.log"

print(f"[Watcher] 启动完训监听进程，目标监听终点: {FINAL_CKPT} ...", flush=True)

while True:
    # 检查 48000 步是否落盘
    if os.path.exists(FINAL_DONE) or (
        os.path.exists(FINAL_CKPT) and os.path.getsize(FINAL_CKPT) > 1.5 * 1024 * 1024 * 1024
    ):
        print(f"[Watcher] ✓ 检测到终局权重 {FINAL_CKPT} 已生成！", flush=True)
        print("[Watcher] 正在等待 train.py 训练主进程完全退出并释放 GPU 显存...", flush=True)
        time.sleep(15)

        # 检查显存是否已经释放
        for retry in range(20):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True
                )
                used_mb = int(res.stdout.strip())
                if used_mb < 2000:
                    print(f"[Watcher] ✓ GPU 显存已完全释放 (当前占用 {used_mb} MiB)！", flush=True)
                    break
            except Exception:
                pass
            time.sleep(10)

        print("[Watcher] >>> 自动拉起全量阶段性 Checkpoints 批量 GPU 评测流水线...", flush=True)
        cmd = f"{PYTHON} -u {EVAL_BATCH_SCRIPT} > {LOG_FILE} 2>&1"
        res = subprocess.run(cmd, shell=True)
        print(f"[Watcher] ★ 全阶段批量评测已全部执行完毕，退出码: {res.returncode}！", flush=True)
        break

    time.sleep(30)
