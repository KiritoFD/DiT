#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""watch_and_chain_tier4.py — Tier 3 (B/2 @ 60k) 完训自动评测与 Tier 4 (L/2) 无缝交接控制器

运行逻辑:
  1. 后台长周期静默守候 Tier 3 B/2 达到 60,000 步 (约今晚 01:40);
  2. 探测到 0060000.pt 落盘且训练主进程释放显存后，自动调用 tools/batch_eval_aug_ladder.py
     在 GPU 上对 10k, 20k, 30k, 40k, 50k, 60k 全周期 checkpoints 执行黄金测试集评测;
  3. 评测完成产出总表后，自动后台启动 Tier 4 Large (DiT-2Cond-L/2, 464M 参数, ~340M+ 纯骨干)
     无缝衔接夜间 16h 算力，冲击超大模型容量极限！
"""

import glob
import os
import subprocess
import sys
import time

CKPT_DIR = "/home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug_v66route/20261006-120802-v66_cap_tier3_b_aug_route2456/checkpoints"
TARGET_CKPT = os.path.join(CKPT_DIR, "0060000.pt")
TARGET_DONE = os.path.join(CKPT_DIR, "0060000.pt.done")

PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
WORKSPACE = "/home/ds/Workspace/DiT"
EVAL_SCRIPT = os.path.join(WORKSPACE, "tools/batch_eval_aug_ladder.py")

TIER4_CONFIG = os.path.join(WORKSPACE, "experiments/capacity_ladder/configs/tier4_l_aug_v66route.json")
TIER4_LOG = os.path.join(WORKSPACE, "experiments/capacity_ladder/logs/tier4_l_aug_v66route_train.log")
CHAIN_LOG = os.path.join(WORKSPACE, "experiments/capacity_ladder/logs/chain_tier4.log")


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(CHAIN_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_training_active():
    """检查是否有 Tier 3 训练主进程在跑。"""
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if "train.py" in line and "tier3_b_aug_v66route" in line:
                return True
    except Exception:
        pass
    return False


def wait_for_tier3_completion():
    log("=== [控制器启动] 开始守候 Tier 3 (B/2) 训练达到 60,000 步 ===")
    log(f"  目标权重: {TARGET_CKPT}")

    check_interval = 120  # 每 2 分钟检查一次，严格杜绝频繁轮询

    while True:
        target_ready = os.path.exists(TARGET_CKPT) or os.path.exists(TARGET_DONE)
        training_running = is_training_active()

        if target_ready and not training_running:
            log("✓ 检测到 0060000.pt 已落盘且训练进程已退出释放显存！")
            time.sleep(10)  # 留 10s 确保显存彻底回收
            break

        if not training_running and not target_ready:
            # 异常退出排查
            log("! 警告: 训练进程未在运行，但尚未达到 0060000.pt。")
            ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pt")))
            if ckpts:
                latest = os.path.basename(ckpts[-1])
                log(f"  当前最新权重: {latest}")
                # 若已有 0055000.pt 以上或确认停止，可在此等待或按最新评测
            time.sleep(check_interval)
            continue

        time.sleep(check_interval)


def run_milestone_evaluation():
    log("=== [第二阶段] 开始全周期里程碑权重批量 GPU 评测 ===")
    cmd = [
        PYTHON, "-u", EVAL_SCRIPT,
        "--ckpt-dir", CKPT_DIR,
        "--steps", "10000,20000,30000,40000,50000,60000",
        "--python", PYTHON
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = WORKSPACE
    res = subprocess.run(cmd, env=env, cwd=WORKSPACE)
    if res.returncode == 0:
        log("✓ Tier 3 (B/2) 全周期里程碑评测圆满成功！")
    else:
        log(f"✗ 评测过程中退出码: {res.returncode}，请检查评测日志。")


def launch_tier4_large():
    log("=== [第三阶段] 自动启动 Tier 4 Large (464M 参数) 旗舰训练 ===")
    os.makedirs(os.path.dirname(TIER4_LOG), exist_ok=True)
    
    cmd_str = (
        f"export PYTHONPATH={WORKSPACE} && "
        f"nohup {PYTHON} -u src/train/train.py "
        f"--config {TIER4_CONFIG} "
        f">> {TIER4_LOG} 2>&1 &"
    )
    log(f"  执行启动命令: {cmd_str}")
    subprocess.Popen(cmd_str, shell=True, cwd=WORKSPACE)
    time.sleep(15)
    
    # 验证 Tier 4 启动状态
    out = subprocess.check_output(["ps", "aux"], text=True)
    t4_running = any("tier4_l_aug" in line for line in out.splitlines())
    if t4_running:
        log("✓ Tier 4 Large (DiT-2Cond-L/2) 训练进程启动成功，显存正常分配！")
    else:
        log("! 警告: 未探测到 Tier 4 活跃进程，请检查日志: " + TIER4_LOG)
    log("=== 控制器任务全部交付完毕 ===")


def main():
    wait_for_tier3_completion()
    run_milestone_evaluation()
    launch_tier4_large()


if __name__ == "__main__":
    main()
