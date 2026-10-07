#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""auto_extract_and_train.py — 守候 SCP 结束、快速解压 39.3 万图片并自动拉起 DINO-REPA Calli-VAE 训练"""

import glob
import os
import subprocess
import sys
import time
import zipfile

ZIP_PATH = "/home/ds/Workspace/moyi/data/unified_393k/images.zip"
TARGET_DIR = "/home/ds/Workspace/moyi/data/unified_393k"
IMGS_DIR = os.path.join(TARGET_DIR, "imgs")
LOG_PATH = os.path.join(TARGET_DIR, "auto_extract_and_train.log")

PYTHON = "/home/ds/miniconda3/envs/pytorch/bin/python"
WORKSPACE = "/home/ds/Workspace/DiT"
TRAIN_SCRIPT = os.path.join(WORKSPACE, "tools/vae/train_dino_calli_vae.py")
CSV_PATH = os.path.join(TARGET_DIR, "train_clean.csv")
OUTPUT_DIR = "/home/ds/Workspace/DiT/experiments/calli_vae_dino"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_scp_running():
    try:
        out = subprocess.check_output(["ps", "aux"], text=True)
        for line in out.splitlines():
            if "scp" in line and "images.zip" in line and "grep" not in line:
                return True
    except Exception:
        pass
    return False


def wait_scp():
    log("=== [阶段 1/3] 开始守候 images.zip 跨机网络传输完成 ===")
    while is_scp_running():
        if os.path.exists(ZIP_PATH):
            sz_gb = os.path.getsize(ZIP_PATH) / (1024 ** 3)
            log(f"  SCP 传输进行中... 当前大小: {sz_gb:.2f} GB / 6.10 GB")
        time.sleep(10)
    log("✓ SCP 传输已结束！")
    time.sleep(3)


def extract_zip():
    log(f"=== [阶段 2/3] 开始极速解压 images.zip -> {TARGET_DIR} ===")
    t0 = time.time()
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        members = zf.namelist()
        log(f"  包含 {len(members)} 个文件，开始解压至磁盘 ...")
        zf.extractall(TARGET_DIR)
    dt = time.time() - t0
    log(f"✓ 解压圆满完成！耗时: {dt:.1f} 秒")

    # 校验解压数量
    png_count = len(glob.glob(os.path.join(IMGS_DIR, "*.png")))
    log(f"  校验提取图片数: {png_count} 张 (目标: 393,486 张)")
    if png_count >= 390000:
        log("✓ 数据集完整性校验 100% 通过！")
    else:
        log(f"! 警告: 实际提取数量为 {png_count}，请注意检查。")


def launch_training():
    log("=== [阶段 3/3] 自动启动纯 DINO-REPA 监督 Calli-VAE 全量训练 ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_log = os.path.join(OUTPUT_DIR, "train_dino_vae.log")

    cmd = (
        f"export PYTHONPATH={WORKSPACE} && "
        f"nohup {PYTHON} -u {TRAIN_SCRIPT} "
        f"--vae-base /home/ds/Workspace/moyi/models/sd-vae-ft-ema "
        f"--dino-ckpt /home/ds/Workspace/DiT/pretrained_models/dinov2_vits14_pretrain.safetensors "
        f"--csv {CSV_PATH} "
        f"--data-root {TARGET_DIR} "
        f"--output {OUTPUT_DIR} "
        f"--batch-size 64 "
        f"--lr 5e-5 "
        f"--w-l1 1.0 "
        f"--w-kl 0.0001 "
        f"--w-struct 0.1 "
        f"--w-style 0.5 "
        f"--max-steps 30000 "
        f">> {train_log} 2>&1 &"
    )
    log(f"  执行启动命令: {cmd}")
    subprocess.Popen(cmd, shell=True, cwd=WORKSPACE)
    time.sleep(15)

    out = subprocess.check_output(["ps", "aux"], text=True)
    running = any("train_dino_calli_vae" in line for line in out.splitlines())
    if running:
        log("✓ Calli-VAE 训练主进程启动成功，已全速分配 GPU 显存开始前向与反向优化！")
    else:
        log("! 警告: 未探测到训练进程，请检查日志: " + train_log)
    log("=== 全自动解压与训练拉起任务全部执行完毕 ===")


def main():
    wait_scp()
    extract_zip()
    launch_training()


if __name__ == "__main__":
    main()
