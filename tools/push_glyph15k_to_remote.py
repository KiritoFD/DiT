#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/push_glyph15k_to_remote.py — 一键推送 15.5k VAE Shards 与元数据至远程 4090 服务器"""
import glob
import os
import subprocess
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

REMOTE_HOST = "4090"
REMOTE_ROOT = "/root/Workspace/xy/DiT"


def check_local_shards():
    print("=== [1/4] 校验本地 VAE Shards 完整性 ===")
    img_shards = sorted(glob.glob("data/supplements_glyph/shards_img/shard_*.npz"))
    std_shards = sorted(glob.glob("data/supplements_glyph/shards_std/shard_*.npz"))

    if not img_shards:
        print("错误: 目标图像分片 shards_img 不存在！")
        return False
    if not std_shards:
        print("错误: 标准骨架分片 shards_std 不存在！")
        return False

    def count_samples(shards):
        cnt = 0
        for s in shards:
            d = np.load(s)
            cnt += len(d["img_ids"])
            d.close()
        return cnt

    img_cnt = count_samples(img_shards)
    std_cnt = count_samples(std_shards)
    print(f"  ✓ 目标图像分片: {len(img_shards)} 个分片，累计 {img_cnt} 条样本")
    print(f"  ✓ 标准骨架分片: {len(std_shards)} 个分片，累计 {std_cnt} 条样本")

    if img_cnt != 15548 or std_cnt != 15548:
        print(f"警告: 样本数异常 (期望 15548，实际 img={img_cnt}, std={std_cnt})！")
        return False
    return True


def push_to_remote():
    print(f"\n=== [2/4] 通过 SSH 流式管道推送 Shards 与 CSV 至远程 {REMOTE_HOST} ===")
    # 确保远程目录存在
    subprocess.run(["ssh", REMOTE_HOST, f"mkdir -p {REMOTE_ROOT}/data/supplements_glyph/shards_img {REMOTE_ROOT}/data/supplements_glyph/shards_std {REMOTE_ROOT}/assets"], check=True)
    print("  ✓ 远程目录准备完毕")

    # 构建传输文件清单
    files_to_send = [
        "data/supplements_glyph/shards_img",
        "data/supplements_glyph/shards_std",
        "assets/train_font_supplements_glyph15k.csv",
        "assets/train_50k_v2_augmented_glyph15k.csv"
    ]

    print(f"  正在流式传输: {' '.join(files_to_send)} ...")
    p1 = subprocess.Popen(["tar", "-cf", "-", *files_to_send], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["ssh", REMOTE_HOST, f"tar -xf - -C {REMOTE_ROOT}/"], stdin=p1.stdout)
    p1.stdout.close()
    p2.communicate()
    if p2.returncode != 0:
        raise subprocess.CalledProcessError(p2.returncode, "tar-pipe")
    print("  ✓ 传输并解压完成！")


def verify_remote():
    print(f"\n=== [3/4] 远程服务器文件落地校验 ===")
    res = subprocess.run(
        ["ssh", REMOTE_HOST, f"ls -lh {REMOTE_ROOT}/data/supplements_glyph/shards_img {REMOTE_ROOT}/data/supplements_glyph/shards_std {REMOTE_ROOT}/assets/*glyph15k.csv"],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace"
    )
    print(res.stdout)

    print("=== [4/4] 远程分片样本数精确核算 ===")
    remote_script = (
        "import glob, numpy as np; "
        "img_s = sorted(glob.glob('" + REMOTE_ROOT + "/data/supplements_glyph/shards_img/shard_*.npz')); "
        "std_s = sorted(glob.glob('" + REMOTE_ROOT + "/data/supplements_glyph/shards_std/shard_*.npz')); "
        "c_img = sum(len(np.load(s)['img_ids']) for s in img_s); "
        "c_std = sum(len(np.load(s)['img_ids']) for s in std_s); "
        "print(f'远程 img: {len(img_s)} 分片, {c_img} 样本 | std: {len(std_s)} 分片, {c_std} 样本')"
    )
    res2 = subprocess.run(
        ["ssh", REMOTE_HOST, f"/opt/conda/envs/cu121/bin/python -c \"{remote_script}\""],
        capture_output=True, text=True, check=True, encoding="utf-8", errors="replace"
    )
    print("  " + res2.stdout.strip())
    print("✓ 远程数据推送并校验 100% 成功！")




def main():
    if not check_local_shards():
        sys.exit(1)
    push_to_remote()
    verify_remote()


if __name__ == "__main__":
    main()
