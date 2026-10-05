import os
import shutil
import sys
import time

print("=" * 80)
print("【4090 磁盘整理与旧实验归档归并】")
print("=" * 80)

dit_root = "/root/Workspace/xy/DiT"
archive_dir = os.path.join(dit_root, "archive_experiments")
os.makedirs(archive_dir, exist_ok=True)

# 1. 整理旧实验目录并保留软链接（确保历史文档/脚本路径不损坏）
moves = [
    (
        os.path.join(dit_root, "data/archive/results_legacy"),
        os.path.join(archive_dir, "results_legacy_s1_to_s32"),
    ),
    (
        os.path.join(dit_root, "_archive"),
        os.path.join(archive_dir, "_archive_historical"),
    ),
    (
        os.path.join(dit_root, "archive_20260903"),
        os.path.join(archive_dir, "archive_20260903"),
    ),
    (
        os.path.join(dit_root, "assets/results/_archive"),
        os.path.join(archive_dir, "results_archive_failed"),
    ),
]

for src, dst in moves:
    if os.path.exists(src) and not os.path.islink(src):
        print(f"-> 移动: {src} -> {dst}")
        try:
            if os.path.exists(dst):
                # 目标存在时合并
                for item in os.listdir(src):
                    s_item = os.path.join(src, item)
                    d_item = os.path.join(dst, item)
                    if not os.path.exists(d_item):
                        shutil.move(s_item, d_item)
                shutil.rmtree(src)
            else:
                shutil.move(src, dst)
            # 建立软链接以防旧脚本找不到路径
            os.symlink(dst, src)
            print(f"   已完成移动并在原位置保留软链接: {src} -> {dst}")
        except Exception as e:
            print(f"   [警告] 移动 {src} 遇到错误: {e}")

# 2. 清理临时废弃 staging 文件
temp_files_to_remove = [
    "/root/Workspace/xy/calligraphy_raw_412k.tar",
    "/root/Workspace/xy/raw_dataset_transfer",
]

for p in temp_files_to_remove:
    if os.path.exists(p):
        print(f"-> 清理临时 staging 冗余文件: {p} ...")
        try:
            if os.path.isdir(p):
                shutil.rmtree(p)
            else:
                os.remove(p)
            print(f"   已安全删除: {p}")
        except Exception as e:
            print(f"   删除 {p} 错误: {e}")

print("=" * 80)
print(f"归档目录统一整理至: {archive_dir}")
for d in os.listdir(archive_dir):
    p = os.path.join(archive_dir, d)
    print(f"  - {d}")
print("=" * 80)
