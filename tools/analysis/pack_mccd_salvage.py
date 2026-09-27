#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/analysis/pack_mccd_salvage.py — 用原生 tarfile 安全流式打包 4,301 张图片 (RAM < 30MB)"""
import os
import sys
import tarfile

sys.stdout.reconfigure(encoding="utf-8")

LIST_FILE = "assets/mccd_tar_filelist.txt"
TAR_FILE = "assets/mccd_salvage_4301.tar.gz"

def main():
    with open(LIST_FILE, "r", encoding="utf-8") as f:
        paths = [line.strip() for line in f if line.strip()]

    print(f"开始打包 {len(paths)} 个文件 -> {TAR_FILE} ...")
    success = 0
    with tarfile.open(TAR_FILE, "w:gz") as tar:
        for idx, p in enumerate(paths):
            if os.path.exists(p):
                # 存为相对路径
                tar.add(p, arcname=p)
                success += 1
            if (idx + 1) % 1000 == 0 or (idx + 1) == len(paths):
                print(f"  ... 已打包 {idx+1}/{len(paths)}")

    sz_mb = os.path.getsize(TAR_FILE) / 1024 / 1024
    print(f"🎉 打包完成！成功收录 {success} 个文件，压缩包大小: {sz_mb:.2f} MB")

if __name__ == "__main__":
    main()
