#!/usr/bin/env python3
"""远程: 读取路径清单, 复制图片到 /tmp/dino_samples/ 并打包"""
import os, shutil, tarfile

REMOTE_BASE = "/root/Workspace/xy/DiT"
PATH_FILE = "/tmp/_paths_ok.txt"
OUT_DIR = "/tmp/dino_samples"
TAR_PATH = "/tmp/dino_samples.tar.gz"

os.makedirs(OUT_DIR, exist_ok=True)
# 清空旧文件
for f in os.listdir(OUT_DIR):
    os.remove(os.path.join(OUT_DIR, f))

with open(PATH_FILE, "rb") as f:
    data = f.read()
# 去BOM和CRLF
data = data.replace(b"\xef\xbb\xbf", b"").replace(b"\r\n", b"\n").replace(b"\r", b"\n")
paths = [p.strip() for p in data.decode("utf-8").splitlines() if p.strip()]

copied = 0
for p in paths:
    full = os.path.join(REMOTE_BASE, p)
    if os.path.isfile(full):
        shutil.copy2(full, os.path.join(OUT_DIR, os.path.basename(p)))
        copied += 1
    else:
        print(f"MISSING: {p}")

print(f"copied {copied}/{len(paths)} files")

with tarfile.open(TAR_PATH, "w:gz") as tar:
    tar.add(OUT_DIR, arcname=".")
print(f"tar size: {os.path.getsize(TAR_PATH)/1e6:.1f}MB")
