# -*- coding: utf-8 -*-
"""把本地 latent_missing/*.npy 打包成 tar.gz（保留相对 latent_missing/ 的路径）。"""
import os, sys, tarfile, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

files = sorted(glob.glob("latent_missing/*.npy"))
print("files:", len(files))
out = "latent_missing.tar.gz"
with tarfile.open(out, "w:gz") as tar:
    for f in files:
        tar.add(f, arcname=os.path.basename(f))
sz = os.path.getsize(out)
print(f"packed {out} size {sz/1e6:.1f} MB")
