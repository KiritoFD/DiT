# -*- coding: utf-8 -*-
"""unzip_calli_tongji.py - extract zip with GBK names to data/imgs/calli_tongji/{书家-书体}/{字}.png"""
import os
import sys
import zipfile

ZIP = sys.argv[1] if len(sys.argv) > 1 else "data/Calli-Tongji.zip"
OUT = sys.argv[2] if len(sys.argv) > 2 else "data/imgs/calli_tongji"
os.makedirs(OUT, exist_ok=True)
z = zipfile.ZipFile(ZIP)
n_saved = 0
for zi in z.infolist():
    if zi.is_dir():
        continue
    try:
        name = zi.filename.encode("cp437").decode("gbk")
    except Exception:
        name = zi.filename
    parts = name.split("/")
    if len(parts) < 3 or not parts[-1].endswith(".png"):
        continue
    callig_script = parts[1]
    ch = parts[-1][:-4]
    d = os.path.join(OUT, callig_script)
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f"{ch}.png")
    if not os.path.exists(dst):
        with z.open(zi) as f, open(dst, "wb") as fo:
            fo.write(f.read())
        n_saved += 1
print(f"extracted {n_saved} images to {OUT}")
