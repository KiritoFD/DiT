#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 5script/train_top6.csv 的图片打包成单个 .npy (uint8 N,H,W,3)。
   顺序顺序读(顺序 IO 快), 支持 mmap 随机访问; canny/skel 单通道同样打包。
   输出: 5script/pixmap/ 下 imgs.npy / cannys.npy / skels.npy + ids.txt
"""
import os, sys, csv, re, time
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")
BASE = "/root/Workspace/xy/DiT/"
CSV = BASE + "5script/train_top6.csv"
IMG_ROOT = BASE + "final_imgs_256"
CAN_ROOT = BASE + "final_canny"
SKEL_ROOT = BASE + "final_skeleton"
OUT = BASE + "5script/pixmap"
os.makedirs(OUT, exist_ok=True)

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
ids = [int(re.search(r"(\d+)\.png", r["image_path"]).group(1)) for r in rows]
n = len(rows)
print(f"rows={n}", flush=True)

def pack(path, shape, reader, load_3ch=True):
    if os.path.exists(path):
        print(f"skip existing {path}", flush=True)
        return
    t0 = time.time()
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.uint8, shape=shape)
    for i, img_id in enumerate(ids):
        arr[i] = reader(img_id)
        if (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n} {time.time()-t0:.0f}s", flush=True)
    arr.flush()
    del arr
    print(f"wrote {path} in {time.time()-t0:.0f}s", flush=True)

pack(OUT + "/imgs.npy", (n, 256, 256, 3),
     lambda iid: np.asarray(Image.open(os.path.join(IMG_ROOT, f"{iid}.png")).convert("RGB")), True)
pack(OUT + "/cannys.npy", (n, 256, 256),
     lambda iid: np.asarray(Image.open(os.path.join(CAN_ROOT, f"{iid}.png")).convert("L")), False)
pack(OUT + "/skels.npy", (n, 256, 256),
     lambda iid: np.asarray(Image.open(os.path.join(SKEL_ROOT, f"{iid}.png")).convert("L")), False)

with open(OUT + "/ids.txt", "w") as f:
    f.write("\n".join(str(i) for i in ids))
print("ids.txt written", flush=True)
print("ALL DONE", flush=True)