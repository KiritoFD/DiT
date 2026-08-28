# -*- coding: utf-8 -*-
"""把 final_images 所有图预处理成 256x256（直接拉伸 INTER_CUBIC，匹配 latent 编码）。
输出 final_imgs_256/{img_id}.png。
GPU 加速批量处理。
"""
import os, sys, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, cv2, torch
from concurrent.futures import ThreadPoolExecutor

SRC = "final_images"
OUT = "final_imgs_256"
SIZE = 256
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(os.path.join(SRC, "*.png")))
print("files:", len(files))
done = 0

def read(p):
    buf = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

pool = ThreadPoolExecutor(max_workers=16)
BATCH = 128
t0 = time.time()
for i in range(0, len(files), BATCH):
    chunk = files[i:i+BATCH]
    futs = {p: pool.submit(read, p) for p in chunk}
    ids, imgs = [], []
    for p, f in futs.items():
        img = f.result()
        if img is None:
            continue
        # 直接拉伸到 256x256（INTER_CUBIC），与 latent 编码一致
        ids.append(os.path.splitext(os.path.basename(p))[0])
        imgs.append(cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))
    for pid, img in zip(ids, imgs):
        cv2.imencode(".png", img)[1].tofile(os.path.join(OUT, f"{pid}.png"))
    done += len(imgs)
    if (i//BATCH+1) % 20 == 0:
        el = time.time()-t0
        print(f"  {done}/{len(files)} {done/el:.0f}/s", flush=True)
el = time.time()-t0
print(f"Done {done} in {el:.0f}s ({done/el:.0f}/s)")
