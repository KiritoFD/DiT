# -*- coding: utf-8 -*-
"""
高性能 Step1：final_images -> final_imgs_256/{img_id}.png（256x256, INTER_CUBIC）。
GPU 批量 F.interpolate + CPU 多线程 IO，拉满。
"""
import os, sys, glob, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np, cv2, torch
import torch.nn.functional as F
from concurrent.futures import ThreadPoolExecutor

SRC = "final_images"
OUT = "final_imgs_256"
SIZE = 256
os.makedirs(OUT, exist_ok=True)

IO = int(os.environ.get("IO_WORKERS", "32"))
BATCH = int(os.environ.get("GPU_BATCH", "256"))

def read(p):
    buf = np.fromfile(p, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def main():
    files = sorted(glob.glob(os.path.join(SRC, "*.png")))
    print("files:", len(files))
    # 增量：跳过已存在的输出
    todo = [p for p in files if not os.path.exists(os.path.join(OUT, os.path.splitext(os.path.basename(p))[0] + ".png"))]
    print("todo (增量):", len(todo), "already done:", len(files) - len(todo))
    device = "cuda"
    done = 0
    skipped = len(files) - len(todo)
    pool = ThreadPoolExecutor(max_workers=IO)
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i+BATCH]
        futs = {p: pool.submit(read, p) for p in chunk}
        raw = []  # (pid, img)
        for p, f in futs.items():
            img = f.result()
            if img is None:
                continue
            raw.append((os.path.splitext(os.path.basename(p))[0], img))
        if not raw:
            continue
        # 按尺寸分组，同尺寸批量 GPU resize
        buckets = {}
        for pid, img in raw:
            buckets.setdefault(img.shape[:2], []).append((pid, img))
        for (h, w), items in buckets.items():
            t = torch.from_numpy(np.stack([it[1] for it in items]))  # (B,H,W,3) uint8
            t = t.permute(0, 3, 1, 2).float().to(device) / 255.0
            r = F.interpolate(t, size=(SIZE, SIZE), mode="bicubic", align_corners=False)
            r = (r.clamp(0, 1) * 255.0).byte().permute(0, 2, 3, 1).cpu().numpy()
            for (pid, _), arr in zip(items, r):
                cv2.imencode(".png", arr)[1].tofile(os.path.join(OUT, f"{pid}.png"))
        done += len(raw)
        if (i//BATCH+1) % 40 == 0:
            el = time.time()-t0
            print(f"  {done}/{len(files)} {done/el:.0f}/s", flush=True)
    el = time.time()-t0
    print(f"Done {done} in {el:.0f}s ({done/el:.0f}/s)")

if __name__ == "__main__":
    main()
