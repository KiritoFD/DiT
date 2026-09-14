# -*- coding: utf-8 -*-
"""
gen_base_sym_aux.py — 为 **增强臂** 的新增图生成 aux 目标图 (skel3 / canny).

增强图 (train_base_sym.csv 中 aug ∈ {tp,tn}, uid 7000000+/7100000+) 需要自己的
skel3 与 canny —— 它们由原图经形态学变换得到, 骨架/边缘与原图不同, 必须重算
(不能复用原图的 aux 图, 否则 aux 监督与图像内容不匹配).

生成规则与 tools/gen_base_images.py 的 gen_one 完全一致:
  skel3 = binary_dilation(skeletonize(ink), ST, iterations=1)   (3px 骨架)
  canny = cv2.Canny(a, 80, 180)
输出目录与 base 线 **相同** (按 img_id 命名, 增强 uid 不冲突):
  data/skel/final_skel3_base/{uid}.png
  data/aux/final_canny_base/{uid}.png
幂等: 两个文件都存在则跳过.
"""
import csv
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

try:
    from skimage.morphology import skeletonize
except ImportError:                                    # 无 skimage 时的等价实现
    from scipy.ndimage import binary_erosion

    def skeletonize(b):
        skel = np.zeros_like(b)
        img = b.copy()
        st = generate_binary_structure(2, 2)
        while img.any():
            e = binary_erosion(img, structure=st)
            skel |= img & ~e
            img = e
        return skel

ST = generate_binary_structure(2, 2)
SRC_CSV = "assets/train_base_sym.csv"
SKEL_OUT = "data/skel/final_skel3_base"
CANNY_OUT = "data/aux/final_canny_base"
NPROC = 48


def gen_one(task):
    iid, path = task
    try:
        im = Image.open(path).convert("L")
        a = np.asarray(im)
        ink = a < 128
        sk3 = binary_dilation(skeletonize(ink), ST, iterations=1)
        Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(
            f"{SKEL_OUT}/{iid}.png")
        import cv2
        edges = cv2.Canny(a, 80, 180)
        Image.fromarray(edges, "L").save(f"{CANNY_OUT}/{iid}.png")
        return iid, None
    except Exception as e:
        return iid, str(e)


def main():
    for d in (SKEL_OUT, CANNY_OUT):
        os.makedirs(d, exist_ok=True)
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    tasks = []
    for r in rows:
        if not r.get("aug"):                    # 只处理新增的增强行
            continue
        m = re.search(r"(\d+)\.png", r["image_path"])
        if not m:
            continue
        iid = int(m.group(1))
        if (os.path.exists(f"{SKEL_OUT}/{iid}.png")
                and os.path.exists(f"{CANNY_OUT}/{iid}.png")):
            continue                            # 幂等
        tasks.append((iid, r["image_path"]))
    print(f"[aux-sym] {len(tasks)} aug images to generate", flush=True)
    if not tasks:
        print("[aux-sym] nothing to do", flush=True)
        return
    t0 = time.time()
    n_fail = 0
    with mp.Pool(NPROC) as pool:
        for n, (iid, err) in enumerate(pool.imap_unordered(gen_one, tasks, chunksize=64), 1):
            if err:
                n_fail += 1
                if n_fail <= 5:
                    print(f"  FAIL {iid}: {err}", flush=True)
            if n % 20000 == 0:
                print(f"  {n}/{len(tasks)} ({n / (time.time() - t0):.0f}/s)", flush=True)
    print(f"[aux-sym] done: {len(tasks) - n_fail} ok, {n_fail} fail, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
