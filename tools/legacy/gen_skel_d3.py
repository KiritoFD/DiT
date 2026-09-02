# -*- coding: utf-8 -*-
"""
gen_skel_d3.py — 从 1px GT skel 生成 3px 膨胀版, 存到 final_skeleton_d3/。

用 scipy ndimage.binary_dilation (圆形结构元, r=3) 对每张 256×256 二值 skel 做膨胀。
多进程并行, 顺序写。输入 final_skeleton/*.png (L 模式, 0/255)。

用法 (远程 CPU 后台):
  /opt/conda/bin/python tools/gen_skel_d3.py --in-dir final_skeleton --out-dir final_skeleton_d3 --workers 16
"""
import os, sys, time, argparse, glob, re
import numpy as np
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")


def dilate_one(path_in, path_out, r=3):
    img = np.asarray(Image.open(path_in).convert("L")) > 127  # bool
    if r > 0:
        from scipy.ndimage import binary_dilation, generate_binary_structure
        # 圆形结构元: 先用 cross (rank=2) 再迭代 r 次近似圆
        se = generate_binary_structure(2, 2)  # 3×3 十字+对角 = 8 邻域
        out = binary_dilation(img, structure=se, iterations=r)
    else:
        out = img
    arr = (out.astype(np.uint8)) * 255
    Image.fromarray(arr, mode="L").save(path_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="final_skeleton")
    ap.add_argument("--out-dir", default="final_skeleton_d3")
    ap.add_argument("--r", type=int, default=3, help="膨胀半径 px")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.png")))
    n = len(files)
    print(f"[gen] {n} files, r={args.r}, out={args.out_dir}", flush=True)

    # 跳过已生成的
    todo = []
    for f in files:
        base = os.path.basename(f)
        out = os.path.join(args.out_dir, base)
        if not os.path.exists(out):
            todo.append((f, out, args.r))
    print(f"[gen] todo: {len(todo)} (skip {n - len(todo)} existing)", flush=True)

    import multiprocessing as mp
    t0 = time.time()
    done = 0
    with mp.Pool(args.workers) as pool:
        for _ in pool.starmap(dilate_one, todo, chunksize=256):
            done += 1
            if done % 50000 == 0:
                print(f"[gen] {done}/{len(todo)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"[gen] DONE {done} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()