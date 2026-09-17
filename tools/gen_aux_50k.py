# -*- coding: utf-8 -*-
"""为 50k 数据集生成 12ch 后训练需要的 aux 图：canny + 3px 骨架。

## 为什么必须重做
旧的 aux（`data/aux/aux_canny_latents_base`、`inst_skel_latents_px60`）是按
**旧 id** 建的（2259..266225 / 60008..60938），与 50k 的 0..51035 **不重叠**。
直接复用 -> 查表大量落空或**命中错行**（静默错位，不报错）。

## 生成规则（与 tools/gen_base_sym_aux.py 完全一致）
    skel3 = binary_dilation(skeletonize(ink), ST, iterations=1)   # 3px 骨架
    canny = cv2.Canny(gray, 80, 180)
⚠ **不能复用原图的 aux 图** —— 新数据经过描满/去噪，图变了，aux 监督必须跟着变，
  否则监督信号与图像内容不匹配。

## 资源
**纯 CPU**（scipy/cv2），多进程 + `nice -n 19`，不碰 GPU、不挤训练。
幂等：两个文件都在就跳过，可断点续跑。

用法:
    python tools/gen_aux_50k.py --nproc 16
"""
import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, generate_binary_structure

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())
try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None

SRC_CSV = "assets/train_50k_v2.csv"
SKEL_OUT = "data/50k/aux_skel3"
CANNY_OUT = "data/50k/aux_canny"


def iid_of(p):
    m = re.search(r"(\d+)\.png$", str(p))
    return int(m.group(1)) if m else None


def gen_one(task):
    """返回 (iid, 是否新生成)。"""
    iid, img_path = task
    sp = f"{SKEL_OUT}/{iid:06d}.png"
    cp = f"{CANNY_OUT}/{iid:06d}.png"
    if os.path.exists(sp) and os.path.exists(cp):
        return iid, False
    try:
        a = np.asarray(Image.open(img_path).convert("L"), dtype=np.uint8)
    except Exception as e:
        return iid, f"读图失败: {e}"
    import cv2
    # 3px 骨架：先二值化墨迹 -> skeletonize -> 膨胀 1 次
    ink = a < 128
    if skeletonize is not None:
        sk = skeletonize(ink)
    else:
        sk = ink
    ST = generate_binary_structure(2, 2)
    sk3 = binary_dilation(sk, ST, iterations=1)
    # ── 极性：统一成**白底(255)黑线(0)**，这是 doc58 修完之后的**新规范** ──
    #   `tools/gen_base_images.py` 已改成 `255 - cv2.Canny(a, 80, 180)`；
    #   而 data/aux/final_canny_base/*.png 是**修复前**的遗留（黑底白线，
    #   编码时靠 invert=True 补回来）。两者最终 latent 等价。
    #   我们直接产白底，编码时用 transform="gray"（不反转）-> 与旧 latent 一致。
    Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(sp)
    edges = 255 - cv2.Canny(a, 80, 180)
    Image.fromarray(edges, "L").save(cp)
    return iid, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=SRC_CSV)
    ap.add_argument("--nproc", type=int, default=16)
    a = ap.parse_args()

    for d in (SKEL_OUT, CANNY_OUT):
        os.makedirs(d, exist_ok=True)
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    tasks = [(iid_of(r["image_path"]), r["image_path"]) for r in rows]
    tasks = [(i, p) for i, p in tasks if i is not None]
    print(f"[aux50k] {a.csv}: {len(tasks)} 张, {a.nproc} 进程 (nice 19)", flush=True)

    t0 = time.time()
    done = new = 0
    errs = 0
    with mp.Pool(a.nproc, initializer=lambda: os.nice(19)) as pool:
        for iid, ok in pool.imap_unordered(gen_one, tasks, chunksize=32):
            done += 1
            if ok is True:
                new += 1
            elif ok is not False:
                errs += 1
                if errs <= 3:
                    print(f"  ⚠ {iid}: {ok}", flush=True)
            if done % 5000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)}  新生成 {new}  失败 {errs}  "
                      f"{done/el:.0f} img/s  已用 {el/60:.1f} 分钟", flush=True)

    print(f"[aux50k] DONE 处理 {done}, 新生成 {new}, 失败 {errs}, "
          f"耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
    print(f"  -> {SKEL_OUT}  ({len(os.listdir(SKEL_OUT))} 个)", flush=True)
    print(f"  -> {CANNY_OUT}  ({len(os.listdir(CANNY_OUT))} 个)", flush=True)


if __name__ == "__main__":
    main()
