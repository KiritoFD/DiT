# -*- coding: utf-8 -*-
"""gen_tj_images.py — 阶段A: 生成 tongji 全部派生图并落盘 (多进程, 快).

  data/skel/final_skel3_tj/{img_id}.png   — GT 图骨架化+3px (8,976 张)
  data/aux/final_canny_tj/{img_id}.png    — GT 图 canny 边缘 (8,976 张)
  data/skel/std_skel3_tj_png/{uid}.png    — 字体渲染标准字形 3px (8,555 个 (script,char))

均为确定性: 同输入同输出; 已存在则跳过 (幂等).
"""
import csv
import glob
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

try:
    from skimage.morphology import skeletonize
except ImportError:
    from scipy.ndimage import binary_erosion

    def skeletonize(b):
        skel = np.zeros_like(b)
        img = b.copy()
        st = generate_binary_structure(2, 2)
        while img.any():
            er = binary_erosion(img, structure=st)
            skel |= img & ~er
            img = er
        return skel

ST = generate_binary_structure(2, 2)
SKEL_OUT = "data/skel/final_skel3_tj"
CANNY_OUT = "data/aux/final_canny_tj"
STD_OUT = "data/skel/std_skel3_tj_png"


def gen_one(task):
    iid, path = task
    try:
        a = np.asarray(Image.open(path).convert("L"))
        if a.shape != (256, 256):
            a = np.asarray(Image.open(path).convert("L").resize((256, 256)))
        # skel3
        sk = skeletonize(a < 127)
        sk3 = binary_dilation(sk, ST, iterations=1)
        Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(
            f"{SKEL_OUT}/{iid}.png")
        # canny
        import cv2
        edges = cv2.Canny(a, 80, 180)
        Image.fromarray(edges, "L").save(f"{CANNY_OUT}/{iid}.png")
        return iid, None
    except Exception as e:
        return iid, str(e)


_FONT = {}


def render(ch, script):
    SCRIPT_FONT = {
        "楷": ["simkai.ttf", "STKAITI.TTF", "NotoSerifSC-VF.ttf"],
        "行": ["STXINGKA.TTF", "FZSTK.TTF"],
        "隶": ["SIMLI.TTF", "STLITI.TTF"],
    }
    cands = list(SCRIPT_FONT.get(script, SCRIPT_FONT["楷"])) + ["simkai.ttf", "simhei.ttf"]
    seen = set()
    for f in cands:
        if f in seen:
            continue
        seen.add(f)
        fp = os.path.join("tools/fonts", f)
        if not os.path.isfile(fp):
            continue
        try:
            font = _FONT.get(f) or ImageFont.truetype(fp, 200)
            _FONT[f] = font
        except Exception:
            continue
        img = Image.new("L", (256, 256), 255)
        ImageDraw.Draw(img).text((128, 128), ch, font=font, fill=0, anchor="mm")
        a = np.asarray(img)
        if (a < 250).sum() < 10:
            continue
        sk = skeletonize(a < 127)
        sk = binary_dilation(sk, ST, iterations=1)
        return np.where(sk, 0, 255).astype(np.uint8)
    return None


def gen_std(task):
    uid, script, ch = task
    arr = render(ch, script)
    if arr is None:
        return uid, "render-miss"
    Image.fromarray(arr, "L").save(f"{STD_OUT}/{uid}.png")
    return uid, None


def main():
    for d in (SKEL_OUT, CANNY_OUT, STD_OUT):
        os.makedirs(d, exist_ok=True)
    rows = list(csv.DictReader(open("assets/train_tongji_only.csv", encoding="utf-8")))
    tasks = []
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        if not os.path.exists(f"{SKEL_OUT}/{iid}.png") or not os.path.exists(f"{CANNY_OUT}/{iid}.png"):
            tasks.append((iid, r["image_path"]))
    print(f"[aux] {len(tasks)} images to generate (skel3+canny)", flush=True)
    t0 = time.time()
    n_fail = 0
    with mp.Pool(32) as pool:
        for k, (iid, err) in enumerate(pool.imap_unordered(gen_one, tasks, chunksize=32), 1):
            if err:
                n_fail += 1
                print(f"FAIL {iid}: {err}", flush=True)
            if k % 1000 == 0:
                print(f"  {k}/{len(tasks)} ({k/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[aux] done: {len(tasks)-n_fail} ok, {n_fail} fail, {time.time()-t0:.0f}s", flush=True)

    # std skel 渲染 (per (script,char) 唯一)
    all_rows = list(csv.DictReader(open("assets/train_fame_tj_kxl.csv", encoding="utf-8")))
    pairs = sorted({(r["script"], r["character"]) for r in all_rows
                    if "calli_tongji" in r["image_path"]})
    # uid 段: 8600000+ (与 tongji img 950000+/sym 98-99M 段错开)
    key2uid = {}
    for i, (script, ch) in enumerate(pairs):
        key2uid[(script, ch)] = 8600000 + i

    tasks2 = []
    for (script, ch) in pairs:
        uid = key2uid[(script, ch)]
        if not os.path.exists(f"{STD_OUT}/{uid}.png"):
            tasks2.append((uid, script, ch))
    print(f"[std] {len(tasks2)} to render (of {len(pairs)})", flush=True)
    t0 = time.time()
    n_fail = 0

    with mp.Pool(32) as pool:
        for k, (uid, err) in enumerate(pool.imap_unordered(gen_std, tasks2, chunksize=16), 1):
            if err:
                n_fail += 1
            if k % 500 == 0:
                print(f"  {k}/{len(tasks2)} ({k/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[std] {len(tasks2)-n_fail} ok, {n_fail} fail, {time.time()-t0:.0f}s", flush=True)
    # uid 映射存盘 (encode 阶段 img_id->uid 用)
    with open("data/skel/std_skel3_tj_key2uid.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["script", "character", "uid"])
        for (script, ch), uid in sorted(key2uid.items()):
            w.writerow([script, ch, uid])
    print("PHASE-A DONE", flush=True)


if __name__ == "__main__":
    main()
