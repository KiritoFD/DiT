# -*- coding: utf-8 -*-
"""
gen_base_images.py — 阶段A(base): 全部 skel3/canny/std-skel 图片生成落盘 (多进程, 幂等).

对 assets/train_base_noaug.csv 全部行 (54,892):
  data/skel/final_skel3_base/{img_id}.png  — GT 图骨架化+3px
  data/aux/final_canny_base/{img_id}.png   — GT 图 canny 边缘
std skel: 全部唯一 (script,char) 未被 fame 现有 shard 覆盖的:
  data/skel/std_skel3_base_png/{uid}.png + data/skel/std_skel3_base_key2uid.csv
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
SKEL_OUT = "data/skel/final_skel3_base"
CANNY_OUT = "data/aux/final_canny_base"
STD_OUT = "data/skel/std_skel3_base_png"
_KEY2UID = "data/skel/std_skel3_base_key2uid.csv"


def gen_one(task):
    iid, path = task
    try:
        im = Image.open(path).convert("L")
        a = np.asarray(im)
        if a.shape != (256, 256):
            a = np.asarray(im.resize((256, 256)))
        sk = skeletonize(a < 127)
        sk3 = binary_dilation(sk, ST, iterations=1)
        Image.fromarray(np.where(sk3, 0, 255).astype(np.uint8), "L").save(f"{SKEL_OUT}/{iid}.png")
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
    rows = list(csv.DictReader(open("assets/train_base_noaug.csv", encoding="utf-8")))
    print(f"base rows: {len(rows)}", flush=True)

    # ── skel3 + canny (全部行, 幂等) ──────────────────────────────────────
    tasks = []
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        if not os.path.exists(f"{SKEL_OUT}/{iid}.png") or not os.path.exists(f"{CANNY_OUT}/{iid}.png"):
            tasks.append((iid, r["image_path"]))
    print(f"[aux] {len(tasks)} images to generate", flush=True)
    t0 = time.time()
    n_fail = 0
    with mp.Pool(48) as pool:
        for k, (iid, err) in enumerate(pool.imap_unordered(gen_one, tasks, chunksize=64), 1):
            if err:
                n_fail += 1
                print(f"FAIL {iid}: {err}", flush=True)
            if k % 5000 == 0:
                print(f"  {k}/{len(tasks)} ({k/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[aux] done: {len(tasks)-n_fail} ok, {n_fail} fail, {time.time()-t0:.0f}s", flush=True)

    # ── std skel: 唯一 (script,char), 未覆盖的渲染 ────────────────────────
    # fame 现有覆盖: 按 img_id (fame3-sym 增强 img id + 原始 id) 在 std_skel3_latents_fame_sym
    pairs = sorted({(r["script"], r["character"]) for r in rows})
    covered = set()
    for sp in glob.glob("data/skel/std_skel3_latents_fame_sym/shard_*.npz"):
        with np.load(sp) as d:
            covered.update(int(i) for i in d["img_ids"])
    id2key = {}
    for r in rows:
        iid = int(re.search(r"(\d+)\.png", r["image_path"]).group(1))
        id2key[iid] = (r["script"], r["character"])
    need = []
    key2uid = {}
    uid = 8600000
    for (script, ch) in pairs:
        key2uid[(script, ch)] = uid
        uid += 1
    # 该 (script,char) 是否已被 fame 覆盖: 检查该 key 的任一 img_id 是否在 covered
    for iid, key in id2key.items():
        if iid in covered:
            continue
        need.append((key2uid[key], key[0], key[1]))
    # 去重 uid + 已存在 PNG 跳过
    need = sorted(set(need))
    tasks2 = [(u, s, c) for (u, s, c) in need if not os.path.exists(f"{STD_OUT}/{u}.png")]
    print(f"[std] {len(tasks2)} to render (pairs={len(pairs)}, covered_rows={len(covered)})", flush=True)
    t0 = time.time()
    n_fail = 0
    with mp.Pool(48) as pool:
        for k, (uid, err) in enumerate(pool.imap_unordered(gen_std, tasks2, chunksize=16), 1):
            if err:
                n_fail += 1
            if k % 500 == 0:
                print(f"  {k}/{len(tasks2)} ({k/(time.time()-t0):.0f}/s)", flush=True)
    print(f"[std] {len(tasks2)-n_fail} ok, {n_fail} fail, {time.time()-t0:.0f}s", flush=True)
    with open(_KEY2UID, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["script", "character", "uid"])
        for (script, ch), u in sorted(key2uid.items()):
            w.writerow([script, ch, u])
    print("PHASE-A DONE", flush=True)


if __name__ == "__main__":
    main()
