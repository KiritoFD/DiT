# -*- coding: utf-8 -*-
"""aug_renders_v4.py — 墨迹增强 v4: **对称加粗/变细** (均值不变).

相对 v3.3 的修正:
  v3.3  target = K*(mid-w) 朝中值收敛 -> 会把笔画宽度分布整体拉向 mid, 均值漂移.
  v4    对每张图生成一对 **±同幅** 变体 (dilate p 次 / erode p 次, 同一个 p):
          thicken: binary_dilation(ink, p)
          thin   : binary_erosion(ink, p)
        两变体关于原图对称 -> 单图均值不变, 全局均值不变.
  p ∈ {1,2} 按 id 哈希确定 (同一图的 ± 对用同一个 p, 保证对称).
  保护: thin 过度腐蚀(面积<15% 或 <20px) -> 减 p; 仍不可 -> 跳过该变体.

产物:
  5script/fame3-sym/<uid>.png             (uid = 900000+idx 加粗 / 1800000+idx 变细)
  5script/train_fame3_sym_full.csv        (orig + 变体, 新增 aug 列: '' / tp / tn)
"""
import csv
import os
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

ROOT = "/root/Workspace/xy/DiT"
SRC_CSV = f"{ROOT}/5script/train_fame3_clean_v8.csv"
OUT_DIR = f"{ROOT}/5script/fame3-sym"
OUT_CSV = f"{ROOT}/5script/train_fame3_sym_full.csv"
ST = generate_binary_structure(2, 2)
UID_T = 900000
UID_N = 1800000
sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(OUT_DIR, exist_ok=True)


def work(task):
    idx, r = task
    src = os.path.join(ROOT, r["image_path"])
    try:
        g = np.asarray(Image.open(src).convert("L"))
    except Exception as e:
        return idx, f"FAIL {src}: {e}"
    ink = g < 128
    if ink.sum() < 20:
        return idx, {"tp": None, "tn": None}
    p = 1 + ((idx * 2654435761) % 2)  # deterministic p in {1,2}
    out = {}
    for kind, base, op in (("tp", UID_T, binary_dilation), ("tn", UID_N, binary_erosion)):
        pp, ok = p, False
        while pp > 0:
            m = op(ink, ST, iterations=pp)
            a = int(m.sum())
            if kind == "tn" and (a < 0.15 * int(ink.sum()) or a < 20):
                pp -= 1
                continue
            if kind == "tp" and a > 3.0 * int(ink.sum()):
                pp -= 1
                continue
            ok = True
            break
        if not ok:
            out[kind] = None
            continue
        uid = base + idx
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(f"{OUT_DIR}/{uid}.png")
        out[kind] = uid
    return idx, out


if __name__ == "__main__":
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    fields = list(rows[0].keys())
    tasks = list(enumerate(rows))
    res = {}
    with Pool(48) as pool:
        for n, (idx, out) in enumerate(pool.imap_unordered(work, tasks, chunksize=64), 1):
            res[idx] = out
            if n % 5000 == 0:
                print(f"{n}/{len(tasks)}", flush=True)
    n_t = sum(1 for v in res.values() if isinstance(v, dict) and v["tp"])
    n_n = sum(1 for v in res.values() if isinstance(v, dict) and v["tn"])
    print(f"done: thicken={n_t} thin={n_n} fails={sum(1 for v in res.values() if isinstance(v,str))}", flush=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + ["aug"])
        w.writeheader()
        for i, r in enumerate(rows):
            r0 = dict(r); r0["aug"] = ""
            w.writerow(r0)
            v = res.get(i)
            if not isinstance(v, dict):
                continue
            for kind in ("tp", "tn"):
                if v.get(kind) is None:
                    continue
                r2 = dict(r)
                r2["image_path"] = f"5script/fame3-sym/{v[kind]}.png"
                r2["aug"] = kind
                w.writerow(r2)
    print(f"{OUT_CSV}: {sum(1 for _ in open(OUT_CSV, encoding='utf-8')) - 1} rows", flush=True)
