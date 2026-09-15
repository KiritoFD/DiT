# -*- coding: utf-8 -*-
"""
aug_base_sym.py — base 数据集的 **v4 对称笔画粗细增强** (增强臂数据).

v4 定义 (与 tools/aug/aug_renders_v4.py 一致):
  对每张原图生成一对 **±同幅** 变体 (dilate p 次 / erode p 次, 同一个 p):
      tp (thicken) = binary_dilation(ink, p)
      tn (thin)    = binary_erosion(ink, p)
  两变体关于原图对称 -> 单图均值不变, 全局均值不变 (v3.3 朝中值收敛的偏差已修).
  p ∈ {1,2} 按 idx 哈希确定 (同一图的 ± 对用同一个 p).
  保护: tn 过度腐蚀(面积<15% 或 <20px) -> 减 p; 仍不可 -> 跳过该变体.
        tp 面积>3x -> 减 p.

输入: assets/train_base_noaug.csv  (54,892 行: fame 27,552 + tongji 2,092 + UniCalli 25,248)
输出: data/imgs/base_sym/{uid}.png   (tp: 7000000+idx / tn: 7100000+idx)
      assets/train_base_sym.csv      (orig + tp + tn, ≈ 164k 行, 含 aug 列)

uid 段选择: 现有占用为 fame原始/增强 900000·1800000、tongji 950000-952992 与
9800000·9900000、UniCalli 960000-987253 -> 7000000/7100000 无冲突.

幂等: 已存在 PNG 直接跳过 (可安全重跑).
字段: **继承源 csv 全部列**, 仅追加 aug 列 ('' / tp / tn) —— 不硬编码字段,
      避免源 csv 缺列时 KeyError.
"""
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir(ROOT)

ST = generate_binary_structure(2, 2)
SRC_CSV = "assets/train_base_noaug.csv"
OUT_DIR = "data/imgs/base_sym"
OUT_CSV = "assets/train_base_sym.csv"
UID_T = 7000000
UID_N = 7100000
NPROC = 48


def work(task):
    """task=(idx, img_path, uid_t, uid_n) -> (idx, {'tp':uid|None,'tn':uid|None})"""
    idx, img_path, uid_t, uid_n = task
    uid_base = {"tp": uid_t, "tn": uid_n}
    out = {"tp": None, "tn": None}
    # 幂等: 两个变体都已存在则直接返回
    if all(os.path.exists(os.path.join(OUT_DIR, f"{uid_base[k] + idx}.png"))
           for k in ("tp", "tn")):
        return idx, {"tp": uid_base["tp"] + idx, "tn": uid_base["tn"] + idx}
    try:
        g = np.asarray(Image.open(img_path).convert("L"))
    except Exception as e:
        return idx, f"FAIL {img_path}: {e}"
    ink = g < 128
    if ink.sum() < 20:
        return idx, out
    p = 1 + ((idx * 2654435761) % 2)          # 确定性 p ∈ {1,2}
    for kind, op in (("tp", binary_dilation), ("tn", binary_erosion)):
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
            continue
        uid = uid_base[kind] + idx
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(
            os.path.join(OUT_DIR, f"{uid}.png"))
        out[kind] = uid
    return idx, out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = list(csv.DictReader(open(SRC_CSV, encoding="utf-8")))
    fields = list(rows[0].keys())
    if "aug" not in fields:
        fields = fields + ["aug"]
    print(f"src rows: {len(rows)}  fields: {fields}", flush=True)

    # ⚠ 这里必须传**基址** (UID_T / UID_N), 不能传 UID_T + k —— work() 里还会
    #   再 +idx, 传 UID_T+k 会变成双重叠加 (= UID_T + 2k)。k>49999 时
    #   7000000+2k 溢出到 tn 段(7100000+), 把 tn 的图覆盖掉: 实测 4892 个 uid 冲突,
    #   csv 里 4292 个 img_id 重复(8584 行 = 5.41%)指向错误的图。
    #   (2026-09-15 定位并修正)
    tasks = [(k, r["image_path"], UID_T, UID_N) for k, r in enumerate(rows)]
    t0 = time.time()
    results = {}
    with mp.Pool(NPROC) as pool:
        for n, (idx, out) in enumerate(pool.imap_unordered(work, tasks, chunksize=64), 1):
            results[idx] = out
            if n % 10000 == 0:
                print(f"  {n}/{len(tasks)} ({n / (time.time() - t0):.0f}/s)", flush=True)

    n_t = sum(1 for v in results.values() if isinstance(v, dict) and v.get("tp"))
    n_n = sum(1 for v in results.values() if isinstance(v, dict) and v.get("tn"))
    n_fail = sum(1 for v in results.values() if isinstance(v, str))
    print(f"aug done: tp={n_t} tn={n_n} fail={n_fail} ({time.time() - t0:.0f}s)", flush=True)

    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for k, r in enumerate(rows):
            r0 = dict(r)
            r0["aug"] = ""                     # 原图
            w.writerow(r0)
            v = results.get(k)
            if not isinstance(v, dict):
                continue
            for kind in ("tp", "tn"):
                uid = v.get(kind)
                if not uid:
                    continue
                r2 = dict(r)                   # 继承全部原始字段
                r2["image_path"] = f"{OUT_DIR}/{uid}.png"
                r2["aug"] = kind
                w.writerow(r2)
    n_rows = sum(1 for _ in open(OUT_CSV, encoding="utf-8")) - 1
    print(f"written {OUT_CSV}: {n_rows} rows "
          f"(orig {len(rows)} + aug {n_rows - len(rows)})", flush=True)


if __name__ == "__main__":
    main()
