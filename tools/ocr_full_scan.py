"""rapidocr 全量扫描 GT 图 —— **保存全部结果和置信度**（不只不一致的）。

用户要求（2026-09-21）:
  "先全量跑 rapid ocr，置信度也保留下来原始数据，后面我们对低置信度的用 VLM"

输出: assets/ocr_full_scan.csv
    idx, image_path, script, calligrapher, char_csv, char_ocr, conf,
    exact, relation
  relation ∈ {same, 简繁, other}
  exact: OCR 是否与 csv 完全一致

特性:
  - 支持断点续跑（每 --chunk 条 flush 一次，已完成的 idx 跳过）
  - conf 全量保留（后续按置信度分层，低置信度交给 VLM 精查）
  - nice 19（不抢 v15b 训练的 CPU）
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0, help="0=全部")
ap.add_argument("--script", default="all")
ap.add_argument("--chunk", type=int, default=1000, help="每多少条 flush 一次")
ap.add_argument("--out", default="assets/ocr_full_scan.csv")
a = ap.parse_args()

from opencc import OpenCC

t2s = OpenCC("t2s")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
if a.script != "all":
    rows = [r for r in rows if r["script"] == a.script]
if a.limit:
    rows = rows[:a.limit]
print(f"  总样本: {len(rows)}", flush=True)

# 断点续跑
done = set()
if os.path.exists(a.out):
    try:
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            done.add(int(r["idx"]))
    except Exception:
        pass
print(f"  已完成: {len(done)}", flush=True)

COLS = ["idx", "image_path", "script", "calligrapher", "char_csv",
        "char_ocr", "conf", "exact", "relation"]


def pad_square(im, pad=0.4):
    w_, h_ = im.size
    s = int(max(w_, h_) * (1 + pad * 2))
    c = Image.new("RGB", (s, s), "white")
    c.paste(im, ((s - w_) // 2, (s - h_) // 2))
    return c


# ★ 多进程 worker —— 单进程只有 2.0/s（nice19 + CPU 竞争），
#   多进程能提速 5-8 倍（每进程一个 onnx 实例，模型很小）。
_OCR = None


def _init():
    global _OCR
    from rapidocr_onnxruntime import RapidOCR
    _OCR = RapidOCR()


def _work(item):
    i, img_path, script, cal, truth = item
    src = img_path if os.path.isabs(img_path) else os.path.join(
        "/root/Workspace/xy/DiT", img_path)
    pred, conf = "?", 0.0
    if os.path.exists(src):
        try:
            res, _ = _OCR(np.array(pad_square(Image.open(src).convert("RGB"))))
            if res:
                pred = "".join(x[1] for x in res)[:1] or "?"
                conf = float(res[0][2]) if len(res[0]) > 2 else 0.0
        except Exception:
            pass
    exact = int(pred == truth)
    if exact:
        rel = "same"
    elif t2s.convert(pred) == t2s.convert(truth):
        rel = "简繁"
    else:
        rel = "other"
    return [i, img_path, script, cal, truth, pred, round(conf, 4), exact, rel]


if __name__ == "__main__":
    import multiprocessing as mp

    todo = [(i, r["image_path"], r["script"], r["calligrapher"], r["character"])
            for i, r in enumerate(rows) if i not in done]
    print(f"  待处理: {len(todo)}  进程数: {a.workers}", flush=True)

    from time import time

    t0 = time()
    fh = open(a.out, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if not os.path.exists(a.out) or os.path.getsize(a.out) == 0:
        w.writerow(COLS)
        fh.flush()

    n_new = 0
    with mp.Pool(a.workers, initializer=_init) as pool:
        for out in pool.imap_unordered(_work, todo, chunksize=16):
            w.writerow(out)
            n_new += 1
            if n_new % a.chunk == 0:
                fh.flush()
                el = time() - t0
                sp = n_new / max(el, 1e-9)
                print(f"    {len(done)+n_new}/{len(rows)}  {sp:.1f}/s  "
                      f"剩余 {(len(todo)-n_new)/max(sp,1e-9)/60:.0f}min",
                      flush=True)
    fh.close()
    print(f"  DONE 新增 {n_new} 条, 用时 {(time()-t0)/60:.1f} min", flush=True)

# 汇总
allr = list(csv.DictReader(open(a.out, encoding="utf-8")))
print(f"\n  === 汇总 (共 {len(allr)}) ===")
from collections import Counter

c = Counter(x["relation"] for x in allr)
print(f"    关系分布: {dict(c)}")
print(f"    完全一致率: {sum(int(x['exact']) for x in allr)/max(len(allr),1)*100:.1f}%")
conf_bins = Counter()
for x in allr:
    f = float(x["conf"])
    b = "0-0.3" if f < 0.3 else "0.3-0.6" if f < 0.6 else "0.6-0.9" if f < 0.9 else "0.9-1.0"
    conf_bins[b] += 1
print(f"    置信度分布: {dict(conf_bins)}")
