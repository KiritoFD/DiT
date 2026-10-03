"""用**已有的 OCR 产物**找"std 与真迹不是同一个字"的行 (不重跑 OCR), 并出反色样本图。

  · assets/ocr_gpu_scan.csv: idx,image_path,script,calligrapher,char_csv,char_ocr,conf,exact,relation
  · 与 exp-std/csv/train.csv 按 **数字 basename** 对齐
输出: exact=0 / relation!=same 的行清单 + 反色 10 张的拼图。
"""
import csv
import os
import re
from collections import Counter

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
os.makedirs("_ot_scratch", exist_ok=True)

OCR = "assets/ocr_gpu_scan.csv"
TR = "exp-std/csv/train.csv"

with open(OCR, encoding="utf-8") as f:
    rd = csv.DictReader(f)
    cols = rd.fieldnames
    ocr = list(rd)
print(f"[ocr] {OCR} n={len(ocr)} cols={cols}")
print(f"[ocr] relation 分布: {dict(Counter(r.get('relation') for r in ocr).most_common(8))}")
print(f"[ocr] exact 分布:    {dict(Counter(r.get('exact') for r in ocr).most_common(5))}")
print(f"[ocr] image_path 前缀: "
      f"{dict(Counter(os.path.dirname(r['image_path']) for r in ocr).most_common(4))}")


def num(p):
    m = re.search(r"(\d+)\.png$", p or "")
    return m.group(1) if m else None


lut = {}
for r in ocr:
    k = num(r.get("image_path", ""))
    if k:
        lut.setdefault(k, r)
print(f"[ocr] 可按数字 id 索引 {len(lut)} 条")

with open(TR, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
hit = [r for r in rows if num(r["image_path"]) in lut]
print(f"[join] 训练集 {len(rows)} 条, 命中 OCR {len(hit)} 条 ({len(hit)/len(rows):.1%})")
if hit:
    cc = Counter(lut[num(r["image_path"])].get("relation") for r in hit)
    print(f"[join] relation 分布: {dict(cc.most_common(8))}")
    bad = [r for r in hit
           if str(lut[num(r["image_path"])].get("exact")) != "1"]
    print(f"[join] exact!=1 (=OCR 与 csv 字不一致) 的 = {len(bad)} "
          f"({len(bad)/max(len(hit),1):.1%})")
    print("[样例] 前 12 条:")
    for r in bad[:12]:
        o = lut[num(r["image_path"])]
        print(f"   id={r['img_id']:>6} csv={r['character']} ocr={o.get('char_ocr')} "
              f"conf={o.get('conf')} rel={o.get('relation')} slot={r['slot_name']}")
    with open("_ot_scratch/ocr_mismatch_rows.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(bad)
    print(f"[落盘] _ot_scratch/ocr_mismatch_rows.csv ({len(bad)} 行)")

# 反色 10 张
IDS = ["019124", "016860", "019079", "019080", "019199",
       "019917", "021516", "019209", "017969", "020666"]
S, cols_ = 150, 2
rows_ = (len(IDS) + cols_ - 1) // cols_
canvas = Image.new("L", (S * 2 * cols_, S * rows_), 255)
for j, i in enumerate(IDS):
    ps, pg = f"data/top10_style23/std/{i}.png", f"data/top10_style23/imgs/{i}.png"
    if not (os.path.exists(ps) and os.path.exists(pg)):
        continue
    r_, c_ = j // cols_, j % cols_
    canvas.paste(Image.open(ps).convert("L").resize((S, S)), (c_ * 2 * S, r_ * S))
    canvas.paste(Image.open(pg).convert("L").resize((S, S)), (c_ * 2 * S + S, r_ * S))
canvas.save("_ot_scratch/inverted_10.png")
print("[图] _ot_scratch/inverted_10.png (左=条件 std, 右=真迹)")
