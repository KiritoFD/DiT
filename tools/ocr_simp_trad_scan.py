"""用 rapidocr 跑大样本，统计简繁错配率（带置信度过滤）。

原理（用户 2026-09-21 提出）:
  OCR 的随机错认**很少正好落在简繁关系上**，所以
  "OCR 输出与 csv 的 character 成简繁关系" 的样本大概率是**真错配**。

输出: assets/simp_trad_mismatch.csv
"""
import argparse
import csv
import os
import random
from collections import Counter, defaultdict

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--script", default="all")
ap.add_argument("--conf", type=float, default=0.6, help="置信度阈值")
ap.add_argument("--out", default="assets/simp_trad_mismatch.csv")
a = ap.parse_args()

from opencc import OpenCC
from rapidocr_onnxruntime import RapidOCR

t2s = OpenCC("t2s")
s2t = OpenCC("s2t")
ocr = RapidOCR()

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
if a.script != "all":
    rows = [r for r in rows if r["script"] == a.script]
random.seed(0)
sample = random.sample(rows, min(a.n, len(rows)))
print(f"  测试 {len(sample)} 张 (书体={a.script}, conf>{a.conf})", flush=True)


def pad_square(im, pad=0.4):
    w, h = im.size
    s = int(max(w, h) * (1 + pad * 2))
    c = Image.new("RGB", (s, s), "white")
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c


ok = fail = lowconf = 0
mismatch = []
for i, r in enumerate(sample):
    p = r["image_path"]
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    if not os.path.exists(src):
        fail += 1
        continue
    try:
        res, _ = ocr(np.array(pad_square(Image.open(src).convert("RGB"))))
    except Exception:
        fail += 1
        continue
    if not res:
        fail += 1
        continue
    pred = "".join(x[1] for x in res)[:1] or "?"
    conf = float(res[0][2]) if len(res[0]) > 2 else 0.0
    truth = r["character"]
    if pred == truth:
        ok += 1
    else:
        rel = ""
        if t2s.convert(pred) == t2s.convert(truth):
            rel = "简繁"
        if conf < a.conf:
            lowconf += 1
            continue
        mismatch.append((r["image_path"], r["script"], r["calligrapher"],
                         truth, pred, round(conf, 3), rel))
    if (i + 1) % 500 == 0:
        print(f"    {i+1}/{len(sample)}  exact={ok} fail={fail} "
              f"mis={len(mismatch)}", flush=True)

n_ok = len(sample) - fail
print(f"\n  === 结果 (n={len(sample)}, 书体={a.script}) ===")
print(f"    完全一致: {ok}/{n_ok} = {ok/max(n_ok,1)*100:.1f}%")
print(f"    失败: {fail}  低置信度丢弃: {lowconf}")
print(f"    高置信度不一致: {len(mismatch)}")

rel_mis = [m for m in mismatch if m[6] == "简繁"]
print(f"\n    ★ 其中简繁关系: {len(rel_mis)} = "
      f"{len(rel_mis)/max(n_ok,1)*100:.1f}% (估计错配率)")

c = Counter(m[6] for m in mismatch)
print(f"    按关系: {dict(c)}")
sc = Counter(m[1] for m in rel_mis)
print(f"    简繁错配按书体: {dict(sc)}")

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["image_path", "script", "calligrapher", "char_csv",
                "char_ocr", "conf", "relation"])
    for m in sorted(mismatch, key=lambda x: (x[6] != "简繁", -x[5])):
        w.writerow(m)
print(f"\n  written {a.out}")

print(f"\n  === 简繁错配明细（全部）===")
for m in rel_mis:
    print(f"    csv={m[3]!r} -> OCR={m[4]!r}  conf={m[5]}  [{m[1]}/{m[2]}]")
