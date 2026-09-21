"""用 rapidocr 检测 GT 图实际字形 vs csv 的 character 是否简繁错配。

⚠ rapidocr 是文本行识别模型 —— 单字图需要 pad 白边（用户建议）。
   书法（尤其行草）准确率会低，所以先小样本测准确率，再决定是否全量。
"""
import argparse
import csv
import os
import random
from collections import Counter

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--script", default="楷", help="只测某个书体(楷/行/隶)，all=全部")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from rapidocr_onnxruntime import RapidOCR

ocr = RapidOCR()

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
if a.script != "all":
    rows = [r for r in rows if r["script"] == a.script]
random.seed(a.seed)
sample = random.sample(rows, min(a.n, len(rows)))
print(f"  测试 {len(sample)} 张 (书体={a.script})")


def pad_square(im, pad_ratio=0.35):
    w, h = im.size
    s = int(max(w, h) * (1 + pad_ratio * 2))
    canvas = Image.new("RGB", (s, s), "white")
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


from opencc import OpenCC

t2s = OpenCC("t2s")
s2t = OpenCC("s2t")

ok = 0
mismatch = []
fail = 0
for i, r in enumerate(sample):
    p = r["image_path"]
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    if not os.path.exists(src):
        fail += 1
        continue
    try:
        im = Image.open(src).convert("RGB")
        im = pad_square(im)
        arr = np.array(im)
        res, _ = ocr(arr)
    except Exception:
        fail += 1
        continue
    if not res:
        fail += 1
        continue
    pred = "".join(x[1] for x in res)[:1] or "?"
    truth = r["character"]
    if pred == truth:
        ok += 1
    else:
        # 是否是简繁关系
        rel = ""
        if t2s.convert(pred) == t2s.convert(truth):
            rel = "简繁"
        mismatch.append((truth, pred, r["script"], r["calligrapher"], rel))
    if (i + 1) % 25 == 0:
        print(f"    {i+1}/{len(sample)}  exact={ok} fail={fail}", flush=True)

n_ok = len(sample) - fail
print(f"\n  === 结果 (n={len(sample)}, 书体={a.script}) ===")
print(f"    完全一致: {ok}/{n_ok} = {ok/max(n_ok,1)*100:.1f}%")
print(f"    OCR 失败: {fail}")
print(f"    不一致:   {len(mismatch)}")
c = Counter(m[4] for m in mismatch)
print(f"    其中属于简繁关系的: {c.get('简繁', 0)}")

print(f"\n  === 不一致样例（前 25）===")
for truth, pred, sc, cal, rel in mismatch[:25]:
    print(f"    csv={truth!r}  OCR={pred!r}  [{sc}/{cal}]  {rel}")
