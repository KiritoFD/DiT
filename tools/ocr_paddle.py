"""PaddleOCR (繁体模型) 识别书法单字 GT，检测简繁错配。

用法: python tools/ocr_paddle.py --n 100 --script 楷 --lang chinese_cht
"""
import argparse
import csv
import os
import random
from collections import Counter

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--script", default="楷")
ap.add_argument("--lang", default="chinese_cht", help="chinese_cht=繁体, ch=简体")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

from paddleocr import PaddleOCR

print(f"  加载 PaddleOCR (lang={a.lang}) ...", flush=True)
ocr = PaddleOCR(lang=a.lang, use_textline_orientation=False,
                show_log=False) if True else None
print("  ✓ 模型已加载", flush=True)

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
if a.script != "all":
    rows = [r for r in rows if r["script"] == a.script]
random.seed(a.seed)
sample = random.sample(rows, min(a.n, len(rows)))
print(f"  测试 {len(sample)} 张 (书体={a.script})", flush=True)


def pad_square(im, pad=0.4):
    w, h = im.size
    s = int(max(w, h) * (1 + pad * 2))
    canvas = Image.new("RGB", (s, s), "white")
    canvas.paste(im, ((s - w) // 2, (s - h) // 2))
    return canvas


from opencc import OpenCC

t2s = OpenCC("t2s")

ok = 0
fail = 0
mismatch = []
for i, r in enumerate(sample):
    p = r["image_path"]
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    if not os.path.exists(src):
        fail += 1
        continue
    try:
        im = pad_square(Image.open(src).convert("RGB"))
        res = ocr.predict(np.array(im))
    except Exception as e:
        fail += 1
        continue
    pred = ""
    try:
        for rr in res:
            t = rr.get("rec_texts") or rr.get("rec_text") or []
            if isinstance(t, str):
                pred = t
            elif t:
                pred = "".join(t)
            if pred:
                break
    except Exception:
        pass
    pred = (pred or "")[:1] or "?"
    truth = r["character"]
    if pred == truth:
        ok += 1
    else:
        rel = "简繁" if t2s.convert(pred) == t2s.convert(truth) else ""
        mismatch.append((truth, pred, r["script"], r["calligrapher"], rel))
    if (i + 1) % 25 == 0:
        print(f"    {i+1}/{len(sample)}  exact={ok} fail={fail}", flush=True)

n_ok = len(sample) - fail
print(f"\n  === 结果 (n={len(sample)}, 书体={a.script}, lang={a.lang}) ===")
print(f"    完全一致: {ok}/{n_ok} = {ok/max(n_ok,1)*100:.1f}%")
print(f"    失败: {fail}")
print(f"    不一致: {len(mismatch)}")
c = Counter(m[4] for m in mismatch)
print(f"    其中简繁关系: {c.get('简繁', 0)}")

print(f"\n  === 简繁错配（真问题）===")
for truth, pred, sc, cal, rel in mismatch:
    if rel == "简繁":
        print(f"    csv={truth!r}  OCR={pred!r}  [{sc}/{cal}]")
print(f"\n  === 其他不一致（多为 OCR 错认，前20）===")
n = 0
for truth, pred, sc, cal, rel in mismatch:
    if rel != "简繁":
        print(f"    csv={truth!r}  OCR={pred!r}  [{sc}/{cal}]")
        n += 1
        if n >= 20:
            break
