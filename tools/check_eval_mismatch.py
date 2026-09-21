"""检查 **eval 集**是否也有简繁错配（否则评测本身就带错配）。

对 eval csv 跑 rapidocr(GPU) + Qwen2.5-VL-7B，用双模型交叉验证判错配。
"""
import csv
import os
import sys
from collections import Counter

import numpy as np
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None
sys.path.insert(0, "/root/Workspace/xy/DiT")

from opencc import OpenCC  # noqa: E402

s2t = OpenCC("s2t")
t2s = OpenCC("t2s")

EVAL_CSVS = ["assets/eval_v13_strict.csv", "assets/eval_v13_seen.csv"]

# ── 1) 收集所有 eval 样本 ─────────────────────────────────────────────
items = []
for f in EVAL_CSVS:
    if not os.path.exists(f):
        continue
    for i, r in enumerate(csv.DictReader(open(f, encoding="utf-8"))):
        r["_src"] = f
        r["_i"] = i
        items.append(r)
print(f"  eval 样本: {len(items)}")

# ── 2) rapidocr（GPU）────────────────────────────────────────────────
from rapidocr_onnxruntime import RapidOCR  # noqa: E402

try:
    ocr = RapidOCR(det_use_cuda=True, cls_use_cuda=True, rec_use_cuda=True)
except Exception:
    ocr = RapidOCR()


def pad(im, p=0.4):
    w, h = im.size
    s = int(max(w, h) * (1 + p * 2))
    c = Image.new("RGB", (s, s), "white")
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c


for it in items:
    p = it["image_path"]
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    it["_ocr"], it["_conf"] = "?", 0.0
    if os.path.exists(src):
        try:
            res, _ = ocr(np.array(pad(Image.open(src).convert("RGB"))))
            if res:
                it["_ocr"] = "".join(x[1] for x in res)[:1] or "?"
                it["_conf"] = float(res[0][2]) if len(res[0]) > 2 else 0.0
        except Exception:
            pass
print(f"  OCR 完成")

# ── 3) 导出给 VLM ────────────────────────────────────────────────────
os.makedirs("assets", exist_ok=True)
tmp = "assets/_eval_for_vlm.csv"
with open(tmp, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher", "char_csv"])
    for k, it in enumerate(items):
        w.writerow([k, it["image_path"], it.get("script", ""),
                    it.get("calligrapher", ""), it["character"]])
print(f"  written {tmp} -> 用 vlm_full_scan.py --csv 跑")

# 保存 OCR 结果
with open("assets/_eval_ocr.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["idx", "image_path", "script", "calligrapher",
                "char_csv", "char_ocr", "conf"])
    for k, it in enumerate(items):
        w.writerow([k, it["image_path"], it.get("script", ""),
                    it.get("calligrapher", ""), it["character"],
                    it["_ocr"], round(it["_conf"], 4)])

# ── 4) 先看 OCR 单模型的线索 ─────────────────────────────────────────
cand = [it for it in items
        if it["_ocr"] != it["character"]
        and s2t.convert(it["character"]) == it["_ocr"]
        and s2t.convert(it["character"]) != it["character"]]
print(f"\n  === OCR 单模型线索（csv 简体 -> OCR 繁体）: {len(cand)} ===")
for it in cand:
    print(f"    {it['character']} -> {it['_ocr']}  conf={it['_conf']:.3f}  "
          f"[{it.get('script','')}/{it.get('calligrapher','')}]  ({it['_src']})")

# 反向
rev = [it for it in items
       if it["_ocr"] != it["character"]
       and t2s.convert(it["character"]) == it["_ocr"]
       and t2s.convert(it["character"]) != it["character"]]
print(f"\n  (对照) csv 繁体 -> OCR 简体: {len(rev)} 条（OCR 局限）")

print(f"\n  === 一致率 ===")
ok = sum(1 for it in items if it["_ocr"] == it["character"])
print(f"    {ok}/{len(items)} = {ok/len(items)*100:.1f}%")
