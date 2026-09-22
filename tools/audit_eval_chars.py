"""扫描评估集的「GT 实际写的字」vs「csv 的 character」是否一致。

## 为什么需要
已发现一条：csv=升，但 GT 实际写「陞」（异体字）。
OpenCC 的 s2t 不覆盖「升->陞」这种异体关系，所以之前的简繁检测漏了。

## 方法
对每条 eval 样本：
  1. 渲染 csv 的 character（用与 std g 同一个字体/流程）—— 或者直接读 std g
  2. OCR 原始 GT 图（用 rapidocr，它有方向性但能给出"最像的字"）
  3. 如果 OCR 结果和 character 不同 -> 标记为可疑
  4. 再用 VLM 复核可疑项

输出: assets/eval_char_audit.csv
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv")
ap.add_argument("--out", default="assets/eval_char_audit.csv")
ap.add_argument("--max", type=int, default=0)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.max:
    rows = rows[:a.max]
print(f"  {a.csv}: {len(rows)} 条", flush=True)

from rapidocr_onnxruntime import RapidOCR  # noqa: E402

ocr = RapidOCR()


def pad(im, p=0.4):
    w, h = im.size
    s = int(max(w, h) * (1 + p * 2))
    c = Image.new("RGB", (s, s), "white")
    c.paste(im, ((s - w) // 2, (s - h) // 2))
    return c


res = []
for i, r in enumerate(rows):
    ch = r["character"]
    p = r["image_path"]
    if not os.path.isabs(p):
        p = os.path.join("/root/Workspace/xy/DiT", p)
    pred = "?"
    if os.path.exists(p):
        try:
            out, _ = ocr(np.array(pad(Image.open(p).convert("RGB"))))
            pred = "".join(x[1] for x in out)[:1] if out else "?"
        except Exception:
            pass
    res.append({
        "idx": i, "character": ch, "ocr_gt": pred,
        "match": "Y" if pred == ch else "N",
        "src": r.get("src_image_path", ""),
        "callig": r.get("calligrapher", ""), "script": r.get("script", ""),
    })
    if (i + 1) % 50 == 0:
        print(f"    {i+1}/{len(rows)}", flush=True)

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
    w.writeheader()
    w.writerows(res)

bad = [x for x in res if x["match"] == "N"]
print(f"\n  不匹配: {len(bad)}/{len(res)} ({len(bad)/max(len(res),1)*100:.1f}%)")
print(f"\n  === 前 25 个可疑 ===")
for x in bad[:25]:
    print(f"    csv={x['character']}  ocr={x['ocr_gt']}  "
          f"({x['callig']}/{x['script']})  {os.path.basename(x['src'])}")
print(f"\n  -> {a.out}")
