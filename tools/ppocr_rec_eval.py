#!/usr/bin/env python
"""用 PP-OCRv4（PaddleOCR 的 rec server 模型，ONNX）识别书法单字，算字准率。

## 为什么试这个
VLM 路线的问题是"认不准书法"（Qwen3-VL-4B 在 GT 上仅 25%）。
PP-OCRv4 是**专用 OCR**（PP-OCRv4_rec_server，字表 6623 字），
对印刷体/规整字形很强；书法是它的弱项，但值得实测一下。

## 与 VLM 的区别
- 不需要"上下文锚点"prompt
- 输出是字表上的 argmax，**不会幻觉**
- 但对"异体字"（如 陞）要看字表里有没有

用法:
  CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python tools/ppocr_rec_eval.py \
      --csv assets/eval_v13_seen_fixed.csv --limit 20
"""
import argparse
import csv
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")

MODEL_DIR = ("_models/ocr/models/"
             "cycloneboy--ch_PP-OCRv4_rec_server_infer/snapshots/master")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/eval_v13_seen_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--device", default="cuda")
ap.add_argument("--fp16", type=int, default=1)
ap.add_argument("--out", default="")
ap.add_argument("--h", type=int, default=48, help="rec 输入高度")
ap.add_argument("--w", type=int, default=320, help="rec 输入宽度")
a = ap.parse_args()

# ── 字表 ────────────────────────────────────────────────────────────
dict_p = os.path.join(MODEL_DIR, "ch_dict.txt")
chars = ["<blank>"]
for ln in open(dict_p, encoding="utf-8"):
    ln = ln.rstrip("\n")
    if ln:
        chars.append(ln)
print(f"  字表: {len(chars)} 项（含 blank）")

# ── ONNX session ────────────────────────────────────────────────────
mp = os.path.join(MODEL_DIR,
                  "fp16_model.onnx" if a.fp16 else "model.onnx")
prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
        if a.device == "cuda" else ["CPUExecutionProvider"])
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(mp, so, providers=prov)
print(f"  模型: {os.path.basename(mp)}")
print(f"  providers: {sess.get_providers()}")
iname = sess.get_inputs()[0].name
print(f"  输入: {iname} {sess.get_inputs()[0].shape}  "
      f"输出: {sess.get_outputs()[0].shape}")


def preprocess(p, h, w, keep_ratio=True):
    """rec 输入: (1,3,H,W)，归一化到 [-1,1]。

    ⚠ PP-OCRv4 的 rec 训练时是**按宽高比 resize + pad**（长文本行），
      对**单字**如果硬拉成 48x320 会把字形压扁 -> 识别率暴跌。
      这里保持比例，短边贴 48，宽边按比例（上限 w），不足则白边补齐。
    """
    im0 = Image.open(p).convert("RGB")
    if keep_ratio:
        r = im0.width / max(im0.height, 1)
        nw = min(w, max(8, int(round(h * r))))
        im = im0.resize((nw, h), Image.BICUBIC)
        if nw < w:
            c = Image.new("RGB", (w, h), (255, 255, 255))
            c.paste(im, (0, 0))
            im = c
    else:
        im = im0.resize((w, h), Image.BICUBIC)
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None]


def ctc_decode(logits, chars):
    """greedy CTC：去 blank(0)、去连续重复。"""
    idx = logits.argmax(-1)[0]          # (T,)
    out, prev = [], -1
    for k in idx:
        k = int(k)
        if k != 0 and k != prev:
            if 0 <= k < len(chars):
                out.append(chars[k])
        prev = k
    return "".join(out)


rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
print(f"\n  {a.csv}: {len(rows)} 条")

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")
res = []
for i, r in enumerate(rows):
    iid = r.get("old_50k_id", "").strip()
    ch = r.get("character", "")
    gt_p = f"data/50k/imgs/{int(iid):06d}.png"
    pred = "?"
    if os.path.exists(gt_p):
        try:
            x = preprocess(gt_p, a.h, a.w)
            logits = sess.run(None, {iname: x})[0]
            pred = ctc_decode(logits, chars)[:1] or "?"
        except Exception as e:
            pred = f"ERR:{type(e).__name__}"
    res.append({"idx": i, "char": ch, "pred": pred,
                "callig": r.get("calligrapher", ""),
                "script": r.get("script", ""),
                "src": r.get("src_image_path", "")})
    if (i + 1) % 20 == 0:
        print(f"    {i+1}/{len(rows)}", flush=True)

ex = sum(1 for x in res if x["pred"] == x["char"])
rl = sum(1 for x in res
         if _c2t.convert(x["pred"]) == _c2t.convert(x["char"]))
print(f"\n  ★ PP-OCRv4: exact={ex/max(len(res),1):.4f}  "
      f"relaxed(简繁)={rl/max(len(res),1):.4f}")
print(f"\n  前 30 条:")
for x in res[:30]:
    ok = "OK " if x["pred"] == x["char"] else "   "
    print(f"    {ok}{x['char']:>2} -> {x['pred']:<4} "
          f"({x['callig']}/{x['script']})")

if a.out:
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\n  -> {a.out}")
