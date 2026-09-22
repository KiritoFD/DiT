#!/usr/bin/env python
"""PP-OCRv4 拼接识别：把 N 个字拼成一行再送 rec 模型。

## 为什么拼接
PP-OCRv4 的 rec 模型是为**文本行**训练的（输入 48x320，一行多字）。
单字硬塞进去时：
  - 单字宽度只有 48px 左右 -> 分辨率极低
  - 与训练分布不符
拼接后每个字能分到 320/N 的宽度（N=4 时 80px/字），且更接近训练分布。

## 关键
- 每个字先各自 resize 到 h=48、宽度按自身宽高比（保持字形不变形）
- 拼成一行（中间加小间隔），总宽 <= w
- CTC 解码后得到 N 个字符，逐字对齐比较

用法:
  CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=... python tools/ppocr_concat_eval.py \
      --csv assets/eval_v13_strict_fixed.csv --group 4
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
ap.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--group", type=int, default=4, help="每行拼几个字")
ap.add_argument("--h", type=int, default=48)
ap.add_argument("--w", type=int, default=320)
ap.add_argument("--gap", type=int, default=8)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="")
a = ap.parse_args()

chars = ["<blank>"]
for ln in open(os.path.join(MODEL_DIR, "ch_dict.txt"), encoding="utf-8"):
    ln = ln.rstrip("\n")
    if ln:
        chars.append(ln)
print(f"  字表: {len(chars)} 项")

mp = os.path.join(MODEL_DIR, "model.onnx")
prov = (["CUDAExecutionProvider", "CPUExecutionProvider"]
        if a.device == "cuda" else ["CPUExecutionProvider"])
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(mp, so, providers=prov)
iname = sess.get_inputs()[0].name
print(f"  providers: {sess.get_providers()}")

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")


def norm_one(im, h, max_w):
    """单个字 -> 高 h，宽按比例（上限 max_w）。返回 (np_img, w)。"""
    r = im.width / max(im.height, 1)
    nw = max(8, min(max_w, int(round(h * r))))
    return np.asarray(im.resize((nw, h), Image.BICUBIC)), nw


def build_row(ims, h, w, gap):
    """把多个 PIL 图拼成一行（白底），总宽 <= w；不够就缩放。"""
    parts, widths = [], []
    budget = w - gap * (len(ims) - 1)
    per = max(8, budget // max(len(ims), 1))
    for im in ims:
        arr, nw = norm_one(im, h, per)
        parts.append(arr)
        widths.append(nw)
    total = sum(widths) + gap * (len(parts) - 1)
    if total > w:                      # 超了就整体等比缩
        sc = w / total
        parts = [np.asarray(Image.fromarray(p).resize(
            (max(4, int(p.shape[1] * sc)), h), Image.BICUBIC)) for p in parts]
        widths = [p.shape[1] for p in parts]
        total = sum(widths) + gap * (len(parts) - 1)
    row = np.full((h, w, 3), 255, np.uint8)
    x = 0
    for p, nw in zip(parts, widths):
        if x + nw > w:
            break
        row[:, x:x + nw] = p
        x += nw + gap
    return row


def to_input(row):
    x = row.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None]


def ctc_decode(logits, chars):
    idx = logits.argmax(-1)[0]
    out, prev = [], -1
    for k in idx:
        k = int(k)
        if k != 0 and k != prev and 0 <= k < len(chars):
            out.append(chars[k])
        prev = k
    return "".join(out)


rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
print(f"\n  {a.csv}: {len(rows)} 条, group={a.group}")

res = []
for g0 in range(0, len(rows), a.group):
    grp = rows[g0:g0 + a.group]
    ims, ok = [], True
    for r in grp:
        iid = r.get("old_50k_id", "").strip()
        p = f"data/50k/imgs/{int(iid):06d}.png"
        if not os.path.exists(p):
            ok = False
            break
        ims.append(Image.open(p).convert("RGB"))
    if not ok:
        for j, r in enumerate(grp):
            res.append({"idx": g0 + j, "char": r.get("character", ""),
                        "pred": "?", "group": g0 // a.group})
        continue
    row = build_row(ims, a.h, a.w, a.gap)
    try:
        logits = sess.run(None, {iname: to_input(row)})[0]
        txt = ctc_decode(logits, chars)
    except Exception as e:
        txt = f"ERR:{type(e).__name__}"
    # 按顺序对齐：预测的第 j 个字符对应组内第 j 条
    for j, r in enumerate(grp):
        pred = txt[j] if j < len(txt) else "?"
        res.append({"idx": g0 + j, "char": r.get("character", ""),
                    "pred": pred, "group": g0 // a.group,
                    "callig": r.get("calligrapher", ""),
                    "script": r.get("script", ""), "row_txt": txt})
    if (g0 // a.group + 1) % 10 == 0:
        print(f"    组 {g0//a.group+1}/{(len(rows)+a.group-1)//a.group}",
              flush=True)

ex = sum(1 for x in res if x["pred"] == x["char"])
rl = sum(1 for x in res
         if _c2t.convert(x["pred"]) == _c2t.convert(x["char"]))
print(f"\n  ★ PP-OCRv4 拼接(group={a.group}): "
      f"exact={ex/max(len(res),1):.4f}  relaxed={rl/max(len(res),1):.4f}")
print(f"\n  前 24 条:")
for x in res[:24]:
    ok = "OK " if x["pred"] == x["char"] else "   "
    print(f"    {ok}{x['char']:>2} -> {x['pred']:<3}  "
          f"({x.get('callig','')}/{x.get('script','')})  row={x.get('row_txt','')[:12]!r}")

if a.out:
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\n  -> {a.out}")
