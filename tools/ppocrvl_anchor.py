#!/usr/bin/env python
"""PaddleOCR-VL + 上下文锚点：把 std 骨架和 GT 拼在一起送进去。

## 为什么
孤立单字是 VLM 的弱项（实测 Qwen3-VL 在 GT 上仅 25-30%）。
Qwen3-VL 加了「已知左边是 X」的锚点后，认出了「陞」（不加锚点只输出「升」）。
PaddleOCR-VL 本身就是 OCR 专用，加锚点可能更强。

## 三种 prompt
  plain    : 只送 GT，问"这是什么字"
  anchor   : 送 [std|GT]，告诉它"左边是 X"，问"右边写的是不是 X，或是别的什么字"
  contrast : 送 [std|GT]，问"右边和左边是不是同一个字；如果不是，右边是什么"

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/ppocrvl_anchor.py \
      --csv assets/eval_v13_seen_fixed.csv --limit 20 --mode anchor
"""
import argparse
import csv
import os
import re
import sys

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

MODEL = ("_models/ocr/models/"
         "PaddlePaddle--PaddleOCR-VL-1.6/snapshots/master")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/eval_v13_seen_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--mode", default="anchor",
                choices=["plain", "anchor", "contrast"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--out", default="")
a = ap.parse_args()

PROMPTS = {
    "plain": "OCR:",
    "anchor": (
        "这是一张拼接图：左边是印刷体标准字（已知它是「{ch}」），"
        "右边是书法家手写的同一个字的不同写法。\n"
        "请仔细看右边的书法，判断它写的是哪个字。\n"
        "如果右边是「{ch}」的繁体字、异体字或不同写法，请输出那个具体的字；"
        "如果右边其实是别的字，就输出你看到的那个字。\n"
        "只输出一个汉字，不要任何解释。"),
    "contrast": (
        "这是一张拼接图：左边是印刷体标准字，右边是书法家手写的单字。\n"
        "请判断：右边写的和左边是不是完全同一个字？\n"
        "注意：如果是繁体/简体差异、异体字、或部件不同，都算**不同**。\n"
        "最后输出两行：\n"
        "SAME: A 或 B   （A=同一个字, B=不是）\n"
        "GT_CHAR: 右边写的那个字"),
}


def make_pair(std_p, gt_p, size=256, gap=16):
    a1 = Image.open(std_p).convert("RGB").resize((size, size))
    a2 = Image.open(gt_p).convert("RGB").resize((size, size))
    c = Image.new("RGB", (size * 2 + gap, size), "white")
    c.paste(a1, (0, 0))
    c.paste(a2, (size + gap, 0))
    ImageDraw.Draw(c).line(
        [(size + gap // 2, 0), (size + gap // 2, size)], fill="gray", width=2)
    return c


from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

print(f"  加载 {a.model} ...", flush=True)
dev = a.device if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained(
    a.model, dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model)
_ip = proc.image_processor
_mp = getattr(_ip, "min_pixels", None) or 28 * 28 * 4
max_pixels = 1280 * 28 * 28

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
print(f"\n  {a.csv}: {len(rows)} 条, mode={a.mode}", flush=True)

res = []
for i, r in enumerate(rows):
    iid = r.get("old_50k_id", "").strip()
    ch = r.get("character", "")
    std_p = r.get("std_path", "").strip() or f"data/50k/std/{int(iid):06d}.png"
    gt_p = f"data/50k/imgs/{int(iid):06d}.png"
    pred, raw = "?", ""
    if os.path.exists(gt_p) and os.path.exists(std_p):
        try:
            if a.mode == "plain":
                im, txt = Image.open(gt_p).convert("RGB"), PROMPTS["plain"]
            else:
                im = make_pair(std_p, gt_p)
                txt = PROMPTS[a.mode].format(ch=ch)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": txt}]}]
            inp = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                images_kwargs={"size": {"shortest_edge": _mp,
                                        "longest_edge": max_pixels}}
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new,
                                     do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[-1]:-1]).strip()
            if a.mode == "contrast":
                m = re.search(r"GT_CHAR\s*[:：]\s*(\S)", raw)
                pred = m.group(1) if m else "?"
            else:
                han = re.findall(r"[\u4e00-\u9fff]", raw)
                pred = han[0] if han else "?"
        except Exception as e:
            pred, raw = "ERR", f"{type(e).__name__}:{str(e)[:60]}"
    res.append({"idx": i, "char": ch, "pred": pred, "raw": raw[:60],
                "callig": r.get("calligrapher", ""),
                "script": r.get("script", "")})
    if (i + 1) % 10 == 0:
        print(f"    {i+1}/{len(rows)}", flush=True)

ex = sum(1 for x in res if x["pred"] == x["char"])
rl = sum(1 for x in res
         if _c2t.convert(x["pred"]) == _c2t.convert(x["char"]))
print(f"\n  ★ PaddleOCR-VL({a.mode}): exact={ex/max(len(res),1):.4f}  "
      f"relaxed={rl/max(len(res),1):.4f}")
print(f"\n  前 30 条:")
for x in res[:30]:
    ok = "OK " if x["pred"] == x["char"] else "   "
    print(f"    {ok}{x['char']:>2} -> {x['pred']:<3}  "
          f"({x['callig']}/{x['script']})  raw={x['raw'][:24]!r}")

if a.out:
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\n  -> {a.out}")
