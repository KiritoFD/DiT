#!/usr/bin/env python
"""用 PaddleOCR-VL-1.6 识别书法单字，算字准率。

## 为什么试它
README 明确说在 **中文古籍 + 中文罕见字** 上有显著提升（OmniDocBench v1.6 96.33% SOTA）。
它是**专用 OCR 的 VLM**（ERNIE4.5 + PaddleOCR），可能兼具
"专用 OCR 的准"和"VLM 的上下文理解"。

## 用法（按 README）
  prompt 就是 "OCR:" / "Spotting:"
  用 AutoModelForImageTextToText + AutoProcessor
  ⚠ transformers>=5.0.0（我们有 5.17.0 ✓）
  ⚠ 需要在 _venv_qwenvl 里跑（accelerate>=1.0 已装 ✓）

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/ppocrvl_eval.py \
      --csv assets/eval_v13_seen_fixed.csv --limit 20
"""
import argparse
import csv
import os
import re
import sys

import torch
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

MODEL = ("_models/ocr/models/"
         "PaddlePaddle--PaddleOCR-VL-1.6/snapshots/master")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/eval_v13_seen_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--task", default="ocr", choices=["ocr", "spotting"])
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=64)
ap.add_argument("--out", default="")
a = ap.parse_args()

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

PROMPTS = {"ocr": "OCR:", "spotting": "Spotting:"}
print(f"  加载 {a.model} ...", flush=True)
dev = a.device if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained(
    a.model, dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model)
_ip = proc.image_processor
_mp = getattr(_ip, "min_pixels", None) or getattr(_ip, "min_pixels_", None) or 28 * 28 * 4
print(f"  device={dev}  min_pixels={_mp}", flush=True)
print(f"  image_processor: {type(_ip).__name__}  attrs={[x for x in dir(_ip) if chr(112)+chr(105)+chr(120)+chr(101)+chr(108) in x.lower()][:6]}", flush=True)

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
print(f"\n  {a.csv}: {len(rows)} 条, task={a.task}", flush=True)

max_pixels = (2048 * 28 * 28 if a.task == "spotting" else 1280 * 28 * 28)
res = []
for i, r in enumerate(rows):
    iid = r.get("old_50k_id", "").strip()
    ch = r.get("character", "")
    p = f"data/50k/imgs/{int(iid):06d}.png"
    pred, raw = "?", ""
    if os.path.exists(p):
        try:
            im = Image.open(p).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPTS[a.task]}]}]
            inp = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
                images_kwargs={"size": {
                    "shortest_edge": _mp,
                    "longest_edge": max_pixels}}).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new,
                                     do_sample=False)
            raw = proc.decode(out[0][inp["input_ids"].shape[-1]:-1]).strip()
            # 取第一个汉字（spotting 会带坐标，先剔掉非汉字）
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
print(f"\n  ★ PaddleOCR-VL-1.6({a.task}): exact={ex/max(len(res),1):.4f}  "
      f"relaxed={rl/max(len(res),1):.4f}")
print(f"\n  前 30 条:")
for x in res[:30]:
    ok = "OK " if x["pred"] == x["char"] else "   "
    print(f"    {ok}{x['char']:>2} -> {x['pred']:<3}  "
          f"({x['callig']}/{x['script']})  raw={x['raw'][:20]!r}")

if a.out:
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
        w.writeheader()
        w.writerows(res)
    print(f"\n  -> {a.out}")
