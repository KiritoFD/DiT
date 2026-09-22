#!/usr/bin/env python
"""用 Qwen2.5-VL-7B 对目录里的 g*.png 算字准率（CPU）。

用法（在 _venv_qwenvl 环境跑）:
  CUDA_VISIBLE_DEVICES= ./_venv_qwenvl/bin/python tools/vlm_ocr_acc.py \
      --dir /tmp/_gtocr --csv assets/eval_v13_seen_fixed.csv
"""
import argparse
import csv
import glob
import os
import re
import sys

import torch
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

from opencc import OpenCC  # noqa: E402

t2s = OpenCC("t2s")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--csv", default="assets/eval_v13_seen_fixed.csv")
ap.add_argument("--model", default="/root/Workspace/xy/DiT/_models/models/"
                                   "Qwen--Qwen2.5-VL-7B-Instruct/snapshots/master")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="")
a = ap.parse_args()

gs = sorted([p for p in glob.glob(os.path.join(a.dir, "g*.png"))
             if re.search(r"g(\d+)\.png$", p)],
            key=lambda p: int(re.search(r"g(\d+)\.png$", p).group(1)))
if a.limit:
    gs = gs[:a.limit]
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
chars = [r["character"] for r in rows[:len(gs)]]
print(f"  g={len(gs)} csv={len(chars)}", flush=True)

from transformers import AutoProcessor  # noqa: E402
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as M
except ImportError:
    from transformers import AutoModelForImageTextToText as M

print(f"  加载 {a.model} ...", flush=True)
m = M.from_pretrained(a.model, dtype=torch.float32,
                      low_cpu_mem_usage=True).eval()
pr = AutoProcessor.from_pretrained(a.model)
try:
    pr.image_processor.max_pixels = 512 * 512
except Exception:
    pass

PROMPT = ("这是一张中国古人书法的单字图片。请识别图片中的汉字。"
          "如果它是繁体字或异体字，请严格按图片实际写法输出繁体/异体字符。"
          "只输出那一个字，不要任何解释、标点或空格。")

preds = []
for i, p in enumerate(gs):
    try:
        im = Image.open(p).convert("RGB")
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": im},
            {"type": "text", "text": PROMPT}]}]
        tx = pr.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True)
        inp = pr(text=[tx], images=[im], return_tensors="pt")
        with torch.no_grad():
            gen = m.generate(**inp, max_new_tokens=8, do_sample=False)
        n_in = inp["input_ids"].shape[1]
        preds.append((pr.batch_decode(gen[:, n_in:],
                                      skip_special_tokens=True)[0] or "?"
                      ).strip()[:1] or "?")
    except Exception as e:
        preds.append("?")
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{len(gs)}", flush=True)

ex = sum(1 for x, y in zip(preds, chars) if x == y)
rel = sum(1 for x, y in zip(preds, chars)
          if t2s.convert(x) == t2s.convert(y))
print(f"\n  ★ {os.path.basename(a.model)}: "
      f"exact={ex/max(len(chars),1):.4f}  relaxed={rel/max(len(chars),1):.4f}",
      flush=True)

print("\n  === 明细 ===")
for p, c, q in zip(gs, chars, preds):
    ok = "OK " if q == c else "   "
    print(f"    {ok}{os.path.basename(p)}  gt={c}  vlm={q}")

if a.out:
    with open(a.out, "w", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "gt", "vlm"])
        for p, c, q in zip(gs, chars, preds):
            w.writerow([os.path.basename(p), c, q])
    print(f"  -> {a.out}")
