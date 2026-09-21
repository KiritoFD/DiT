"""Qwen3-VL-4B 识别书法单字（CPU）。

用法:
  _venv_qwenvl/bin/python tools/vlm_ocr.py --n 5          # 小样本测试
  _venv_qwenvl/bin/python tools/vlm_ocr.py --csv <csv>    # 对指定清单跑
"""
import argparse
import csv
import os
import time

import torch
from PIL import Image

os.chdir("/root/Workspace/xy/DiT")
Image.MAX_IMAGE_PIXELS = None

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="/root/Workspace/xy/DiT/_models/models/"
                                   "Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
ap.add_argument("--n", type=int, default=5, help="从训练集随机取几张测试")
ap.add_argument("--csv", default="", help="指定清单 csv（含 image_path 列）")
ap.add_argument("--out", default="assets/vlm_ocr_result.csv")
ap.add_argument("--device", default="cpu")
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

PROMPT = ("这是一张中国古人书法的单字图片。请识别图片中的汉字。"
          "如果它是繁体字或异体字，请严格按图片实际写法输出繁体/异体字符。"
          "只输出那一个字，不要任何解释、标点或空格。")

print(f"  加载模型 (device={a.device}) ...", flush=True)
t0 = time.time()
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as _M
    _cls = "Qwen3VLForConditionalGeneration"
except ImportError:
    from transformers import AutoModelForImageTextToText as _M
    _cls = "AutoModelForImageTextToText"

model = _M.from_pretrained(a.model, dtype=torch.float32,
                           low_cpu_mem_usage=True).to(a.device).eval()
proc = AutoProcessor.from_pretrained(a.model)
print(f"  ✓ {_cls} 加载完成 {time.time()-t0:.0f}s", flush=True)

# 数据
if a.csv:
    items = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    if a.limit:
        items = items[:a.limit]
else:
    import random
    rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
    random.seed(0)
    items = random.sample(rows, min(a.n, len(rows)))
print(f"  待识别: {len(items)} 张", flush=True)

COLS = ["idx", "image_path", "script", "calligrapher", "char_csv", "char_vlm"]
out_rows = []
t1 = time.time()
for i, r in enumerate(items):
    p = r.get("image_path", "")
    src = p if os.path.isabs(p) else os.path.join("/root/Workspace/xy/DiT", p)
    pred = "?"
    if os.path.exists(src):
        try:
            im = Image.open(src).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
            inp = proc(text=[text], images=[im], return_tensors="pt").to(a.device)
            with torch.no_grad():
                gen = model.generate(**inp, max_new_tokens=8, do_sample=False)
            n_in = inp["input_ids"].shape[1]
            pred = proc.batch_decode(gen[:, n_in:], skip_special_tokens=True)[0].strip()
            pred = pred[:1] or "?"
        except Exception as e:
            pred = f"ERR:{type(e).__name__}"
    out_rows.append([r.get("idx", i), p, r.get("script", ""),
                     r.get("calligrapher", ""), r.get("character", ""), pred])
    el = time.time() - t1
    print(f"    [{i+1}/{len(items)}] csv={r.get('character','')!r} "
          f"VLM={pred!r}  {el/(i+1):.1f}s/张", flush=True)

with open(a.out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(COLS)
    w.writerows(out_rows)
print(f"\n  written {a.out}")
ok = sum(1 for r in out_rows if r[4] == r[5])
print(f"  完全一致: {ok}/{len(out_rows)} = {ok/max(len(out_rows),1)*100:.1f}%")
print(f"  平均 {((time.time()-t1)/max(len(out_rows),1)):.1f}s/张")
