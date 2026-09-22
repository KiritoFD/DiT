#!/usr/bin/env python
"""PaddleOCR-VL-1.6 批量评测（batch 推理 + 断点续跑）。

## 为什么单独写
PaddleOCR-VL 只有 1.92GB（约 1B 参数，GQA 16/2），fp16 仅占 ~2G 显存，
24G 卡能开很大 batch -> 50,786 条全量才可行。

## 断点续跑
逐条 append 到 --out，启动时读已有结果跳过已处理的。
50,786 条要几小时，必须能中断重跑。

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/ppocrvl_batch.py \
      --csv assets/train_50k_v2_fixed.csv --batch 16 \
      --out assets/ppocrvl_train50k.csv
"""
import argparse
import csv
import os
import re
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

MODEL = ("_models/ocr/models/"
         "PaddlePaddle--PaddleOCR-VL-1.6/snapshots/master")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/train_50k_v2_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--out", default="assets/ppocrvl_train50k.csv")
ap.add_argument("--prompt", default="OCR:")
a = ap.parse_args()

from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: E402

print(f"  加载 {a.model} ...", flush=True)
dev = a.device if torch.cuda.is_available() else "cpu"
model = AutoModelForImageTextToText.from_pretrained(
    a.model, dtype=torch.bfloat16).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model)
_ip = proc.image_processor
_mp = getattr(_ip, "min_pixels", None) or 28 * 28 * 4
max_pixels = 1280 * 28 * 28
print(f"  device={dev}  min_pixels={_mp}  batch={a.batch}", flush=True)

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]

# ── 断点续跑：读已有结果 ──────────────────────────────────────────
done = set()
if os.path.exists(a.out):
    try:
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            done.add(int(r["idx"]))
    except Exception:
        pass
print(f"  {a.csv}: {len(rows)} 条, 已完成 {len(done)} 条", flush=True)

todo = [i for i in range(len(rows)) if i not in done]
if not todo:
    print("  ✓ 全部已完成")
    raise SystemExit(0)

hdr = not os.path.exists(a.out)
fout = open(a.out, "a", newline="", encoding="utf-8")
w = csv.writer(fout)
if hdr:
    w.writerow(["idx", "img_id", "char", "pred", "relaxed_eq",
                "raw", "callig", "script"])

t0 = time.time()
n_done = 0
B = max(1, a.batch)
for g0 in range(0, len(todo), B):
    idxs = todo[g0:g0 + B]
    ims, ok_idx = [], []
    for i in idxs:
        iid = rows[i].get("old_50k_id", "").strip()
        p = f"data/50k/imgs/{int(iid):06d}.png"
        if os.path.exists(p):
            try:
                ims.append(Image.open(p).convert("RGB"))
                ok_idx.append(i)
            except Exception:
                pass
    outs = {}
    if ims:
        try:
            msgs_b = [[{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": a.prompt}]}] for im in ims]
            inp = proc.apply_chat_template(
                msgs_b, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
                images_kwargs={"size": {"shortest_edge": _mp,
                                        "longest_edge": max_pixels}}
            ).to(model.device)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=a.max_new,
                                     do_sample=False)
            n_in = inp["input_ids"].shape[-1]
            texts = proc.batch_decode(out[:, n_in:], skip_special_tokens=True)
            for k, i in enumerate(ok_idx):
                outs[i] = (texts[k] or "").strip()
        except Exception as e:
            for i in ok_idx:
                outs[i] = f"ERR:{type(e).__name__}"

    for i in idxs:
        r = rows[i]
        raw = outs.get(i, "")
        han = re.findall(r"[\u4e00-\u9fff]", raw)
        pred = han[0] if han else "?"
        ch = r.get("character", "")
        w.writerow([i, r.get("old_50k_id", ""), ch, pred,
                    int(_c2t.convert(pred) == _c2t.convert(ch)),
                    raw[:40], r.get("calligrapher", ""),
                    r.get("script", "")])
    fout.flush()
    n_done += len(idxs)
    if (g0 // B + 1) % 10 == 0:
        el = time.time() - t0
        sp = n_done / max(el, 1e-9)
        eta = (len(todo) - n_done) / max(sp, 1e-9) / 3600
        print(f"    {n_done}/{len(todo)}  {sp:.1f} 条/s  ETA {eta:.1f}h",
              flush=True)

fout.close()
print(f"\n  ✓ 完成 {n_done} 条 -> {a.out}", flush=True)
