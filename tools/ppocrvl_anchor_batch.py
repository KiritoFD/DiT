#!/usr/bin/env python
"""PaddleOCR-VL 批量识别（带 std 锚点 + 单字约束 + 置信度 + batch）。

## 与 ppocrvl_batch.py 的区别（按用户要求）
1. **给 std 锚点**：把 [std 骨架 | GT] 拼成一张图送进去，
   prompt 里写「左边是印刷体标准字，右边是书法手写」
   —— 实测 Qwen3-VL 加了锚点后认出了「陞」（不加只输出「升」）
2. **限定输出单字**：prompt 里明确「只输出一个汉字，不要解释、不要重复」
   —— 之前裸 "OCR:" 会输出「堂堂堂堂...」这种重复
3. **要置信度**：让模型自评 0-1
4. **batch 可调**（默认 32，可试 64）

## 输出格式（每行）
  CONF: 0.0~1.0
  CHAR: 一个字

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/ppocrvl_anchor_batch.py \
      --csv assets/train_50k_v2_fixed.csv --batch 64 \
      --out assets/ppocrvl_anchor_train.csv
"""
import argparse
import csv
import os
import re
import sys
import time

import torch
from PIL import Image, ImageDraw

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")

MODEL = ("_models/ocr/models/"
         "PaddlePaddle--PaddleOCR-VL-1.6/snapshots/master")

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/train_50k_v2_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=32)
ap.add_argument("--out", default="assets/ppocrvl_anchor_train.csv")
ap.add_argument("--size", type=int, default=224, help="拼图里每张的边长")
a = ap.parse_args()

PROMPT = (
    "这是一张拼接图：**左边**是印刷体标准字骨架，**右边**是书法家手写的单字。\n"
    "请识别**右边**书法写的那个字。\n"
    "注意：\n"
    "- 如果右边写的是左边这个字的繁体/异体/不同写法，请输出那个**具体的字**\n"
    "- 如果右边其实是别的字，就输出你看到的那个字\n"
    "**只输出一个汉字，不要解释、不要重复、不要标点**\n"
    "最后严格按两行输出：\n"
    "CONF: 0.0~1.0 的数字（你对识别结果的置信度）\n"
    "CHAR: 一个汉字"
)


def make_pair(std_p, gt_p, size):
    a1 = Image.open(std_p).convert("RGB").resize((size, size), Image.BICUBIC)
    a2 = Image.open(gt_p).convert("RGB").resize((size, size), Image.BICUBIC)
    c = Image.new("RGB", (size * 2 + 12, size), "white")
    c.paste(a1, (0, 0))
    c.paste(a2, (size + 12, 0))
    ImageDraw.Draw(c).line([(size + 6, 0), (size + 6, size)],
                           fill="gray", width=2)
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
print(f"  device={dev}  batch={a.batch}  size={a.size}", flush=True)

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]

done = set()
if os.path.exists(a.out):
    try:
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            done.add(int(r["idx"]))
    except Exception:
        pass
print(f"  {a.csv}: {len(rows)} 条, 已完成 {len(done)}", flush=True)

todo = [i for i in range(len(rows)) if i not in done]
if not todo:
    print("  ✓ 全部已完成")
    raise SystemExit(0)

hdr = not os.path.exists(a.out)
fout = open(a.out, "a", newline="", encoding="utf-8")
w = csv.writer(fout)
if hdr:
    w.writerow(["idx", "img_id", "char", "pred", "conf", "relaxed_eq",
                "raw", "callig", "script"])


def parse(raw):
    conf, ch = "", "?"
    m = re.search(r"CONF\s*[:：]\s*([0-9]*\.?[0-9]+)", raw)
    if m:
        conf = m.group(1)
    m = re.search(r"CHAR\s*[:：]\s*(\S)", raw)
    if m:
        ch = m.group(1)
    else:                       # 兜底：取最后一个汉字
        han = re.findall(r"[\u4e00-\u9fff]", raw)
        ch = han[-1] if han else "?"
    return ch, conf


t0 = time.time()
n_done = 0
B = max(1, a.batch)
for g0 in range(0, len(todo), B):
    idxs = todo[g0:g0 + B]
    ims, ok_idx = [], []
    for i in idxs:
        r = rows[i]
        iid = r.get("old_50k_id", "").strip()
        std_p = (r.get("std_path", "").strip()
                 or f"data/50k/std/{int(iid):06d}.png")
        gt_p = f"data/50k/imgs/{int(iid):06d}.png"
        if os.path.exists(std_p) and os.path.exists(gt_p):
            try:
                ims.append(make_pair(std_p, gt_p, a.size))
                ok_idx.append(i)
            except Exception:
                pass
    outs = {}
    if ims:
        try:
            msgs_b = [[{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT}]}] for im in ims]
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
        pred, conf = parse(raw)
        ch = r.get("character", "")
        w.writerow([i, r.get("old_50k_id", ""), ch, pred, conf,
                    int(_c2t.convert(pred) == _c2t.convert(ch)),
                    raw[:50], r.get("calligrapher", ""), r.get("script", "")])
    fout.flush()
    n_done += len(idxs)
    if (g0 // B + 1) % 5 == 0:
        el = time.time() - t0
        sp = n_done / max(el, 1e-9)
        eta = (len(todo) - n_done) / max(sp, 1e-9) / 3600
        print(f"    {n_done}/{len(todo)}  {sp:.1f} 条/s  ETA {eta:.2f}h",
              flush=True)

fout.close()
print(f"\n  ✓ 完成 {n_done} 条 -> {a.out}", flush=True)
