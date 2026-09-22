#!/usr/bin/env python
"""PaddleOCR-VL + 文本锚点：把「已知的标准字」写在 prompt 文本里（不拼图）。

## 与 ppocrvl_anchor_batch.py 的区别
之前拼图 [std|GT] -> 模型被干扰（raw 里出现"数数"、重复）
现在**只送 GT 图**，字的信息用**文本**给：
    「这张图的书法单字，对应的标准字是「升」。
      请判断实际写的是不是「升」；如果是它的繁体/异体字，输出那个字。
      只输出一个汉字。」

## 为什么这样更好
- 不改变输入图像分布（PaddleOCR-VL 是按"单图 OCR"训练的）
- 字的信息用文本给，模型只需要"看图 -> 在候选里选"
- 天然是"分类"而不是"开放识别"，难度更低

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/ppocrvl_textanchor.py \
      --csv assets/eval_v13_strict_fixed.csv --batch 64
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
ap.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default=MODEL)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=16)
ap.add_argument("--style", default="ask",
                choices=["ask", "cands"],
                help="ask=只给标准字问它写的是什么；cands=给候选字表")
ap.add_argument("--out", default="")

# ask: 只给标准字
PROMPT_ASK = (
    "这张图是一个中国书法的单字。\n"
    "已知它对应的标准字（印刷体写法）是「{ch}」。\n"
    "请判断：图里实际写的是不是「{ch}」？\n"
    "- 如果就是「{ch}」，输出「{ch}」\n"
    "- 如果写的是它的**繁体字 / 异体字 / 另一种写法**，输出**那个具体的字**\n"
    "- 如果其实是完全不同的字，输出你看到的那个字\n"
    "**只输出一个汉字，不要解释、不要重复、不要标点。**"
)
# cands: 给候选字表（含常见变体）
PROMPT_CANDS = (
    "这张图是一个中国书法的单字。\n"
    "请从下面的候选字中选出图里实际写的那个字：\n"
    "{cands}\n"
    "**只输出选中的那一个汉字，不要解释、不要重复。**"
)

ap.add_argument("--variants-json", default="",
                help="可选：{character: [变体...]} 的 json（cands 模式用）")
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
print(f"  device={dev}  batch={a.batch}  style={a.style}", flush=True)

from opencc import OpenCC  # noqa: E402
_c2t = OpenCC("s2t")
_s2t = OpenCC("s2t")
_t2s = OpenCC("t2s")

VAR = {}
if a.variants_json and os.path.exists(a.variants_json):
    import json
    VAR = json.load(open(a.variants_json, encoding="utf-8"))

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]

out = a.out or f"/tmp/_pptext_{a.style}.csv"
done = set()
if os.path.exists(out):
    try:
        for r in csv.DictReader(open(out, encoding="utf-8")):
            done.add(int(r["idx"]))
    except Exception:
        pass
print(f"  {a.csv}: {len(rows)} 条, 已完成 {len(done)}", flush=True)

todo = [i for i in range(len(rows)) if i not in done]
if not todo:
    print("  ✓ 全部已完成")
    raise SystemExit(0)

hdr = not os.path.exists(out)
fout = open(out, "a", newline="", encoding="utf-8")
w = csv.writer(fout)
if hdr:
    w.writerow(["idx", "img_id", "char", "pred", "relaxed_eq", "raw",
                "callig", "script"])


def build_prompt(ch):
    if a.style == "cands":
        vs = VAR.get(ch, [])
        cands = [ch] + list(vs)
        # 去重保序
        seen, cc = set(), []
        for x in cands:
            if x not in seen:
                seen.add(x)
                cc.append(x)
        return PROMPT_CANDS.format(cands="、".join(cc))
    return PROMPT_ASK.format(ch=ch)


t0 = time.time()
n_done = 0
B = max(1, a.batch)
for g0 in range(0, len(todo), B):
    idxs = todo[g0:g0 + B]
    ims, prompts, ok_idx = [], [], []
    for i in idxs:
        r = rows[i]
        iid = r.get("old_50k_id", "").strip()
        p = f"data/50k/imgs/{int(iid):06d}.png"
        if os.path.exists(p):
            try:
                ims.append(Image.open(p).convert("RGB"))
                prompts.append(build_prompt(r.get("character", "")))
                ok_idx.append(i)
            except Exception:
                pass
    outs = {}
    if ims:
        try:
            msgs_b = [[{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": pr}]}]
                for im, pr in zip(ims, prompts)]
            inp = proc.apply_chat_template(
                msgs_b, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True,
                images_kwargs={"size": {"shortest_edge": _mp,
                                        "longest_edge": max_pixels}}
            ).to(model.device)
            with torch.no_grad():
                o = model.generate(**inp, max_new_tokens=a.max_new,
                                   do_sample=False)
            n_in = inp["input_ids"].shape[-1]
            texts = proc.batch_decode(o[:, n_in:], skip_special_tokens=True)
            for k, i in enumerate(ok_idx):
                outs[i] = (texts[k] or "").strip()
        except Exception as e:
            for i in ok_idx:
                outs[i] = f"ERR:{type(e).__name__}"

    for i in idxs:
        r = rows[i]
        raw = outs.get(i, "")
        ch = r.get("character", "")
        han = re.findall(r"[\u4e00-\u9fff]", raw)
        pred = han[0] if han else "?"
        w.writerow([i, r.get("old_50k_id", ""), ch, pred,
                    int(_s2t.convert(pred) == _s2t.convert(ch)),
                    raw[:50], r.get("calligrapher", ""),
                    r.get("script", "")])
    fout.flush()
    n_done += len(idxs)
    if (g0 // B + 1) % 5 == 0:
        el = time.time() - t0
        sp = n_done / max(el, 1e-9)
        print(f"    {n_done}/{len(todo)}  {sp:.1f} 条/s  "
              f"ETA {(len(todo)-n_done)/max(sp,1e-9)/3600:.2f}h", flush=True)

fout.close()

# 汇总
res = list(csv.DictReader(open(out, encoding="utf-8")))
ex = sum(1 for r in res if r["pred"] == r["char"])
rl = sum(int(r["relaxed_eq"]) for r in res)
print(f"\n  ★ style={a.style}: n={len(res)}  exact={ex/len(res):.4f}  "
      f"relaxed={rl/len(res):.4f}", flush=True)
print(f"\n  前 20 条:")
for r in res[:20]:
    ok = "OK " if r["pred"] == r["char"] else "   "
    print(f"    {ok}{r['char']:>2} -> {r['pred']:<3} raw={r['raw'][:26]!r}")
print(f"\n  -> {out}")
