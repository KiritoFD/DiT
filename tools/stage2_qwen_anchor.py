#!/usr/bin/env python
"""二阶段：对一阶段筛出的可疑样本，用 Qwen3-VL-4B + std 锚点复核。

## 流程
一阶段（PaddleOCR-VL 裸 OCR:）-> 50,786 条，39.9% 判为"与 csv 不一致"
但 PaddleOCR-VL 本身错 40% -> 大量假阳性 ✗

二阶段：只对可疑的，用 Qwen3-VL-4B 问
    「已知标准字是「{ch}」，图里实际写的是不是它？是它的繁体/异体就输出那个字」
实测 Qwen3-VL 加了这种锚点后能认出「陞」（PaddleOCR-VL 认成「坣」）

## 判定（三档）
  strong   : 两模型都认为"不是 csv 的字"，且 Qwen3-VL 给出的字与 PaddleOCR-VL 一致
  weak     : 只有 Qwen3-VL 认为不同
  reject   : Qwen3-VL 认为就是 csv 的字 -> 一阶段的判定是假阳性

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/stage2_qwen_anchor.py \
      --stage1 assets/ppocrvl_train50k.csv \
      --csv assets/train_50k_v2_fixed.csv \
      --batch 64 --out assets/stage2_review.csv
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

MODEL = "/root/Workspace/xy/DiT/_models/models/Qwen--Qwen3-VL-4B-Instruct/snapshots/master"

ap = argparse.ArgumentParser()
ap.add_argument("--stage1", default="assets/ppocrvl_train50k.csv")
ap.add_argument("--csv", default="assets/train_50k_v2_fixed.csv")
ap.add_argument("--model", default=MODEL)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--device", default="cuda")
ap.add_argument("--max-new", type=int, default=24)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--sample-every", type=int, default=0,
                help=">1 时按 idx 均匀抽样（每隔 N 条取 1 条），用于分区评估")
ap.add_argument("--out", default="assets/stage2_review.csv")
ap.add_argument("--only-suspicious", type=int, default=1)
a = ap.parse_args()

PROMPT = (
    "这张图是一个中国书法的单字。\n"
    "已知它对应的标准字（印刷体写法）是「{ch}」。\n"
    "请判断：图里实际写的是不是「{ch}」？\n"
    "- 如果就是「{ch}」，输出「{ch}」\n"
    "- 如果写的是它的繁体字、异体字或另一种写法，输出那个具体的字\n"
    "- 如果其实是完全不同的字，输出你看到的那个字\n"
    "严格按两行输出，不要其它内容：\n"
    "CONF: 0.0~1.0 的数字（你的置信度）\n"
    "CHAR: 一个汉字"
)

from transformers import AutoProcessor  # noqa: E402
try:
    from transformers import Qwen3VLForConditionalGeneration as M
except ImportError:
    from transformers import Qwen2_5_VLForConditionalGeneration as M

print(f"  加载 {a.model} ...", flush=True)
dev = a.device if torch.cuda.is_available() else "cpu"
model = M.from_pretrained(a.model, dtype=torch.bfloat16,
                          low_cpu_mem_usage=True).to(dev).eval()
proc = AutoProcessor.from_pretrained(a.model)
# ⚠ decoder-only 模型必须左 padding，否则批量生成结果错乱
#   （transformers 会警告 right-padding detected）
try:
    proc.tokenizer.padding_side = "left"
except Exception:
    pass

# 汉字范围: BMP(U+4E00-9FFF) + 扩展A(U+3400-4DBF) + 扩展B..(U+20000+)
# ⚠ 只用 \u4e00-\u9fff 会漏掉模型输出的扩展区生僻字（实测 raw 里出现「㑺」「㝡」）
HAN_RE = re.compile(
    "[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0003ffff]")

from opencc import OpenCC  # noqa: E402
_s2t = OpenCC("s2t")

s1 = {int(r["idx"]): r for r in csv.DictReader(
    open(a.stage1, encoding="utf-8"))}
rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
print(f"  stage1: {len(s1)} 条  csv: {len(rows)} 条", flush=True)

if a.only_suspicious:
    todo = [i for i in range(len(rows))
            if i in s1 and not int(s1[i].get("relaxed_eq", 1))]
else:
    todo = [i for i in range(len(rows)) if i in s1]
if a.limit:
    todo = todo[:a.limit]
# 分区抽样：按 idx 均匀取（分层），用于看不同区段的行为差异
if a.sample_every and a.sample_every > 1:
    todo = todo[::a.sample_every]
print(f"  待复核: {len(todo)} 条"
      + (f"（每 {a.sample_every} 条取 1）" if a.sample_every > 1 else ""),
      flush=True)

done = set()
if os.path.exists(a.out):
    try:
        for r in csv.DictReader(open(a.out, encoding="utf-8")):
            done.add(int(r["idx"]))
    except Exception:
        pass
todo = [i for i in todo if i not in done]
if not todo:
    print("  ✓ 全部已完成")
    raise SystemExit(0)
print(f"  实际要跑: {len(todo)} 条", flush=True)

hdr = not os.path.exists(a.out)
fout = open(a.out, "a", newline="", encoding="utf-8")
w = csv.writer(fout)
if hdr:
    w.writerow(["idx", "img_id", "char", "s1_pred", "s2_pred", "s2_conf",
                "s2_same_as_csv", "s2_relaxed_eq", "agree_with_s1",
                "verdict", "raw", "callig", "script"])


def verdict(ch, s1p, s2p):
    """strong / weak / reject"""
    s2_diff = _s2t.convert(s2p) != _s2t.convert(ch)
    s1_diff = _s2t.convert(s1p) != _s2t.convert(ch)
    if not s2_diff:
        return "reject"          # Qwen3 认为就是 csv 的字 -> 一阶段假阳性
    if s1_diff and _s2t.convert(s2p) == _s2t.convert(s1p):
        return "strong"          # 两模型一致认为不是 csv 的字，且给出的字相同
    return "weak"


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
                prompts.append(PROMPT.format(ch=r.get("character", "")))
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
            tx = proc.apply_chat_template(
                msgs_b, tokenize=False, add_generation_prompt=True)
            inp = proc(text=tx, images=ims, return_tensors="pt",
                       padding=True).to(model.device)
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
        # 优先按 CHAR: 字段取；没有就退回"第一个汉字"（含扩展区）
        m = re.search(r"CHAR\s*[:：]\s*(\S)", raw)
        if m:
            s2p = m.group(1)
        else:
            han = re.findall(HAN_RE, raw)
            s2p = han[0] if han else "?"
        mc = re.search(r"CONF\s*[:：]\s*([0-9]*\.?[0-9]+)", raw)
        s2conf = mc.group(1) if mc else ""
        ch = r.get("character", "")
        s1p = s1.get(i, {}).get("pred", "?")
        v = verdict(ch, s1p, s2p)
        w.writerow([i, r.get("old_50k_id", ""), ch, s1p, s2p, s2conf,
                    int(s2p == ch), int(_s2t.convert(s2p) == _s2t.convert(ch)),
                    int(_s2t.convert(s2p) == _s2t.convert(s1p)), v,
                    raw[:40], r.get("calligrapher", ""), r.get("script", "")])
    fout.flush()
    n_done += len(idxs)
    if (g0 // B + 1) % 10 == 0:
        el = time.time() - t0
        sp = n_done / max(el, 1e-9)
        print(f"    {n_done}/{len(todo)}  {sp:.1f} 条/s  "
              f"ETA {(len(todo)-n_done)/max(sp,1e-9)/3600:.2f}h", flush=True)

fout.close()
print(f"\n  ✓ 完成 {n_done} 条 -> {a.out}", flush=True)
