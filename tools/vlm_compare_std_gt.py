#!/usr/bin/env python
"""用 VLM 判断「std g 渲染的字」vs「GT 实际写的字」是否同一个字。

## 为什么用「比较」而不是「识别」
直接识别书法单字准确率只有 25%（实测）——瓶颈是 VLM 认不准书法。
但**比较两个字形**不需要识别，只需形状匹配，绕过了字表/书法识别瓶颈。

## 输出三件事（按用户要求）
  1. same    : 是不是同一个字（A=是 / B=否 / C=不确定）
  2. conf    : 置信度 0-1
  3. gt_char : VLM 认为 GT 写的是什么字

## thinking
Qwen3-VL 支持 `enable_thinking=True`（先输出思考再给答案）。
Qwen2.5-VL 不支持 —— 用 --thinking 1 时请配 --model 指向 Qwen3-VL。

用法:
  CUDA_VISIBLE_DEVICES=0 ./_venv_qwenvl/bin/python tools/vlm_compare_std_gt.py \
      --csv assets/eval_v13_strict_fixed.csv --device cuda --thinking 1
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

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="assets/eval_v13_strict_fixed.csv")
ap.add_argument("--model", default="/root/Workspace/xy/DiT/_models/models/"
                                   "Qwen--Qwen3-VL-4B-Instruct/snapshots/master")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="")
ap.add_argument("--device", default="cuda")
ap.add_argument("--offload", type=int, default=0,
                help="1=大模型用 device_map=auto 把权重放 RAM")
ap.add_argument("--thinking", type=int, default=1,
                help="1=开思考（仅 Qwen3-VL 支持）")
ap.add_argument("--max-new", type=int, default=512)
a = ap.parse_args()

rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
if a.limit:
    rows = rows[:a.limit]
tag = "think" if a.thinking else "nothink"
out = a.out or (f"assets/vlm_cmp/"
                f"{os.path.splitext(os.path.basename(a.csv))[0]}__{tag}.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
print(f"  {a.csv}: {len(rows)} 条 -> {out}  thinking={a.thinking}", flush=True)

PROMPT = (
    "这是一张拼接图：左边是印刷体（某个汉字的标准写法），"
    "右边是书法家手写的单字（同一书家、同一书体）。\n\n"
    "已知左边印刷体的字是「{ch}」。请**仔细比对**右边的书法，"
    "回答右边写的**是不是**「{ch}」这个字。\n\n"
    "比对方法（请逐条思考）：\n"
    "1. 先看右边的**整体结构**（左右/上下/独体）和左边是否一致\n"
    "2. 再看**主要部件**（如'亻''口''木''氵'）是否都能对应上\n"
    "3. 再数**笔画数**是否接近（书法有连笔，±2 笔内算一致）\n"
    "4. 特别注意：书法常把**繁体写成简体**、或写成**异体字**"
    "（如'陞'写成'升'、'昇'写成'升'）—— 这类**算不同字**\n\n"
    "最后必须按下面格式输出三行（不要输出其它内容）：\n"
    "SAME: A 或 B 或 C   （A=完全同一个字, B=不是同一个字, C=无法判断）\n"
    "CONF: 0.0~1.0 的数字 （你对 SAME 判断的置信度）\n"
    "GT_CHAR: 你认为右边书法写的那个字（一个汉字；认不出写 ？）"
)


def make_pair(std_p, gt_p, size=256, gap=16):
    a1 = Image.open(std_p).convert("RGB").resize((size, size))
    a2 = Image.open(gt_p).convert("RGB").resize((size, size))
    c = Image.new("RGB", (size * 2 + gap, size), "white")
    c.paste(a1, (0, 0))
    c.paste(a2, (size + gap, 0))
    d = ImageDraw.Draw(c)
    d.line([(size + gap // 2, 0), (size + gap // 2, size)], fill="gray", width=2)
    return c


from transformers import AutoProcessor  # noqa: E402

# ⚠ 必须**按模型路径**选类，不能靠 try-import：
#   - Qwen3VLForConditionalGeneration 在当前 transformers 里存在，
#     用它加载 qwen2_5_vl 权重会报
#       "You are using a model of type qwen2_5_vl to instantiate a model of type qwen3_vl"
#     然后卡死（实测 GPU 0%）
#   - 30B-A3B 是 **MoE**（model_type=qwen3_vl_moe），用 dense 类加载会丢掉全部
#     experts.* 权重、把 dense mlp 全新初始化 -> 输出全是 '?'（实测）
_low = a.model.lower()
if "moe" in _low or "-a3b" in _low or "a22b" in _low:
    from transformers import Qwen3VLMoeForConditionalGeneration as M
    _is_qwen3 = True
    print("  [model-class] Qwen3VLMoeForConditionalGeneration (MoE)", flush=True)
elif "qwen3" in _low:
    from transformers import Qwen3VLForConditionalGeneration as M
    _is_qwen3 = True
elif "qwen2_5" in _low or "qwen2.5" in _low:
    from transformers import Qwen2_5_VLForConditionalGeneration as M
    _is_qwen3 = False
else:
    try:
        from transformers import Qwen3VLForConditionalGeneration as M
        _is_qwen3 = True
    except ImportError:
        from transformers import Qwen2_5_VLForConditionalGeneration as M
        _is_qwen3 = False

print(f"  加载 {a.model} (qwen3={_is_qwen3}) ...", flush=True)
kw = dict(low_cpu_mem_usage=True)
kw["dtype"] = torch.float16 if a.device == "cuda" else torch.float32
if a.offload:
    # MoE + 大 RAM：整模型放 RAM，按需把激活的 expert 搬到 GPU。
    # 30B-A3B 激活只 3B，速度可接受，且不用量化。
    kw["device_map"] = "auto"
    kw["max_memory"] = {0: "20GiB", "cpu": "200GiB"}
    m = M.from_pretrained(a.model, **kw).eval()
    print(f"  [offload] hf_device_map 层数={len(getattr(m, chr(104)+chr(102)+chr(95)+chr(100)+chr(101)+chr(118)+chr(105)+chr(99)+chr(101)+chr(95)+chr(109)+chr(97)+chr(112), {}))}", flush=True)
else:
    m = M.from_pretrained(a.model, **kw).to(a.device).eval()
pr = AutoProcessor.from_pretrained(a.model)

TMPL_KW = {}
if _is_qwen3:
    TMPL_KW = {"enable_thinking": bool(a.thinking)}


def parse(raw):
    same, conf, gtc = "?", "", ""
    mm = re.search(r"SAME\s*[:：]\s*([ABCabc])", raw)
    if mm:
        same = mm.group(1).upper()
    mm = re.search(r"CONF\s*[:：]\s*([0-9]*\.?[0-9]+)", raw)
    if mm:
        try:
            conf = float(mm.group(1))
        except ValueError:
            conf = ""
    mm = re.search(r"GT_CHAR\s*[:：]\s*(\S)", raw)
    if mm:
        gtc = mm.group(1)
    return same, conf, gtc


res = []
for i, r in enumerate(rows):
    iid = r.get("old_50k_id", "").strip()
    ch = r.get("character", "")
    std_p = r.get("std_path", "").strip() or f"data/50k/std/{int(iid):06d}.png"
    gt_p = f"data/50k/imgs/{int(iid):06d}.png"
    same, conf, gtc, raw = "?", "", "", ""
    if os.path.exists(std_p) and os.path.exists(gt_p):
        try:
            im = make_pair(std_p, gt_p)
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": im},
                {"type": "text", "text": PROMPT.format(ch=ch)}]}]
            tx = pr.apply_chat_template(msgs, tokenize=False,
                                        add_generation_prompt=True, **TMPL_KW)
            inp = pr(text=[tx], images=[im], return_tensors="pt")
            if not a.offload:
                inp = inp.to(a.device)
            with torch.no_grad():
                g = m.generate(**inp, max_new_tokens=a.max_new,
                               do_sample=False)
            raw = pr.batch_decode(g[:, inp["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
            same, conf, gtc = parse(raw)
        except Exception as e:
            same, raw = "ERR", f"{type(e).__name__}:{str(e)[:60]}"
    res.append({"idx": i, "char": ch, "same": same, "conf": conf,
                "gt_char": gtc, "callig": r.get("calligrapher", ""),
                "script": r.get("script", ""),
                "src": r.get("src_image_path", ""), "raw": raw[:200]})
    if (i + 1) % 5 == 0:
        print(f"    {i+1}/{len(rows)}", flush=True)

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(res[0].keys()))
    w.writeheader()
    w.writerows(res)

from collections import Counter  # noqa: E402
from opencc import OpenCC  # noqa: E402

_c2t = OpenCC("s2t")
c = Counter(x["same"] for x in res)
print(f"\n  SAME 分布: {dict(c)}")
print(f"  conf 分布: {dict(Counter(x['conf'] for x in res).most_common(5))}")


def _norm(s):
    """归一化：简->繁，这样 贫/貧、刚/剛、臥/卧 都算一致。"""
    return _c2t.convert(s) if s else s


# ★ 主判据: GT_CHAR 与 character 在**简繁归一化后**仍不同
#   SAME 字段实测不可靠（会自相矛盾：GT_CHAR 说'堂'却判 B）
#   GT_CHAR 在给了「已知左边是 X」的上下文锚点后变可靠（实测认出了「陞」）
bad = [x for x in res
       if x["gt_char"] not in ("", "？", "?", "None")
       and _norm(x["gt_char"]) != _norm(x["char"])]
eq = [x for x in res
      if x["gt_char"] not in ("", "？", "?", "None")
      and _norm(x["gt_char"]) == _norm(x["char"])
      and x["gt_char"] != x["char"]]
print(f"\n  ★ 归一化后仍不同（可疑）: {len(bad)}/{len(res)} "
      f"({len(bad)/max(len(res),1)*100:.1f}%)")
print(f"  （其中 {len(eq)} 条只是简繁差异，已排除）")
for x in bad[:60]:
    print(f"    {x['char']:>2} -> vlm认为={x['gt_char']}  conf={x['conf']}  "
          f"({x['callig']}/{x['script']})")
print(f"\n  -> {out}")
