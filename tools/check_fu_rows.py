"""专查 eval200(原/修正) 里 character 含 复/復 的行, 并把它们的图拼出来看。

目的: 回答"eval 里的复到底改了没有、为什么没改"。
判据(标注来源 + 渲染来源)全部打印, 不猜。
产出: _ot_scratch/fu_rows.png  每行 4 格 = 真迹GT | 原std | 修正std | character重渲染
"""
import csv
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
os.makedirs("_ot_scratch", exist_ok=True)

TARGETS = "复復覆"
FIXED_PNG = "exp-std/data/std_fixed_eval200"

rows = list(csv.DictReader(open("exp-std/csv/eval200.csv", encoding="utf-8")))
fixed = {}
if os.path.exists("exp-std/csv/eval200_fixed.csv"):
    fixed = {r["img_id"]: r for r in csv.DictReader(
        open("exp-std/csv/eval200_fixed.csv", encoding="utf-8"))}

hit = [r for r in rows if any(t in (r.get("character") or "") for t in TARGETS)]
print(f"[eval200] 总 {len(rows)} 行; character 含 {TARGETS} 的 = {len(hit)} 行")
print(f"[fixed]   {len(fixed)} 行")
for r in hit:
    bid = os.path.basename(r["image_path"])
    iid = r["img_id"]
    f = fixed.get(iid)
    print(f"  img_id={iid:>6} character={r['character']!r} slot={r.get('slot_name')} "
          f"src={r.get('source')} std_path={r.get('std_path')}")
    print(f"          修正版收录? {'是' if f else '否(被剔除!)'} "
          f"fix={f.get('fix') if f else '-'} "
          f"新std存在? {os.path.exists(os.path.join(FIXED_PNG, bid))}")

# 若一行都没有, 说明问题不在 character 字段 -> 把"修正前后差异最大"的 6 行拉出来看
if not hit:
    print("\n[空] eval200 的 character 字段里没有 复/復 —— 说明那个字形问题"
          "不是 character 标错, 而是 std 图渲染错了。列 6 行对比:")
    ids = [r["img_id"] for r in rows[:6]]
else:
    ids = [r["img_id"] for r in hit[:6]]

S = 150
canvas = Image.new("L", (S * 4, S * len(ids)), 255)
for j, iid in enumerate(ids):
    r = next(x for x in rows if x["img_id"] == iid)
    bid = os.path.basename(r["image_path"])
    for k, p in enumerate((
            os.path.join("data/top10_style23/imgs", bid),
            os.path.join("data/top10_style23/std", bid),
            os.path.join(FIXED_PNG, bid),
            f"_ot_scratch/_r_{bid}")):
        if os.path.exists(p):
            canvas.paste(Image.open(p).convert("L").resize((S, S)), (k * S, j * S))
canvas.save("_ot_scratch/fu_rows.png")
print("[图] _ot_scratch/fu_rows.png  列 = GT真迹 | 原std | 修正std | (第4列=临时重渲染, 缺则空)")
