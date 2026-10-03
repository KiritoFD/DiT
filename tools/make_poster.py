"""把一次 in_mem_eval 的结果拼成**列数可控、看得清**的对比海报。

为什么另写: in_mem_eval 自带的 poster 在 eval200(187 列) 下宽 47872px, 缩略后完全看不清。
行: 标准字 std(模型输入) | 模型生成 | 真迹 GT   —— 列随机采样 K 列, 每列标出字。
用法: python tools/make_poster.py --dir exp-std/reeval/<run>__<step> --set eval200 --cols 16
"""
import argparse
import csv
import glob
import os

from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--set", default="eval200")
ap.add_argument("--cols", type=int, default=16)
ap.add_argument("--step", default="")
ap.add_argument("--csv", default="", help="该 set 的清单(取字用), 留空自动猜")
ap.add_argument("--out", default="")
a = ap.parse_args()

steps = sorted(glob.glob(os.path.join(a.dir, "eval_samples_ctrl", "step*")))
if not steps:
    raise SystemExit(f"没找到 {a.dir}/eval_samples_ctrl/step*")
step = a.step or steps[-1].split("step")[-1]
gdir = os.path.join(a.dir, "eval_samples_ctrl", f"step{int(step):07d}", a.set)
# ★ in_mem_eval 把 seen 集写在子目录 `g` 下 (见 in_mem_eval.py:907 _sub = "g" if name in ("seen","g"))
if not os.path.isdir(gdir):
    alt = os.path.join(a.dir, "eval_samples_ctrl", f"step{int(step):07d}", "g")
    if os.path.isdir(alt):
        gdir = alt
    else:
        raise SystemExit(f"没找到 {gdir}")
idir = os.path.join(a.dir, "eval_samples_ctrl", f"{a.set}_input_g")

csvp = a.csv or {"eval200": "exp-std/csv/eval200_fixed.csv",
                 "seen": "exp-std/csv/seen20.csv"}.get(a.set, "")
chars = []
if csvp and os.path.exists(csvp):
    chars = [r.get("character", "") for r in csv.DictReader(open(csvp, encoding="utf-8"))]

ok = sorted(int(os.path.basename(p)[1:-4])
            for p in glob.glob(os.path.join(gdir, "g[0-9]*.png")))
if not ok:
    raise SystemExit(f"{gdir} 里没有 g*.png")
K = min(a.cols, len(ok))
idx = ([ok[int(round(i * (len(ok) - 1) / (K - 1)))] for i in range(K)] if K > 1
       else ok[:1])
idx = list(dict.fromkeys(idx))            # 去重保持顺序

S, BAND, TOP = 224, 26, 26
rows = [("std 输入", idir, "g"), ("模型生成", gdir, "g"), ("真迹 GT", gdir, "gt")]
H = TOP + S * len(rows) + BAND
canvas = Image.new("RGB", (S * len(idx), H), (255, 255, 255))
d = ImageDraw.Draw(canvas)
f_big = ImageFont.truetype("_fonts/msyh.ttc", 20)
f_lab = ImageFont.truetype("_fonts/msyh.ttc", 16)

d.text((6, 4), f"{os.path.basename(a.dir.rstrip('/'))}   set={a.set}   step={int(step)}   "
               f"n={len(ok)} (显示 {len(idx)} 列)", font=f_big, fill=(0, 0, 0))
for r, (lab, dd, pre) in enumerate(rows):
    y = TOP + r * S
    for c, i in enumerate(idx):
        p = os.path.join(dd, f"{pre}{i}.png")
        if os.path.exists(p):
            canvas.paste(Image.open(p).convert("RGB").resize((S, S)), (c * S, y))
    d.text((4, y + 4), lab, font=f_lab, fill=(200, 0, 0))
for c, i in enumerate(idx):
    ch = chars[i] if i < len(chars) else ""
    d.text((c * S + S // 2 - 8, TOP + S * len(rows) + 2), ch, font=f_lab, fill=(0, 0, 0))

out = a.out or os.path.join(a.dir, "posters", f"{a.set}_readable.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
canvas.save(out)
print(f"[out] {out}  size={canvas.size}  每列: " +
      " ".join(chars[i] if i < len(chars) else "?" for i in idx))
