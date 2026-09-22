"""拼 poster：把「墨迹 SSIM 最差」的字逐个放出来，检查 std g 对不对。

布局（每个字一行）：
  列1 = GT 书法 | 列2 = std g（印刷骨架） | 列3 = 模型生成
  文字标注: 字 / 书体 / ink_ssim
"""
import csv
import glob
import os
import sys

from PIL import Image, ImageDraw, ImageFont

os.chdir("/root/Workspace/xy/DiT")

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SZ = 150
PAD = 8
TXT_H = 24

rows = list(csv.DictReader(open("assets/eval_low_ink.csv",
                                encoding="utf-8")))[:N]

# 找生成图目录（ink_eval 收集的）
# ⚠ glob "g*.png" 会把 "gt137.png" 也匹配进来 -> 必须用 g[0-9]*.png
gd = sorted(glob.glob("assets/ink_eval/*__strict"))[0]
print(f"  用生成图目录: {gd}")
gen_files = sorted(glob.glob(os.path.join(gd, "g[0-9]*.png")),
                   key=lambda p: int(os.path.basename(p)[1:-4]))
print(f"  生成图 {len(gen_files)} 张")

try:
    font = ImageFont.truetype("_fonts/msyh.ttc", 16)
    font_s = ImageFont.truetype("_fonts/msyh.ttc", 14)
except Exception:
    font = font_s = ImageFont.load_default()

COLS = 3
W = COLS * SZ + (COLS + 1) * PAD
H = N * (SZ + TXT_H + PAD) + 40
canvas = Image.new("RGB", (W, H), "white")
d = ImageDraw.Draw(canvas)
d.text((PAD, 8), f"列1=GT书法  列2=std g(印刷骨架)  列3=模型生成   "
                 f"（按 ink_ssim 升序，最差 {N} 个）", font=font, fill="black")

for i, r in enumerate(rows):
    y0 = 40 + i * (SZ + TXT_H + PAD)
    iid = int(r["img_id"])
    idx = int(r["idx"])
    paths = [f"data/50k/imgs/{iid:06d}.png",
             f"data/50k/std/{iid:06d}.png",
             gen_files[idx] if idx < len(gen_files) else None]
    for k, p in enumerate(paths):
        x0 = PAD + k * (SZ + PAD)
        if p and os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((SZ, SZ), Image.BICUBIC)
            canvas.paste(im, (x0, y0))
        else:
            d.rectangle([x0, y0, x0 + SZ, y0 + SZ], outline="lightgray")
        d.rectangle([x0, y0, x0 + SZ, y0 + SZ], outline="gray")
    d.text((PAD, y0 + SZ + 3),
           f"{r['char']} ({r['script']})  ink_ssim={r['ink']}  "
           f"iou={r['iou']}  ssim={r['ssim']}  id={r['img_id']}",
           font=font_s, fill="black")

out = "assets/poster_low_ink.png"
canvas.save(out)
print(f"  ✓ -> {out}  {canvas.size}")
