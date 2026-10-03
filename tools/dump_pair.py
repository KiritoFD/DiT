"""把若干 img_id 的 (std 条件图 | 真迹目标图) 拼成一张图拉回来看。纯 CPU。"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
ids = sys.argv[1:] or ["014577", "020799", "016004", "022702"]
S = 200
canvas = Image.new("L", (S * 2, S * len(ids)), 255)
for j, i in enumerate(ids):
    ps = f"data/top10_style23/std/{i}.png"
    pg = f"data/top10_style23/imgs/{i}.png"
    a = Image.open(ps).convert("L").resize((S, S))
    b = Image.open(pg).convert("L").resize((S, S))
    canvas.paste(a, (0, j * S))
    canvas.paste(b, (S, j * S))
os.makedirs("_ot_scratch", exist_ok=True)
p = "_ot_scratch/pairs_std_gt.png"
canvas.save(p)
print(p)
