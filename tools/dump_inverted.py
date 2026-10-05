"""把审计出的反色(拓片)样本拼一张图 (左=条件 std, 右=真迹目标)。纯 CPU。"""
import os

import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
IDS = ["019124", "016860", "019079", "019080", "019199",
       "019917", "021516", "019209", "017969", "020666"]
S = 150
cols = 2
rows = (len(IDS) + cols - 1) // cols
canvas = Image.new("L", (S * 2 * cols, S * rows), 255)
for j, i in enumerate(IDS):
    ps = f"data/top10_style23/std/{i}.png"
    pg = f"data/top10_style23/imgs/{i}.png"
    if not (os.path.exists(ps) and os.path.exists(pg)):
        continue
    r, c = j // cols, j % cols
    canvas.paste(Image.open(ps).convert("L").resize((S, S)), (c * 2 * S, r * S))
    canvas.paste(Image.open(pg).convert("L").resize((S, S)), (c * 2 * S + S, r * S))
os.makedirs("_ot_scratch", exist_ok=True)
p = "_ot_scratch/inverted_10.png"
canvas.save(p)
print(p)
