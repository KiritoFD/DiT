"""拼一张接触表: 按底色分三档(白/灰/黑)各取若干, 每张附其"反相"版本, 供目视判断。"""
import os, random, collections
import numpy as np
from PIL import Image

ROOT = "/root/Workspace/xy/HCSU/wild_extract"
OUT = "/root/Workspace/xy/HCSU/_contact.png"
random.seed(1)

files = []
for d in sorted(os.listdir(ROOT)):
    dp = os.path.join(ROOT, d)
    if os.path.isdir(dp):
        for f in os.listdir(dp):
            if f.lower().endswith(".png"):
                files.append((d, f))

buckets = {"white": [], "gray": [], "black": []}
for d, f in random.sample(files, 6000):
    try:
        im = Image.open(os.path.join(ROOT, d, f)).convert("RGB")
    except Exception:
        continue
    a = np.asarray(im, dtype=np.float32) / 255.0
    g = a.mean(axis=2)
    b = float(np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]]).mean())
    k = "white" if b > 0.7 else ("black" if b < 0.3 else "gray")
    if len(buckets[k]) < 6:
        buckets[k].append((d, f))
    if all(len(v) >= 6 for v in buckets.values()):
        break

TH = 190
cols, rows = 12, 3
sheet = Image.new("RGB", (cols * TH, rows * TH * 2 + 40), "white")
from PIL import ImageDraw
dr = ImageDraw.Draw(sheet)
y = 0
for bi, k in enumerate(["white", "gray", "black"]):
    dr.text((4, y + 2), f"--- {k} background (left: original | right: inverted) ---", fill="red")
    y += 20
    for ci, (d, f) in enumerate(buckets[k]):
        im = Image.open(os.path.join(ROOT, d, f)).convert("L").resize((TH, TH), Image.LANCZOS)
        sheet.paste(im.convert("RGB"), (ci * TH, y))
        sheet.paste(Image.fromarray(255 - np.asarray(im)).convert("RGB"), (ci * TH, y + TH))
    y += TH * 2
sheet.save(OUT)
print("saved", OUT, sheet.size)
for k, v in buckets.items():
    print(f"  {k}: {[f'{d}/{f}' for d, f in v[:3]]}")
