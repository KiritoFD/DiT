"""二值化前后对照表: 每个源取若干张, 左=原图, 右=二值化产物。"""
import os, random
import numpy as np
from PIL import Image, ImageDraw

HCSU = "/root/Workspace/xy/HCSU"
OUT = f"{HCSU}/_bin_contact.png"
random.seed(21)

SRC = {"wild": f"{HCSU}/wild_extract", "tie": f"{HCSU}/tie_extract", "bei": f"{HCSU}/bei_extract"}
TH = 130
COLS = 10

def files_of(root):
    out = []
    for d in sorted(os.listdir(root)):
        dp = os.path.join(root, d)
        if os.path.isdir(dp):
            for f in sorted(os.listdir(dp)):
                if f.lower().endswith(".png"):
                    out.append((d, f))
    return out

rows_total = 0
plan = []
for tag in ["wild", "tie", "bei"]:
    fs = files_of(SRC[tag])
    # 分反相/正相两组各取若干
    inv, pos = [], []
    for d, f in random.sample(fs, min(1500, len(fs))):
        p = os.path.join(SRC[tag], d, f)
        if not os.path.exists(os.path.join(f"{HCSU}/_bin", tag, d, f)):
            continue
        a = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        b = float(np.concatenate([a[0, :], a[-1, :], a[:, 0], a[:, -1]]).mean())
        nd = int((a < b - 0.25).sum()); nl = int((a > b + 0.25).sum())
        (inv if nl > nd else pos).append((d, f))
    plan.append((tag, "反相(inverted)", inv[:COLS]))
    plan.append((tag, "正相(as-is)", pos[:COLS]))

nrow = len(plan)
sheet = Image.new("RGB", (COLS * TH, nrow * 2 * TH + nrow * 20), "white")
dr = ImageDraw.Draw(sheet)
y = 0
for tag, kind, lst in plan:
    dr.text((4, y + 3), f"{tag}  |  {kind}   (left = original, right = binarized)", fill=(200, 0, 0))
    y += 20
    for i, (d, f) in enumerate(lst):
        o = Image.open(os.path.join(SRC[tag], d, f)).convert("L").resize((TH, TH), Image.LANCZOS)
        sheet.paste(o.convert("RGB"), (i * TH, y))
        bp = os.path.join(f"{HCSU}/_bin", tag, d, f)
        if os.path.exists(bp):
            b = Image.open(bp).convert("L").resize((TH, TH), Image.NEAREST)
            sheet.paste(b.convert("RGB"), (i * TH, y + TH))
    y += 2 * TH
sheet.save(OUT)
print("saved", OUT, sheet.size)
for tag, kind, lst in plan:
    print(f"  {tag}/{kind}: {len(lst)} 例  {[f'{d}/{f}' for d,f in lst[:3]]}")
