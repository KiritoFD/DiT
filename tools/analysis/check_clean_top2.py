import csv
from pathlib import Path
from PIL import Image
import numpy as np
from scipy.ndimage import label

r2 = list(csv.DictReader(open("exp-std/csv/eval_top2_100.csv", encoding="utf-8-sig")))

clean_top2 = []
for i, r in enumerate(r2):
    p = Path(r["image_path"])
    if not p.exists():
        continue
    im = Image.open(p).convert("L")
    arr = np.array(im)
    ink = arr < 128
    
    # 15px border check
    m_top = int(ink[:12, :].sum())
    m_bot = int(ink[-12:, :].sum())
    m_left = int(ink[:, :12].sum())
    m_right = int(ink[:, -12:].sum())
    touches = bool(m_top > 0 or m_bot > 0 or m_left > 0 or m_right > 0)
    
    lbl, num = label(ink)
    sizes = np.bincount(lbl.ravel())[1:] if num > 0 else []
    debris = int(sum(s < 40 for s in sizes))
    
    if not touches and num <= 14 and debris <= 3 and ink.sum() > 2500:
        clean_top2.append((i, r["calligrapher"], r["script"], r["character"], int(ink.sum()), num, debris, p))

print(f"Clean samples in Top2 (out of 100): {len(clean_top2)}")
for x in clean_top2[:25]:
    print(f"  #{x[0]:02d} {x[1]} {x[2]} [{x[3]}]: ink={x[4]}, comp={x[5]}, debris={x[6]}")
