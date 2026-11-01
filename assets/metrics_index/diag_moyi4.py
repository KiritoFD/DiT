"""Structural check: why does MoYi GT not match the local library?"""
import io
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
REF = Path("data/top10_style23/imgs")

for name in ["gt0.png", "g0.png", "gt1.png", "gt100.png"]:
    p = PANEL / name
    if not p.exists():
        continue
    with Image.open(p) as im:
        a = np.asarray(im.convert("L"))
    print(name, "size", a.shape, "nonzero", round(float((a > 10).mean()), 4),
          "mean", round(float(a.mean()), 1), "unique", len(np.unique(a)))

# a known-good reference for comparison
for q in list(REF.glob("*.png"))[:2]:
    with Image.open(q) as im:
        a = np.asarray(im.convert("L"))
    print(q.name, "size", a.shape, "nonzero", round(float((a > 10).mean()), 4),
          "mean", round(float(a.mean()), 1))

# ink histogram of GT0 vs its own generated g0
with Image.open(PANEL / "g0.png") as im:
    g = np.asarray(im.convert("L"))
print("g0 nonzero", round(float((g > 10).mean()), 4), "mean", round(float(g.mean()), 1))
