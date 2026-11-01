"""Check whether MoYi GT and eval200_fixed are the same artworks under a rendering shift."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32) / 255.0


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
# structural comparison: downsampled + binarized + IoU, plus aspect-ratio check
def feat(p):
    a = arr(p)
    b = (a > 0.5).astype(np.float32)
    return a, b


for i in [0, 1, 2, 5]:
    g = PANEL / f"gt{i}.png"
    ga, gb = feat(g)
    best = []
    for j, r in enumerate(rows):
        ca, cb = feat(Path(r["image_path"]))
        iou = (gb * cb).sum() / max(1e-6, ((gb + cb) > 0).sum())
        best.append((iou, j, r["character"], r["calligrapher"]))
    best.sort(reverse=True)
    print("gt", i, "best IoU", [(round(b[0], 3), b[2], b[3]) for b in best[:3]])
    with Image.open(g) as im:
        print("   gt size", im.size, "ink frac", round(float(gb.mean()), 4))
