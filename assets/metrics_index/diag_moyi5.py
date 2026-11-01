"""Structure-based identification for MoYi GT: binary-shape IoU over the local library."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
INDEX = "assets/data_clean/top10_20261021_v1/scan/inputs/train.csv"


def shape(p, size=64, invert=False):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L").resize((size, size))).astype(np.float32) / 255.0
    if invert:
        a = 1.0 - a
    return (a < 0.5).astype(np.float32).ravel()


def best_for(g, tag):
    inter = (B * g).sum(1)
    union = ((B + g) > 0).sum(1)
    iou = inter / np.maximum(union, 1)
    j = int(iou.argmax())
    print(f"  {tag}: IoU {iou[j]:.3f} -> {rows[j]['character']} / {rows[j]['calligrapher']}")
    return iou[j], j


rows = [r for r in csv.DictReader(open(INDEX, encoding="utf-8-sig")) if Path(r["image_path"]).is_file()]
B = np.stack([shape(Path(r["image_path"])) for r in rows])
print("bank", B.shape, "mean ink", round(float(B.mean()), 4))

for i in range(6):
    print(f"gt{i}:")
    best_for(shape(PANEL / f"gt{i}.png"), "normal ")
    best_for(shape(PANEL / f"gt{i}.png", invert=True), "inverted")
