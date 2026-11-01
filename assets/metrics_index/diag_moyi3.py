"""Verify MoYi GT identity after inverting (panels store inverted GT)."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
CSV = "exp-std/csv/eval200_fixed.csv"
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")


def arr(p, invert=False):
    with Image.open(p) as im:
        a = np.asarray(im.convert("L")).astype(np.float32) / 255.0
    return 1.0 - a if invert else a


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
B = np.stack([arr(Path(r["image_path"])).ravel() for r in rows])

idx, ds = [], []
for i in range(187):
    g = arr(PANEL / f"gt{i}.png", invert=True).ravel()
    d = np.abs(B - g).mean(1)
    j = int(d.argmin())
    idx.append(j)
    ds.append(float(d[j]))

print("mean dist", round(float(np.mean(ds)), 4), "max", round(float(np.max(ds)), 4))
print("exact order:", idx == list(range(187)))
print("unique mapped", len(set(idx)))
print("first10", idx[:10])
print("last5", idx[-5:])
