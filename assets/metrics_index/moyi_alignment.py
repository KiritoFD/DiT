"""Decisive test: is MoYi GT the same artwork as the eval200_fixed row at the same index?

If a substantial majority of the 187 GT images are near-identical to the canonical
image at their own index, then the row alignment holds and eval200_fixed supplies
correct writer labels (the 50 char mismatches are OCR misreads, already shown to be
low-confidence).
"""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))


def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0


canon = np.stack([arr(Path(r["image_path"])) for r in rows])
dists_same, ranks = [], []
for i in range(187):
    g = arr(PANEL / f"gt{i}.png")
    d = np.abs(canon - g).mean(1)
    dists_same.append(float(d[i]))
    ranks.append(int((d < d[i]).sum()))
dists_same = np.array(dists_same)
print(f"GT vs canonical image at SAME index: mean={dists_same.mean():.4f} median={np.median(dists_same):.4f}")
print(f"exact-rank==0 (self is the nearest): {sum(1 for r in ranks if r == 0)}/187")
print(f"self within top-3: {sum(1 for r in ranks if r <= 2)}/187")
print(f"dist < 0.02 (near identical): {(dists_same < 0.02).sum()}/187")
print(f"dist < 0.05: {(dists_same < 0.05).sum()}/187")
