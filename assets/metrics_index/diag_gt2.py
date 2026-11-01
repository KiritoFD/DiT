"""Find which transform maps panel GT back to the protocol CSV image."""
import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CSV = "exp-std/csv/eval200_fixed.csv"
ROWS = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))


def variants(img):
    a = np.asarray(img)
    out = {"L": hashlib.sha256(img.convert("L").tobytes()).hexdigest()}
    if a.ndim == 2:
        out["RGB_from_L"] = hashlib.sha256(
            Image.fromarray(a).convert("RGB").tobytes()).hexdigest()
    return out


# index CSV by several hash forms
idx = {}
for i, r in enumerate(ROWS):
    with Image.open(r["image_path"]) as im:
        for k, h in variants(im.convert("RGB")).items():
            idx.setdefault((k, h), []).append(i)

d = Path("assets/server_48_evals/moyi_12ch_eval200fix")
with Image.open(d / "gt0.png") as im:
    a = np.asarray(im)
print("gt0 shape", a.shape, "unique vals", np.unique(a)[:6], "min/max", a.min(), a.max())
for k, h in variants(im.convert("RGB")).items():
    print(k, "hit", (k, h) in idx, idx.get((k, h)))

# compare pixel-wise to every csv row (cheap: only first image)
gp = np.asarray(Image.open(d / "gt0.png").convert("L")).astype(np.int16)
best = []
for i, r in enumerate(ROWS[:40]):
    with Image.open(r["image_path"]) as im:
        c = np.asarray(im.convert("L")).astype(np.int16)
    if c.shape == gp.shape:
        best.append((float(np.abs(c - gp).mean()), i, r["character"], r["calligrapher"]))
best.sort()
print("closest csv rows to gt0:", best[:5])
