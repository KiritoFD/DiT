"""Check whether the 111 self-matching GT images are a rendering variant or a coincidence."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))


def raw(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L")).astype(np.float32) / 255.0


def prep(a, mode):
    if mode == "norm":
        lo, hi = np.percentile(a, 5), np.percentile(a, 95)
        return np.clip((a - lo) / max(1e-6, hi - lo), 0, 1)
    if mode == "bin":
        return (a < np.percentile(a, 30)).astype(np.float32)
    return a


def feat(a, mode, size=64):
    im = Image.fromarray((prep(a, mode) * 255).astype(np.uint8))
    im = im.resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(im, dtype=np.float32).ravel() / 255.0


for mode in ["norm", "bin"]:
    canon = np.stack([feat(raw(Path(r["image_path"])), mode) for r in rows])
    ranks = []
    for i in range(187):
        g = feat(raw(PANEL / f"gt{i}.png"), mode)
        d = np.abs(canon - g).mean(1)
        ranks.append(int((d < d[i]).sum()))
    self0 = sum(1 for r in ranks if r == 0)
    print(f"mode={mode:5s} self-nearest {self0}/187, top3 {sum(1 for r in ranks if r <= 2)}/187")

# Also: compare MoYi GT with the GT that MoYi's own eval produced (both from same pipeline)
same_shape = 0
for i in range(10):
    with Image.open(PANEL / f"gt{i}.png") as a, Image.open(rows[i]["image_path"]) as b:
        if a.size == b.size:
            same_shape += 1
print("same size as canonical:", same_shape, "/10")
