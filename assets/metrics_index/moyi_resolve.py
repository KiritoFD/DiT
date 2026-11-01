"""Resolve the 50 disagreements: OCR misread, or genuinely different characters?"""
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
DEC = json.loads(Path("assets/metrics_index/moyi_gt_decoded.json").read_text())
pred, conf = DEC["pred"], DEC["conf"]
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))

bad = [(i, pred[i], rows[i]["character"], conf[i]) for i in range(187)
       if pred[i] != rows[i]["character"]]
print(f"disagreements {len(bad)}; low-confidence(<0.5) among them: "
      f"{sum(1 for b in bad if b[3] < 0.5)}")
print("\nfirst 25 disagreements (idx, decoded, canonical, conf):")
for b in bad[:25]:
    print(f"  {b[0]:3d}  {b[1]}  vs  {b[2]}   conf={b[3]:.2f}")

# Structural test: is the GT image closer to the canonical image of its OWN row
# than to a random other row? If yes, the order is right and only the label was misread.
def arr(p, size=64):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((size, size))).astype(np.float32).ravel() / 255.0

diffs_own = []
for i, _, _, _ in bad[:20]:
    g = arr(PANEL / f"gt{i}.png")
    own = arr(Path(rows[i]["image_path"]))
    others = np.stack([arr(Path(rows[j]["image_path"]))
                       for j in range(0, 187, 7) if j != i])
    d_own = float(np.abs(g - own).mean())
    d_min_other = float(np.abs(others - g).mean(1).min())
    diffs_own.append(d_own <= d_min_other)
print(f"\namong 20 disagreements: GT closer to its own row than to any other row: "
      f"{sum(diffs_own)}/20")
