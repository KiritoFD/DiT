"""Find which local CSV the MoYi GT set belongs to."""
import csv
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")


def arr(p):
    with Image.open(p) as im:
        return np.asarray(im.convert("L").resize((64, 64))).astype(np.float32).ravel() / 255.0


gts = []
for i in range(187):
    p = PANEL / f"gt{i}.png"
    if p.is_file():
        gts.append(arr(p))
G = np.stack(gts)
print("gt", G.shape)

cands = [p for p in Path(".").rglob("*.csv")
         if "csv" in str(p) and p.stat().st_size > 2000
         and ("eval" in p.name.lower() or "top10" in str(p).lower() or "protocol" in str(p).lower())]
print("csv candidates", len(cands))
best = []
for c in cands:
    try:
        rows = list(csv.DictReader(open(c, encoding="utf-8-sig")))
    except Exception:
        continue
    if not rows or "image_path" not in rows[0]:
        continue
    cols = [r["image_path"] for r in rows[:400]]
    ok = []
    for r in cols[:40]:
        if not Path(r).is_file():
            break
        ok.append(arr(Path(r)))
    if len(ok) < 40:
        continue
    B = np.stack(ok)
    hits = 0
    for g in G[:40]:
        d = np.abs(B - g).mean(1)
        if d.min() < 0.02:
            hits += 1
    best.append((hits, len(rows), str(c)))
best.sort(reverse=True)
for b in best[:6]:
    print(b)
