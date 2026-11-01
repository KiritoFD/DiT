"""Verify MoYi 12ch panel identity by exact GT-pixel matching against the 187-row CSV."""
import csv
import hashlib
from pathlib import Path

from PIL import Image

PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")
CSV = Path("exp-std/csv/eval200_fixed.csv")


def pix(p):
    with Image.open(p) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


rows = list(csv.DictReader(CSV.open(encoding="utf-8-sig")))
print("csv rows", len(rows))
lookup = {}
for r in rows:
    lookup.setdefault(pix(r["image_path"]), []).append(r)
print("unique gt pixels", len(lookup))

matched, unmatched, ambiguous = 0, [], []
for i in range(len(rows)):
    gp = PANEL / f"gt{i}.png"
    if not gp.is_file():
        unmatched.append((i, "no gt"))
        continue
    h = pix(gp)
    if h in lookup and len(lookup[h]) == 1:
        matched += 1
    elif h in lookup:
        ambiguous.append(i)
    else:
        unmatched.append((i, "no pixel match"))
print("gt matched", matched, "ambiguous", len(ambiguous), "unmatched", len(unmatched), unmatched[:5])
