"""Diagnose GT pixel matching for local panels."""
import csv
import hashlib
from pathlib import Path

from PIL import Image

CSV = "exp-std/csv/eval200_fixed.csv"


def pix(p):
    with Image.open(p) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
m = {}
for r in rows:
    m.setdefault(pix(r["image_path"]), []).append(r)
print("csv unique pixel hashes", len(m), "of", len(rows))

for d in ["assets/server_48_evals/moyi_12ch_eval200fix",
          "assets/v85_verified_v84A_200k",
          "assets/style_eval_src/v88_step0050000"]:
    d = Path(d)
    hits = sum(1 for i in range(187) if (d / f"gt{i}.png").is_file() and pix(d / f"gt{i}.png") in m)
    print(d.name, "gt hits", hits, "/187")
    # also try comparing shape/hash of the CSV std_path
    gp = d / "gt0.png"
    if gp.is_file():
        with Image.open(gp) as im:
            print("   gt0 size/mode", im.size, im.mode)
        r0 = rows[0]
        with Image.open(r0["image_path"]) as im:
            print("   csv0 size/mode", im.size, im.mode)
        with Image.open(r0["std_path"]) as im:
            print("   std0 size/mode", im.size, im.mode, "hash==gt0",
                  pix(Path(r0["std_path"])) == pix(gp))
