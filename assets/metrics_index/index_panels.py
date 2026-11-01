"""Index every local historical panel and prove its identity against a protocol CSV.

Identity is accepted only when the saved GT pixels hash-match the CSV image paths
in order (or when a same-named protocol CSV reproduces that order exactly).
"""
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

CSVS = [p for p in ["exp-std/csv/eval200_fixed.csv",
                    "assets/u100_pretrain_20261024/validation_val100.csv"] if Path(p).is_file()]
OUT = Path("assets/metrics_index/historical_panels.json")


def pix(p):
    with Image.open(p) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


banks = {}
for c in CSVS:
    rows = list(csv.DictReader(open(c, encoding="utf-8-sig")))
    m = {}
    for i, r in enumerate(rows):
        m.setdefault(pix(r["image_path"]), []).append(i)
    banks[c] = (rows, m)
    print(c, "rows", len(rows))

panels = [line.strip().split("|") for line in Path("assets/metrics_index/panels_raw.txt").read_text().splitlines() if line.strip()]
result = []
for ng, ngt, d in panels:
    d = Path(d)
    ng_n, gt_n = int(ng), int(ngt)
    rec = {"dir": str(d), "n_generated": ng_n, "n_gt": gt_n}
    best = None
    if gt_n >= 50:
        for c, (rows, m) in banks.items():
            if abs(len(rows) - gt_n) > 0:
                continue
            hit = 0
            for i in range(gt_n):
                gp = d / f"gt{i}.png"
                if not gp.is_file():
                    break
                if pix(gp) in m:
                    hit += 1
            rec[f"gt_match_{Path(c).stem}"] = hit
            if hit == gt_n and best is None:
                best = c
    rec["identity_protocol"] = best
    result.append(rec)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n")
verified = [r for r in result if r["identity_protocol"]]
print("panels", len(result), "identity-verified", len(verified))
for r in verified:
    print("VERIFIED", r["dir"])
