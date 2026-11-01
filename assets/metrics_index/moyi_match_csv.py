"""Match MoYi's decoded 187 characters against every candidate protocol file.

If a candidate's character multiset (or ordered prefix) matches, that CSV supplies
the writer labels MoYi lost, which is what E-G needs.
"""
import csv
import io
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
DEC = json.loads(Path("assets/metrics_index/moyi_gt_decoded.json").read_text())
pred = DEC["pred"]
P = Counter(pred)
print("decoded", len(pred), "unique", len(P))

cands = [
    "exp-std/csv/eval200_fixed.csv",
    "exp-std/csv/eval200.csv",
    "exp-std/csv/eval200_combo_unseen.csv",
    "exp-std/csv/eval200_split_mid.csv",
    "exp-std/csv/eval200_split_top.csv",
    "exp-std/csv/eval_top2_100.csv",
    "exp-std/csv/eval5_top30.csv",
]
hits = []
for c in cands:
    p = Path(c)
    if not p.is_file():
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    chars = [r["character"] for r in rows]
    C = Counter(chars)
    inter = sum((P & C).values())
    pos = sum(1 for a, b in zip(pred, chars) if a == b)
    hits.append((pos, inter, len(rows), c))
    print(f"{c:44s} n={len(rows):4d} positional={pos:3d} multiset_overlap={inter:3d}")

hits.sort(reverse=True)
print("\nbest by positional:", hits[0][3] if hits else None)
