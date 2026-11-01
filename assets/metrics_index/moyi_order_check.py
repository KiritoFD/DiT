"""Compare the decoded MoYi GT characters against eval200_fixed to recover ordering."""
import csv
import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
dec = json.loads(Path("assets/metrics_index/moyi_gt_decoded.json").read_text())
pred = dec["pred"]
rows = list(csv.DictReader(open("exp-std/csv/eval200_fixed.csv", encoding="utf-8-sig")))
canon = [r["character"] for r in rows]

same = sum(1 for a, b in zip(pred, canon) if a == b)
print(f"positional match vs canonical order: {same}/187")
# where do they diverge?
first_bad = next((i for i, (a, b) in enumerate(zip(pred, canon)) if a != b), None)
print("first mismatch at", first_bad)
if first_bad is not None:
    print("  pred :", "".join(pred[first_bad:first_bad + 12]))
    print("  canon:", "".join(canon[first_bad:first_bad + 12]))

# is it a permutation? check set overlap
ps, cs = set(pred), set(canon)
print(f"set overlap: {len(ps & cs)} chars; pred-only {len(ps - cs)}; canon-only {len(cs - ps)}")
print("pred-only sample:", list(ps - cs)[:10])
