# -*- coding: utf-8 -*-
"""Parse /tmp/s19_midclean_train.log for Steps/Sec trend over time."""
import re, sys

log = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s19_midclean_train.log"
rows = []
with open(log, encoding="utf-8", errors="replace") as f:
    for line in f:
        m = re.search(r"step=(\d+)\).*?Steps/Sec:\s*([0-9.]+)", line)
        if m:
            rows.append((int(m.group(1)), float(m.group(2))))

print(f"log entries with steps/s: {len(rows)}")
if not rows:
    sys.exit(0)
span = max(1, len(rows) // 24)  # sample ~24 points across the run
print(f"{'step':>10} {'steps/s':>8}")
for i in range(0, len(rows), span):
    print(f"{rows[i][0]:>10} {rows[i][1]:>8.2f}")
print(f"{rows[-1][0]:>10} {rows[-1][1]:>8.2f}")
lo = min(r[1] for r in rows); hi = max(r[1] for r in rows)
print(f"min={lo:.2f} max={hi:.2f} mean={sum(r[1] for r in rows)/len(rows):.2f}")
# first 20 vs last 20
first = [r[1] for r in rows[:20]]; last = [r[1] for r in rows[-20:]]
print(f"first20 avg={sum(first)/len(first):.2f} last20 avg={sum(last)/len(last):.2f}")