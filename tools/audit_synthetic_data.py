import csv, os, sys
from collections import defaultdict, Counter

csv_path = "assets/train_top10_style23.csv"
with open(csv_path, "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Group by slot_name
slot_stats = defaultdict(lambda: {"total": 0, "real": 0, "synthetic": 0, "sources": Counter()})

for r in rows:
    slot = r["slot_name"]
    src = r.get("source", "")
    slot_stats[slot]["total"] += 1
    slot_stats[slot]["sources"][src] += 1
    if "font" in src:
        slot_stats[slot]["synthetic"] += 1
    else:
        slot_stats[slot]["real"] += 1

print(f"{'Slot Name':16s} | {'Total':6s} | {'Real':6s} | {'Synth':6s} | {'Synth %':7s} | Main Real Sources")
print("-" * 75)
for slot, d in sorted(slot_stats.items(), key=lambda x: x[1]["total"], reverse=True):
    tot = d["total"]
    rl = d["real"]
    syn = d["synthetic"]
    pct = (syn / tot) * 100
    top_real = ", ".join([f"{k}:{v}" for k, v in d["sources"].items() if "font" not in k][:2])
    print(f"{slot:16s} | {tot:6d} | {rl:6d} | {syn:6d} | {pct:6.1f}% | {top_real}")

# Total summary
tot_all = sum(d["total"] for d in slot_stats.values())
real_all = sum(d["real"] for d in slot_stats.values())
syn_all = sum(d["synthetic"] for d in slot_stats.values())
print("-" * 75)
print(f"{'TOTAL':16s} | {tot_all:6d} | {real_all:6d} | {syn_all:6d} | {syn_all/tot_all*100:6.1f}% |")

# Look at some synthetic image paths
syn_rows = [r for r in rows if "font" in r.get("source", "")]
print(f"\nSample 5 synthetic entries:")
for r in syn_rows[:5]:
    print(f"  slot={r['slot_name']}, char={r['character']}, img_path={r['image_path']}, src_path={r.get('src_image_path', '')}")

