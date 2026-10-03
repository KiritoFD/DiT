import csv, os, sys
from collections import Counter

csv_path = "assets/train_top10_style23.csv"
if not os.path.exists(csv_path):
    print("CSV not found:", csv_path)
    sys.exit(1)

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"Total rows in {csv_path}: {len(rows)}")

sources = Counter(r.get("source", "") for r in rows)
print("\nSources breakdown:")
for s, c in sources.most_common():
    print(f"  {s:30s}: {c:6d} ({c/len(rows)*100:5.1f}%)")

augs = Counter(r.get("aug", "") for r in rows)
print("\nAugmentations breakdown:")
for a, c in augs.most_common():
    print(f"  '{a}': {c:6d}")

# Check slot distributions
slots = Counter(r.get("slot_name", "") for r in rows)
print(f"\nSlots breakdown ({len(slots)} slots):")
for sl, c in slots.most_common():
    print(f"  {sl:15s}: {c:6d}")

# Check how many characters
chars = Counter(r.get("character", "") for r in rows)
print(f"\nUnique characters: {len(chars)}")

# Check eval_top10_strict_subset84.csv
eval_csv = "assets/eval_top10_strict_subset84.csv"
if os.path.exists(eval_csv):
    with open(eval_csv, "r", encoding="utf-8") as f:
        eval_rows = list(csv.DictReader(f))
    print(f"\nEval strict subset rows: {len(eval_rows)}")
    eval_sources = Counter(r.get("source", "") for r in eval_rows)
    print("Eval sources:", eval_sources)
    eval_slots = Counter(r.get("slot_name", "") for r in eval_rows)
    print("Eval slots count:", len(eval_slots))
    eval_chars = set(r.get("character", "") for r in eval_rows)
    train_chars = set(r.get("character", "") for r in rows)
    overlap = eval_chars.intersection(train_chars)
    print(f"Eval unique chars: {len(eval_chars)}, overlap with train chars: {len(overlap)}")
