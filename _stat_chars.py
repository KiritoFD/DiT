import csv, sys
from collections import Counter, defaultdict

def load(csv_path):
    per_char = Counter()          # char -> total samples
    combos = defaultdict(set)     # char -> {calligrapher}
    n_rows = 0
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_char[r["character"]] += 1
            combos[r["character"]].add(r["calligrapher"])
            n_rows += 1
    return n_rows, per_char, combos

for path in ["5script/train_3top30_nobeike.csv", "5script/eval500_3top30.csv"]:
    n_rows, per_char, combos = load(path)
    print(f"=== {path} ===")
    print(f"rows={n_rows} chars={len(per_char)}")
    # bucket distribution
    buckets = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000]
    counts = [0] * len(buckets)
    for c, n in per_char.items():
        for i, b in enumerate(buckets):
            if n <= b:
                counts[i] += 1
                break
        else:
            counts[-1] += 1
    print("char samples bucket (<=N chars):")
    prev = 0
    for b, ct in zip(buckets, counts):
        if ct:
            print(f"  <= {b:4d}: {ct:5d} chars")
        prev = b
    # calligraphers per char
    n1 = sum(1 for c in per_char if len(combos[c]) == 1)
    n2 = sum(1 for c in per_char if len(combos[c]) == 2)
    n3p = sum(1 for c in per_char if len(combos[c]) >= 3)
    print(f"chars with 1 calligrapher: {n1}, 2: {n2}, >=3: {n3p}")
    print()

# threshold analysis on nobeike
n_rows, per_char, combos = load("5script/train_3top30_nobeike.csv")
for th in [5, 10, 20, 50, 100, 200, 500]:
    keep = {c for c, n in per_char.items() if n >= th}
    dropped = len(per_char) - len(keep)
    keep_rows = sum(n for c, n in per_char.items() if c in keep)
    print(f"thresh>={th:4d}: keep {len(keep):5d} chars / {keep_rows:6d} rows, drop {dropped:5d} chars ({dropped/len(per_char)*100:.1f}%), keep rows {keep_rows/n_rows*100:.1f}%")