"""Analyze (script, char, calli) combo sample-count distribution to design
targeted augmentation: rare combos get more variants, common combos get none."""
import csv, sys
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding='utf-8')

train = list(csv.DictReader(open('5script/train_3top30_nobeike.csv', encoding='utf-8')))
print(f"total train images: {len(train)}")

# combo = (script, character, calligrapher) — the generative identity
combo = Counter()
for r in train:
    combo[(r['script'], r['character'], r['calligrapher'])] += 1

n_combos = len(combo)
counts = sorted(combo.values())
print(f"unique (script,char,calli) combos: {n_combos}")
print(f"per-combo count: min={counts[0]}, median={counts[len(counts)//2]}, "
      f"mean={len(train)/n_combos:.2f}, max={counts[-1]}")

# distribution buckets
buckets = defaultdict(int)
for c in counts:
    if c == 1: buckets['1'] += 1
    elif c == 2: buckets['2'] += 1
    elif c <= 4: buckets['3-4'] += 1
    elif c <= 8: buckets['5-8'] += 1
    elif c <= 16: buckets['9-16'] += 1
    else: buckets['17+'] += 1
print("\ncombo count distribution:")
for k in ['1','2','3-4','5-8','9-16','17+']:
    print(f"  {k:>5} samples/combo: {buckets[k]:>6} combos ({100*buckets[k]/n_combos:.1f}%)")

# images locked in rare combos
for thresh in [1, 2, 4, 8]:
    n_rare_combos = sum(1 for c in counts if c <= thresh)
    n_rare_imgs = sum(c for c in counts if c <= thresh)
    print(f"\ncombos with <= {thresh} samples: {n_rare_combos} combos, {n_rare_imgs} images "
          f"({100*n_rare_imgs/len(train):.1f}% of data)")

# per-script breakdown
by_script = defaultdict(Counter)
for r in train:
    by_script[r['script']][(r['script'], r['character'], r['calligrapher'])] += 1
print("\nper-script combo stats:")
for s in ['楷', '行', '隶']:
    c = by_script[s]
    vals = sorted(c.values())
    n1 = sum(1 for v in vals if v == 1)
    print(f"  {s}: combos={len(c)}, n==1: {n1} ({100*n1/len(c):.0f}%), "
          f"max={vals[-1]}, median={vals[len(vals)//2]}")
