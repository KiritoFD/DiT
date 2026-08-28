# -*- coding: utf-8 -*-
"""Verify mid-clean stats on train_3top30_common.csv (GB2312 一级+二级 filtered).

Reproduce the dry-run: count combos needing aug to reach target=6,
report removed chars/samples vs original, char/sample distribution.
"""
import os, sys, csv
from collections import Counter, defaultdict

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_COMMON = os.path.join(BASE, "5script", "train_3top30_common.csv")
TRAIN_ORIG = os.path.join(BASE, "5script", "train_3top30_nobeike.csv")
TARGET = 6


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def combo_count(rows):
    combo = Counter()
    for r in rows:
        combo[(r["script"], r["character"], r["calligrapher"])] += 1
    return combo


def char_count(rows):
    cc = Counter()
    for r in rows:
        cc[r["glyph_id"]] += 1
    return cc


orig = load(TRAIN_ORIG)
clean = load(TRAIN_COMMON)
print(f"orig: {len(orig)} rows, {len(set(r['glyph_id'] for r in orig))} chars")
print(f"clean: {len(clean)} rows, {len(set(r['glyph_id'] for r in clean))} chars")
removed_rows = len(orig) - len(clean)
removed_chars = len(set(r['glyph_id'] for r in orig)) - len(set(r['glyph_id'] for r in clean))
print(f"removed: {removed_rows} rows ({removed_rows/len(orig)*100:.1f}%), "
      f"{removed_chars} chars ({removed_chars/len(set(r['glyph_id'] for r in orig))*100:.1f}%)")

# char sample distribution on cleaned
cc = char_count(clean)
counts = sorted(cc.values(), reverse=True)
import numpy as np
arr = np.array(counts)
print(f"\n=== char sample distribution (cleaned) ===")
print(f"total chars: {len(arr)}")
for t in [1, 2, 3, 5, 10, 20, 50, 100]:
    print(f"  >= {t}: {(arr >= t).sum()} chars")

# combo distribution + aug needed
combo = combo_count(clean)
cvals = np.array(list(combo.values()))
print(f"\n=== combo (script,char,calligrapher) distribution (cleaned) ===")
print(f"total combos: {len(combo)}")
for t in [1, 2, 3, 5, 6, 10, 20, 50]:
    print(f"  = {t}: {(cvals == t).sum()}  (< {t+1}: {(cvals <= t).sum()})")

need_aug = {k: TARGET - n for k, n in combo.items() if n < TARGET}
total_aug = sum(need_aug.values())
print(f"\n=== aug-to-6 dry run ===")
print(f"combos needing aug: {len(need_aug)} ({len(need_aug)/len(combo)*100:.1f}%)")
print(f"combos already >= 6: {len(combo) - len(need_aug)}")
print(f"total aug images needed: {total_aug}")
print(f"final dataset size: {len(clean) + total_aug} rows "
      f"({(len(clean)+total_aug)/len(clean):.2f}x)")

# by script
by_script = defaultdict(lambda: [0, 0])  # [rows, aug]
for k, n in combo.items():
    s = k[0]
    by_script[s][0] += n
for k, n in need_aug.items():
    by_script[k[0]][1] += n
print(f"\n=== by script ===")
print(f"{'script':<6} {'rows':>8} {'aug':>8} {'final':>8}")
for s in sorted(by_script):
    r, a = by_script[s]
    print(f"{s:<6} {r:>8} {a:>8} {r+a:>8}")
