#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统计 3top30 训练集样本分布：
- 按 (script, character, calligrapher) 组合
- 按 character（字）
- 按 calligrapher（书法家）
- 按 script
输出稀疏尾部详情。
"""
import csv, sys, os
from collections import Counter, defaultdict
sys.stdout.reconfigure(encoding="utf-8")

csv_path = sys.argv[1] if len(sys.argv) > 1 else "5script/train_3top30.csv"

rows = []
with open(csv_path, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"Total rows: {len(rows)}")

combo = Counter()
per_char = defaultdict(Counter)      # character -> {calli: count}
per_calli = defaultdict(Counter)     # calligrapher -> {char: count}
per_char_tot = Counter()
per_calli_tot = Counter()
per_script = Counter()
has_glyph = 0
for r in rows:
    sc, ch, ca = r["script"], r["character"], r["calligrapher"]
    key = (sc, ch, ca)
    combo[key] += 1
    per_char[ch][ca] += 1
    per_calli[ca][ch] += 1
    per_char_tot[ch] += 1
    per_calli_tot[ca] += 1
    per_script[sc] += 1
    if r.get("glyph_id") and r["glyph_id"] != "":
        has_glyph += 1

print(f"  scripts: {dict(per_script)}")
print(f"  glyph_id present: {has_glyph}/{len(rows)}")

# combo distribution
vals = sorted(combo.values())
print(f"\nCombo (script,char,calli) count: {len(combo)}")
print(f"  min={vals[0]} max={vals[-1]} mean={sum(vals)/len(vals):.2f}")
for n in sorted(set(vals)):
    print(f"  combos with exactly {n} samples: {sum(1 for v in vals if v == n)}")

# character level
char_vals = sorted(per_char_tot.values())
print(f"\nCharacter count: {len(per_char_tot)}")
print(f"  min={char_vals[0]} (chars: {[c for c,v in per_char_tot.items() if v==char_vals[0]][:20]})")
print(f"  max={char_vals[-1]} mean={sum(char_vals)/len(char_vals):.1f}")
# how many characters have < 10, < 20, < 30 samples
for th in (5, 10, 20, 30, 50, 100):
    print(f"  chars with < {th} samples: {sum(1 for v in char_vals if v < th)}")

# per character per calli
sparse_combo_per_char = 0
for ch, cc in per_char.items():
    for ca, cnt in cc.items():
        if cnt <= 4:
            sparse_combo_per_char += 1
print(f"\n(char, calli) pairs with <=4 samples: {sparse_combo_per_char}")

# calligrapher level
calli_vals = sorted(per_calli_tot.values())
print(f"\nCalligrapher count: {len(per_calli_tot)}")
print(f"  min={calli_vals[0]} max={calli_vals[-1]} mean={sum(calli_vals)/len(calli_vals):.1f}")
for th in (5, 10, 20, 50, 100, 500):
    print(f"  callis with < {th} samples: {sum(1 for v in calli_vals if v < th)}")

# Bottom 10 chars by count
print("\nBottom 10 chars by total sample count:")
for ch, v in sorted(per_char_tot.items(), key=lambda x: x[1])[:10]:
    print(f"  {ch}: {v} samples, callis={len(per_char[ch])}")

# files existence check
miss = 0
from PIL import Image
import os
img_root = sys.argv[2] if len(sys.argv) > 2 else "final_imgs_256"
checked = 0
for r in rows[:2000]:
    iid = os.path.basename(r["image_path"])[:-4]
    p = os.path.join(img_root, f"{iid}.png")
    if not os.path.exists(p):
        miss += 1
    checked += 1
print(f"\nImage file check on first {checked}: {miss} missing")