import os
import csv

base = "/root/Workspace/xy/DiT"
v10_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_fame3_strict_clean_v9.csv"), encoding="utf-8")))[:50]
v13_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_v13_strict_fixed.csv"), encoding="utf-8")))

v10_chars = {r.get("character") or r.get("char"): (i, r.get("calligrapher"), r.get("script")) for i, r in enumerate(v10_rows)}
v13_chars = {}
for i, r in enumerate(v13_rows):
    ch = r.get("character") or r.get("char")
    if ch not in v13_chars:
        v13_chars[ch] = []
    v13_chars[ch].append((i, r.get("calligrapher"), r.get("script")))

print("=== v10b strict50 与 v13 strict 的公共字 ===")
common = set(v10_chars.keys()) & set(v13_chars.keys())
print(f"共有 {len(common)} 个公共字: {sorted(list(common))}")
for ch in sorted(list(common)):
    print(f"  【{ch}】: v10b #{v10_chars[ch][0]:02d} ({v10_chars[ch][1]}-{v10_chars[ch][2]}) vs v13 #{v13_chars[ch][0][0]:03d} ({v13_chars[ch][0][1]}-{v13_chars[ch][0][2]})")
