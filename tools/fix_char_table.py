"""生成修复后的字表（把 752 条错配样本的 character 改成 GT 实际字形）。

同时把 std_path 指向正确字形的 std 图 —— 这样 g 就和 GT 同简繁了。
输出: assets/train_50k_v2_fixed.csv + 差异报告
"""
import csv
import os
import re
from collections import Counter, defaultdict

os.chdir("/root/Workspace/xy/DiT")

conf = list(csv.DictReader(open("assets/mismatch_confirmed.csv",
                                encoding="utf-8")))
print(f"  确认错配: {len(conf)}")

rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
print(f"  原 csv: {len(rows)} 行")

# 建 char -> std_path（正确字形的 std）
std_of = {}
for r in rows:
    std_of.setdefault(r["character"], r["std_path"])
print(f"  字表: {len(std_of)}")

# image_path -> 修正后的 character
fix = {}
miss_std = 0
for c in conf:
    ip = c["image_path"]
    new_ch = c["char_true"]
    fix[ip] = new_ch
    if new_ch not in std_of:
        miss_std += 1
print(f"  需要修的样本: {len(fix)}")
print(f"  修正字不在字表(无 std 可换): {miss_std}")

# 写修复后的 csv
out = "assets/train_50k_v2_fixed.csv"
cols = list(rows[0].keys())
n_fixed = 0
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    for r in rows:
        ip = r["image_path"]
        if ip in fix:
            new_ch = fix[ip]
            if new_ch in std_of:
                r["character"] = new_ch
                r["std_path"] = std_of[new_ch]
                n_fixed += 1
        w.writerow(r)
print(f"\n  ✓ written {out}")
print(f"    修正了 {n_fixed} 行（character + std_path 都换成繁体版）")

# 差异报告
print(f"\n  === 修正明细 Top20 ===")
for (a, b), n in Counter((c["char_csv"], c["char_true"])
                         for c in conf).most_common(20):
    print(f"    {a} -> {b}: {n}")

# 校验
new_rows = list(csv.DictReader(open(out, encoding="utf-8")))
ch_changed = sum(1 for a, b in zip(rows, new_rows)
                 if a["character"] != b["character"])
std_changed = sum(1 for a, b in zip(rows, new_rows)
                  if a["std_path"] != b["std_path"])
print(f"\n  === 校验 ===")
print(f"    character 变化: {ch_changed}")
print(f"    std_path 变化:  {std_changed}")
print(f"    行数一致: {len(rows)} == {len(new_rows)}")
