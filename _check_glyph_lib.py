# -*- coding: utf-8 -*-
"""核对标准字形 latent 字典（v1/v2）对 fame 数据的实际覆盖，暴露接线 bug。

检查项：
  1. v1 目录 std_glyph_latent 是否存在（dataset 当前接线的是 v1）
  2. v2 目录 std_glyph_latent_v2 覆盖哪些书体 / 字数
  3. fame 训练集中，v1 与 v2 各自能命中多少（按书体拆分）
"""
import os, sys, csv, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import glyph_latent as v1
from src.utils import glyph_latent_v2 as v2

print("=" * 60)
print("1) 库目录是否存在")
print(f"  v1 LIB_DIR = {v1.LIB_DIR}")
print(f"    exists   = {os.path.isdir(v1.LIB_DIR)}   <-- dataset 接线的是这个")
print(f"  v2 LIB_DIR = {v2.DEFAULT_LIB_DIR}")
print(f"    exists   = {os.path.isdir(v2.DEFAULT_LIB_DIR)}")

print("\n2) v2 各字体字数")
if os.path.isdir(v2.DEFAULT_LIB_DIR):
    for font in sorted(os.listdir(v2.DEFAULT_LIB_DIR)):
        d = os.path.join(v2.DEFAULT_LIB_DIR, font)
        if os.path.isdir(d):
            n = len([f for f in os.listdir(d) if f.endswith(".npy")])
            print(f"    {font:<10} {n}")

print("\n3) v1 / v2 对 fame 训练集的命中率")
lk1 = v1.get_glyph_lookup()
lk2 = v2.get_glyph_lookup_v2(preload=True)
print(f"  v1 SCRIPT_TO_BOOK = {v1.SCRIPT_TO_BOOK}")
print(f"  v2 SCRIPT_FONTS   = {v2.SCRIPT_FONTS}")

rows = []
with open("5script/train_fame.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        rows.append(r)
print(f"  fame train rows = {len(rows)}")

stat = {}
for r in rows:
    sc = r["script"]
    sid = int(r["script_id"])
    ch = r["character"]
    h1 = lk1.get(sid, ch) is not None
    h2 = lk2.get(sid, ch) is not None
    d = stat.setdefault(sc, {"n": 0, "v1": 0, "v2": 0})
    d["n"] += 1
    d["v1"] += int(h1)
    d["v2"] += int(h2)

print(f"\n  {'书体':<6}{'样本':>8}{'v1命中':>10}{'v1%':>8}{'v2命中':>10}{'v2%':>8}")
tot = {"n": 0, "v1": 0, "v2": 0}
for sc, d in sorted(stat.items(), key=lambda x: -x[1]["n"]):
    tot["n"] += d["n"]
    tot["v1"] += d["v1"]
    tot["v2"] += d["v2"]
    print(f"  {sc:<6}{d['n']:>8}{d['v1']:>10}{d['v1']/d['n']*100:>8.1f}"
          f"{d['v2']:>10}{d['v2']/d['n']*100:>8.1f}")
print(f"  {'合计':<6}{tot['n']:>8}{tot['v1']:>10}{tot['v1']/tot['n']*100:>8.1f}"
      f"{tot['v2']:>10}{tot['v2']/tot['n']*100:>8.1f}")

print("\n4) 结论")
if tot["v1"] == 0:
    print("  !! v1 命中 0 —— dataset 接线的字典完全失效；")
    print("     latent_dataset.py:286 缺失时返回零张量，因此 w_glyph_cond 启用后")
    print("     条件全为 0，实验静默无效（loss 正常下降但条件从未生效）。")
print(f"  v2 覆盖 {tot['v2']/tot['n']*100:.1f}%（草/篆/六体无标准字体，属已知缺口）")
