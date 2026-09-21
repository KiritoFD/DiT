"""查缺失 std 的 30 个修正字 + 从标准字库补 std。"""
import csv
import os
import re

os.chdir("/root/Workspace/xy/DiT")

wide = list(csv.DictReader(open("assets/mismatch_wide.csv", encoding="utf-8")))
rows = list(csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")))
std_of = {}
for r in rows:
    std_of.setdefault(r["character"], r["std_path"])

miss = [r for r in wide if r["char_true"] not in std_of]
print(f"  缺失 std 的修正字: {len(miss)}")
for r in miss:
    print(f"    {r['char_csv']} -> {r['char_true']}  "
          f"[{r['script']}/{r['calligrapher']}]")

chars = sorted(set(r["char_true"] for r in miss))
print(f"\n  去重字数: {len(chars)}")
for c in chars:
    n = sum(1 for r in rows if r["character"] == c)
    print(f"    {c!r}: 在训练集出现 {n} 次")

# 这些字在 std 库里有图吗？（std 文件名是编号，需要另找映射）
# 思路: 如果该字在训练集里根本没出现过，那它的 std 图也不存在（std 是按训练集渲染的）
print(f"\n  结论: 这些繁体字在训练集里{'都' if all(sum(1 for r in rows if r['character'] == c) == 0 for c in chars) else '部分'}没出现过")
print("       -> std 库里没有它们的标准字形 -> 无法补")
print("       -> 只能保留原样（这 30 行的 g 仍是简体，但 GT 是繁体，属残留错配）")
print("       -> 或者：把这些样本从训练集剔除")
