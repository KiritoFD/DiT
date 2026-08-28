import csv
from collections import Counter

def gb2312_level1():
    chars = []
    for q in range(0xB0, 0xD8):      # 16-55 区
        for p in range(0xA1, 0xFF):
            try:
                chars.append(bytes([q, p]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return set(chars)

def gb2312_level2():
    chars = []
    for q in range(0xD8, 0xF8):      # 56-87 区
        for p in range(0xA1, 0xFF):
            try:
                chars.append(bytes([q, p]).decode("gb2312"))
            except UnicodeDecodeError:
                pass
    return set(chars)

L1 = gb2312_level1()
L2 = gb2312_level2()
print(f"GB2312 一级: {len(L1)} 字, 二级: {len(L2)} 字")

for path in ["5script/train_3top30_nobeike.csv", "5script/eval500_3top30.csv"]:
    n_rows = 0
    per_char = Counter()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            per_char[r["character"]] += 1
            n_rows += 1
    inL1 = {c for c in per_char if c in L1}
    inL2 = {c for c in per_char if c in L2 and c not in L1}
    outside = {c for c in per_char if c not in L1 and c not in L2}
    rows_inL1 = sum(per_char[c] for c in inL1)
    rows_inL2 = sum(per_char[c] for c in inL2)
    rows_out = sum(per_char[c] for c in outside)
    print(f"\n=== {path} ===")
    print(f"总字符 {len(per_char)} 个, 总样本 {n_rows} 行")
    print(f"  一级常用字: {len(inL1):5d} 字符 / {rows_inL1:6d} 样本")
    print(f"  二级次常用: {len(inL2):5d} 字符 / {rows_inL2:6d} 样本")
    print(f"  国标外(繁/异/生僻): {len(outside):5d} 字符 / {rows_out:6d} 样本")
    # 展示几个国标外的样本字和计数
    top_out = sorted(outside, key=lambda c: -per_char[c])  # set of chars
    print(f"  国标外 top10 字: " + ", ".join(f"{c}({per_char[c]})" for c in top_out[:10]))
    # 一级字中也有极少样本的? 不在本统计范围内
    l1_low = {c: n for c, n in per_char.items() if c in L1 and n < 6}
    print(f"  一级常用字中每字<6样本的字数: {len(l1_low)}")