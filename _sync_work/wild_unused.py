#!/usr/bin/env python
"""wild_extract 里每个 (书家-书体) 文件夹: 总数 vs 被 50k 训练用掉的数量 -> 找没用过的数据"""
import csv, os, re
from collections import Counter

os.chdir("/root/Workspace/xy/DiT")
WILD = "/root/Workspace/xy/HCSU/wild_extract"

used = Counter()
for r in csv.DictReader(open("assets/train_50k_v2.csv", encoding="utf-8")):
    m = re.match(r".*wild/([^/]+)/[^/]+$", r.get("src_image_path", "") or "")
    if m:
        used[m.group(1)] += 1

print(f"{'folder':<14}{'total':>6}{'used50k':>9}{'unused':>8}")
rows = []
for name in sorted(os.listdir(WILD)):
    p = os.path.join(WILD, name)
    if not os.path.isdir(p):
        continue
    n = len([f for f in os.listdir(p) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    u = used.get(name, 0)
    rows.append((name, n, u, n - u))
for name, n, u, un in sorted(rows, key=lambda x: -(x[3])):
    print(f"{name:<14}{n:>6}{u:>9}{un:>8}")
tot_unused = sum(r[3] for r in rows)
print(f"\nfolders={len(rows)}  total_unused={tot_unused}")
full_unused = [r[0] for r in rows if r[2] == 0 and r[1] >= 120]
print("完全没用过且 >=120 张:", full_unused)