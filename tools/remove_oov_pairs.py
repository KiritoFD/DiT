"""剔除所有 eval csv 里 (书家,书体) pair 不在词表的行。

⚠ 不动 fs*/fs50* 文件 —— 那些是 few-shot 实验数据，
   **故意**用词表外的"新书家"来测 few-shot 能力。
"""
import csv
import glob
import json
import os

os.chdir("/root/Workspace/xy/DiT")

m = json.load(open("assets/callig_script_id_map.json", encoding="utf-8"))
pm = set(m["pair_map"].keys())
print(f"  词表: {len(pm)} 个 pair\n")

# 只处理 eval 系列
FILES = sorted(glob.glob("assets/eval_*.csv"))
total_removed = 0

for f in FILES:
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except Exception as e:
        print(f"  ✗ {os.path.basename(f)}: {e}")
        continue
    if not rows or "calligrapher_id" not in rows[0] or "script_id" not in rows[0]:
        continue

    keep, drop = [], []
    for r in rows:
        if f"{r['calligrapher_id']}:{r['script_id']}" in pm:
            keep.append(r)
        else:
            drop.append(r)

    if not drop:
        print(f"  ✓ {os.path.basename(f)}: {len(rows)} 行，无需剔除")
        continue

    # 备份
    bak = f + ".bak_pairs"
    if not os.path.exists(bak):
        os.rename(f, bak)
        src = bak
    else:
        src = f

    cols = list(rows[0].keys())
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
    print(f"  ★ {os.path.basename(f)}: {len(rows)} -> {len(keep)} 行 "
          f"(剔 {len(drop)})")
    total_removed += len(drop)

print(f"\n  合计剔除: {total_removed} 行")
print(f"  备份后缀: .bak_pairs")

# 复验
print("\n  === 复验 ===")
for f in FILES:
    try:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
    except Exception:
        continue
    if not rows or "calligrapher_id" not in rows[0]:
        continue
    bad = [r for r in rows
           if f"{r['calligrapher_id']}:{r['script_id']}" not in pm]
    if bad:
        print(f"  ✗ {os.path.basename(f)}: 仍有 {len(bad)} 条词表外")
    else:
        print(f"  ✓ {os.path.basename(f)}: {len(rows)} 行，全部在词表内")
