"""对比新旧数据集的 id 范围，判断新数据集的 character_id 稀疏是否异常。"""
import csv
import os

os.chdir("/root/Workspace/xy/DiT")

for name, path in (("旧 fame-kxl-tj-px60", "assets/train_fame-kxl-tj-px60.csv"),
                   ("新 50k            ", "assets/train_50k.csv")):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    print(f"\n  === {name} ({len(rows):,} 行) ===")
    for col in ("calligrapher_id", "character_id", "glyph_id"):
        if col not in rows[0]:
            print(f"    {col}: 无此列")
            continue
        v = [int(r[col]) for r in rows]
        u = sorted(set(v))
        gaps = sum(1 for a, b in zip(u, u[1:]) if b - a > 1)
        print(f"    {col:<18} 范围 {min(v):>6}..{max(v):<6} 唯一 {len(u):>6}  "
              f"-> 需要表大小 {max(v) + 1:>7}  (空隙段数 {gaps})")
