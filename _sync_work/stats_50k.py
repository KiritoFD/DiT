"""新数据集统计 + 与 v12 配置的兼容性核对。"""
import csv
import json
import os

os.chdir("/root/Workspace/xy/DiT")

rows = list(csv.DictReader(open("assets/train_50k.csv", encoding="utf-8")))
print(f"  行数 {len(rows)}")
for col in ("calligrapher", "script", "character"):
    print(f"  {col}: {len(set(r[col] for r in rows))} 唯一")
for col in ("calligrapher_id", "script_id", "character_id", "glyph_id"):
    v = [int(r[col]) for r in rows]
    print(f"  {col}: {min(v)}..{max(v)}  (n_unique={len(set(v))})")

# 书家 id 映射表是否覆盖
m = json.load(open("assets/callig_id_map_base.json", encoding="utf-8"))
print()
print(f"  callig_id_map_base.json: {len(m)} 条")
ids = set(int(r["calligrapher_id"]) for r in rows)
print(f"  数据集 calligrapher_id 是否都在表里: {ids <= set(int(k) for k in m)}")

# 与 v12 配置对比
d = json.load(open("src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json",
                   encoding="utf-8"))
print()
print(f"  v12 num_calligraphers = {d['num_calligraphers']}  "
      f"-> 新数据集需要 {max(ids) + 1}")
