# -*- coding: utf-8 -*-
"""评估"空间字形条件"方案的数据可行性。

方案：用每个 glyph (script×char) 的 medoid final_skeleton 作为空间字形条件。
检查：
  1. unique glyph 数
  2. 每 glyph 有多少张图（够不够取 medoid）
  3. 用 final_skeleton 作为 glyph spatial 参考是否可行
"""
import csv, os, glob
from collections import defaultdict

rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
print(f"train rows: {len(rows)}")

glyph_rows = defaultdict(list)
for r in rows:
    glyph_rows[r["glyph_id"]].append(r["image_path"])
print(f"unique glyph_id: {len(glyph_rows)}")

# 分布
counts = sorted(len(v) for v in glyph_rows.values())
print(f"glyph 图数分布: min={counts[0]} median={counts[len(counts)//2]} max={counts[-1]}")
multi = sum(1 for v in glyph_rows.values() if len(v) >= 2)
single = sum(1 for v in glyph_rows.values() if len(v) == 1)
print(f">=2 图的 glyph: {multi} ({multi/len(glyph_rows)*100:.1f}%) | 仅1图 glyph: {single} ({single/len(glyph_rows)*100:.1f}%)")

# 检查 final_skeleton 覆盖（能不能从训练图取骨架）
skel_ids = set(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("final_skeleton/*.png"))
img_id_of_sample = os.path.basename(rows[0]["image_path"])[:-4]
print(f"final_skeleton 总图数: {len(skel_ids)}")
print(f"示例样本 {img_id_of_sample} 有骨架: {img_id_of_sample in skel_ids}")

# 结论：能否为每个 glyph 取到一张 medoid 骨架
can_medoid = sum(1 for v in glyph_rows.values() if len(v) >= 1)
print(f"每个glyph至少1图(可取骨架): {can_medoid}/{len(glyph_rows)}")
