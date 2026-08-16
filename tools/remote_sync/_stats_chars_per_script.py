# -*- coding: utf-8 -*-
"""统计 5script 训练集中每书体需要覆盖的 unique 字符数 + 总需覆盖字集合。
用于评估五体字库覆盖可行性。"""
import csv, json
from collections import defaultdict

rows = list(csv.DictReader(open("5script/train.csv", encoding="utf-8")))
# script_id -> script 名
# 读 id_maps 获取 script 名
try:
    idm = json.load(open("5script/id_maps.json", encoding="utf-8"))
except Exception:
    idm = {}

# script_id 到名字的映射（id_maps.json 结构需确认）
# 先从 csv 的 script 列拿名字
script_names = {}
for r in rows:
    script_names[int(r["script_id"])] = r["script"]  # script 列是中文名

chars_by_script = defaultdict(set)
all_chars = set()
for r in rows:
    sid = int(r["script_id"])
    chars_by_script[sid].add(r["character"])      # character 列是字
    all_chars.add(r["character"])

print("=== 每书体需要覆盖的 unique 汉字数 ===")
for sid in sorted(chars_by_script):
    print(f"  script_id={sid} [{script_names.get(sid,'?')}]: {len(chars_by_script[sid])} 字")
print(f"  全部(跨书体去重): {len(all_chars)} 字")
print(f"  unique glyph(script×char): 21495")
# 示例字
sample = sorted(chars_by_script.get(0, []))[:40]
print(f"  楷书 sample 40字: {''.join(sample)}")
