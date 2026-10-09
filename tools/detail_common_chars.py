import os
import csv
import json

base = "/root/Workspace/xy/DiT"

v10_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_fame3_strict_clean_v9.csv"), encoding="utf-8")))
v13_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_v13_strict_fixed.csv"), encoding="utf-8")))
e200_rows = list(csv.DictReader(open(os.path.join(base, "exp-std/csv/eval200_fixed.csv"), encoding="utf-8")))

# 检查 17 个公共字
common_chars = ['仲', '兩', '其', '典', '冠', '出', '化', '北', '呼', '夜', '尚', '巡', '悟', '照', '豈', '起', '鼓']

print("=== 17 个公共字符在各严格评测集中的具体行定义与索引 ===")
matched_entries = []

for ch in common_chars:
    # 查找在 e200 中的条目
    e200_matches = [(i, r) for i, r in enumerate(e200_rows) if (r.get("character") or r.get("char")) == ch]
    # 查找在 v13 中的条目
    v13_matches = [(i, r) for i, r in enumerate(v13_rows) if (r.get("character") or r.get("char")) == ch]
    # 查找在 v10 中的条目
    v10_matches = [(i, r) for i, r in enumerate(v10_rows) if (r.get("character") or r.get("char")) == ch]

    for idx_200, r_200 in e200_matches:
        for idx_13, r_13 in v13_matches:
            entry = {
                "char": ch,
                "e200_idx": idx_200,
                "e200_callig": r_200.get("calligrapher"),
                "e200_script": r_200.get("script"),
                "v13_idx": idx_13,
                "v13_callig": r_13.get("calligrapher"),
                "v13_script": r_13.get("script"),
                "v10_info": [(i, r.get("calligrapher"), r.get("script")) for i, r in v10_matches]
            }
            matched_entries.append(entry)

print(f"共找到 {len(matched_entries)} 个匹配对:")
for item in matched_entries[:20]:
    v10_str = f"v10_idx={item['v10_info'][0][0]} ({item['v10_info'][0][1]})" if item['v10_info'] else "v10:无"
    print(f"字:【{item['char']}】 | e200(v66/v68/v70): #{item['e200_idx']:03d} {item['e200_callig']}-{item['e200_script']} | v13/v21/v23: #{item['v13_idx']:03d} {item['v13_callig']}-{item['v13_script']} | {v10_str}")
