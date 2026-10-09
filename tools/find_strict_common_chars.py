import os
import csv

base = "/root/Workspace/xy/DiT"
v10_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_fame3_strict_clean_v9.csv"), encoding="utf-8")))[:50]
v13_rows = list(csv.DictReader(open(os.path.join(base, "assets/eval_v13_strict_fixed.csv"), encoding="utf-8")))
e200_rows = list(csv.DictReader(open(os.path.join(base, "exp-std/csv/eval200_fixed.csv"), encoding="utf-8")))

print("=== 检查 v10b strict50 (0..49) 与 v13 / eval200 的重合字 ===")
v10_dict = {}
for i, r in enumerate(v10_rows):
    ch = r.get("character") or r.get("char")
    v10_dict[ch] = (i, r.get("calligrapher"), r.get("script"))

v13_dict = {}
for i, r in enumerate(v13_rows):
    ch = r.get("character") or r.get("char")
    if ch not in v13_dict:
        v13_dict[ch] = []
    v13_dict[ch].append((i, r.get("calligrapher"), r.get("script")))

e200_dict = {}
for i, r in enumerate(e200_rows):
    ch = r.get("character") or r.get("char")
    if ch not in e200_dict:
        e200_dict[ch] = []
    e200_dict[ch].append((i, r.get("calligrapher"), r.get("script")))

# 1. 查找 v10 ∩ v13 ∩ e200
three_way = set(v10_dict.keys()) & set(v13_dict.keys()) & set(e200_dict.keys())
print(f"1. 三方全覆盖公共字 (v10 ∩ v13 ∩ e200) ({len(three_way)} 个): {sorted(list(three_way))}")
for ch in sorted(list(three_way)):
    print(f"   【{ch}】:")
    print(f"      v10b: #{v10_dict[ch][0]:02d} {v10_dict[ch][1]}-{v10_dict[ch][2]}")
    for item in v13_dict[ch]:
        print(f"      v13/v21/v23: #{item[0]:03d} {item[1]}-{item[2]}")
    for item in e200_dict[ch]:
        print(f"      v66/v68/v70: #{item[0]:03d} {item[1]}-{item[2]}")

# 2. 查找 v13 ∩ e200 (覆盖从 v13 到 v70 的 6 个阶段)
common_13_200 = set(v13_dict.keys()) & set(e200_dict.keys())
print(f"\n2. 六阶段全覆盖公共字 (v13 ∩ e200: v13, v21, v23, v66, v68, v70) ({len(common_13_200)} 个): {sorted(list(common_13_200))}")
for ch in sorted(list(common_13_200)):
    match_str = ""
    # 是否有同书家同书体
    exacts = []
    for item13 in v13_dict[ch]:
        for item200 in e200_dict[ch]:
            if item13[1] == item200[1] and item13[2] == item200[2]:
                exacts.append((item13, item200))
    if exacts:
        print(f"   ★ 完美同名同书体【{ch}】: v13/v21/v23 #{exacts[0][0][0]} vs e200 #{exacts[0][1][0]} ({exacts[0][0][1]} · {exacts[0][0][2]})")
    else:
        print(f"   • 公共字【{ch}】: v13/v21/v23 #{v13_dict[ch][0][0]} ({v13_dict[ch][0][1]}-{v13_dict[ch][0][2]}) vs e200 #{e200_dict[ch][0][0]} ({e200_dict[ch][0][1]}-{e200_dict[ch][0][2]})")
