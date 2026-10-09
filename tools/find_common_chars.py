import os
import csv
import json

base = "/root/Workspace/xy/DiT"

csv_dict = {
    "v10_seen": "assets/eval_seen_v10.csv",
    "v13_seen": "assets/eval_v13_seen.csv",
    "v13_strict": "assets/eval_v13_strict.csv",
    "top10_seen20": "assets/eval_top10_seen_20.csv",
    "eval200_fixed": "exp-std/csv/eval200_fixed.csv"
}

loaded_csvs = {}
for name, rel in csv_dict.items():
    p = os.path.join(base, rel)
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
            loaded_csvs[name] = rows
            print(f"[{name}] {len(rows)} 行")

# 1. 提取每个 csv 中的 (char, callig, script) 集合以及纯 char 集合
char_sets = {}
triple_sets = {}
for name, rows in loaded_csvs.items():
    chars = set()
    triples = set()
    for r in rows:
        ch = r.get("character") or r.get("char")
        cal = r.get("calligrapher")
        sc = r.get("script")
        if ch:
            chars.add(ch)
        if ch and cal and sc:
            triples.add((ch, cal, sc))
    char_sets[name] = chars
    triple_sets[name] = triples

print("\n=== 字符交集分析 (纯字 char) ===")
# 对比 eval200_fixed 与其它 csv 的字符交集
for name in ["v10_seen", "v13_seen", "v13_strict", "top10_seen20"]:
    if name in char_sets:
        common = char_sets["eval200_fixed"] & char_sets[name]
        print(f"eval200_fixed 与 {name} 的公共字 ({len(common)} 个): {sorted(list(common))}")

print("\n=== 三元组交集分析 (char, callig, script) ===")
for name in ["v10_seen", "v13_seen", "v13_strict", "top10_seen20"]:
    if name in triple_sets:
        common_t = triple_sets["eval200_fixed"] & triple_sets[name]
        print(f"eval200_fixed 与 {name} 的完全相同 (字, 书家, 书体) ({len(common_t)} 个): {sorted(list(common_t))}")

print("\n=== v13_seen 与其它阶段 ===")
for name in ["v10_seen", "v13_strict", "top10_seen20"]:
    common = char_sets["v13_seen"] & char_sets[name]
    print(f"v13_seen 与 {name} 的公共字 ({len(common)} 个): {sorted(list(common))}")
