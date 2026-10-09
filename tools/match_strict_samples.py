import os
import csv
import json

base = "/root/Workspace/xy/DiT"

# 1. 查找 v10b strict50 对应的 csv
v10_csv_candidates = [
    "assets/eval_fame3_strict_clean_v9.csv",
    "assets/eval_strict50.csv",
    "assets/eval_fame3_strict.csv"
]
v10_strict_rows = None
for cp in v10_csv_candidates:
    p = os.path.join(base, cp)
    if os.path.exists(p):
        v10_strict_rows = list(csv.DictReader(open(p, encoding="utf-8")))
        print(f"v10b strict csv: {cp} ({len(v10_strict_rows)} rows)")
        break

# 2. v13 / v21 / v23 strict 对应的 csv (249 / 250 rows)
v13_csv_candidates = [
    "assets/eval_v13_strict_fixed.csv",
    "assets/eval_v13_strict.csv"
]
v13_strict_rows = None
for cp in v13_csv_candidates:
    p = os.path.join(base, cp)
    if os.path.exists(p):
        v13_strict_rows = list(csv.DictReader(open(p, encoding="utf-8")))
        print(f"v13/v21/v23 strict csv: {cp} ({len(v13_strict_rows)} rows)")
        break

# 3. v66 / v68 / v70 eval200fix 对应的 csv (187 rows)
eval200_p = os.path.join(base, "exp-std/csv/eval200_fixed.csv")
eval200_rows = list(csv.DictReader(open(eval200_p, encoding="utf-8")))
print(f"v66/v68/v70 eval200fix csv: exp-std/csv/eval200_fixed.csv ({len(eval200_rows)} rows)")

# 4. 构建三者的索引: (char, callig, script) -> index
def build_meta_map(rows):
    res = {}
    for i, r in enumerate(rows):
        ch = r.get("character") or r.get("char")
        cal = r.get("calligrapher")
        sc = r.get("script")
        res[i] = {
            "idx": i,
            "char": ch,
            "callig": cal,
            "script": sc,
            "key": f"{ch}_{cal}_{sc}",
            "char_only": ch
        }
    return res

m_v10 = build_meta_map(v10_strict_rows) if v10_strict_rows else {}
m_v13 = build_meta_map(v13_strict_rows) if v13_strict_rows else {}
m_e200 = build_meta_map(eval200_rows)

print("\n--- 查找三者之间的公共字 (严格 character 匹配) ---")
# 找出三者的公共字符
chars_v10 = set(v["char"] for v in m_v10.values())
chars_v13 = set(v["char"] for v in m_v13.values())
chars_e200 = set(v["char"] for v in m_e200.values())

common_13_200 = chars_v13 & chars_e200
print(f"v13/v21/v23 与 v66/v68/v70 的公共字 ({len(common_13_200)} 个): {sorted(list(common_13_200))}")

common_all3 = chars_v10 & chars_v13 & chars_e200 if chars_v10 else set()
print(f"所有阶段全重合的公共字 ({len(common_all3)} 个): {sorted(list(common_all3))}")
if not common_all3 and chars_v10:
    print(f"v10 与 v13 的公共字: {chars_v10 & chars_v13}")
    print(f"v10 与 e200 的公共字: {chars_v10 & chars_e200}")
