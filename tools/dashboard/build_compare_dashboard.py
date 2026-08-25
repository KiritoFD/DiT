# -*- coding: utf-8 -*-
"""把 s7 历史数据与 s8 实时数据合并，生成对比 dashboard。

输出: G:\\GitHub\\DiT\\tools\\dashboards\\s7_vs_s8.html
- s7 数据来自旧的 s7_klf4_top30 dashboard 快照
- s8 数据来自当前 train_data.json (实时更新)
"""
import os, json, re

HERE = os.path.dirname(os.path.abspath(__file__))
S7_SNAP = os.path.join(HERE, "dashboards", "s7_klf4_top30.html")
S8_JSON = os.path.join(HERE, "train_data.json")
S7_HIST = os.path.join(HERE, "s7_history.json")
OUT = os.path.join(HERE, "dashboards", "s7_vs_s8.html")
TPL = os.path.join(HERE, "compare_dashboard.html")  # 新模板

# --- s7 历史数据 ---
if os.path.exists(S7_HIST):
    s7 = json.load(open(S7_HIST, encoding="utf-8"))
else:
    # 从 s7 dashboard 快照提取
    txt = open(S7_SNAP, encoding="utf-8").read()
    m = re.search(r"const __DATA__\s*=\s*(\{.*?\});\s*const COLORS", txt, re.S)
    if not m:
        raise SystemExit("can't find __DATA__ in s7 snapshot")
    s7 = json.loads(m.group(1))
    # 保存为独立文件供后续使用
    json.dump(s7, open(S7_HIST, "w", encoding="utf-8"), ensure_ascii=False)
    print("saved s7_history.json (%d rows)" % len(s7.get("rows", [])))

s8 = json.load(open(S8_JSON, encoding="utf-8"))

# 标注来源
for r in s7.get("rows", []):
    r["exp"] = "s7"
for r in s8.get("rows", []):
    r["exp"] = "s8"

# 用 compare_dashboard.html 模板
if not os.path.exists(TPL):
    raise SystemExit("template compare_dashboard.html not found")

t = open(TPL, encoding="utf-8").read()
merged_js = json.dumps({"s7": s7, "s8": s8}, ensure_ascii=False)
t = t.replace("/*__MERGED_DATA__*/{}", merged_js, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(t)
print("wrote", OUT)
print("s7 rows: %d, last step %s" % (len(s7.get("rows",[])), s7.get("rows",[{}])[-1].get("step") if s7.get("rows") else "?"))
print("s8 rows: %d, last step %s" % (len(s8.get("rows",[])), s8.get("rows",[{}])[-1].get("step") if s8.get("rows") else "?"))