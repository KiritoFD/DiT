# -*- coding: utf-8 -*-
"""从 s7_history.json 重建独立的 s7_klf4_top30.html dashboard。
(原文件被旧监控进程用 s8 数据覆盖且未进 git, 这里用抢救回来的完整 4003 行历史重建。)
复用 train_dashboard.html 模板 + build_dashboard 的注入逻辑。"""
import os, json

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = os.path.join(HERE, "s7_history.json")
TPL = os.path.join(HERE, "train_dashboard.html")
OUT = os.path.join(HERE, "dashboards", "s7_klf4_top30.html")

data = json.load(open(HIST, encoding="utf-8"))
t = open(TPL, encoding="utf-8").read()
data_js = json.dumps(data, ensure_ascii=False)
t = t.replace("const COLORS = {", f"const __DATA__ = {data_js};\nconst COLORS = {{", 1)
old_fetch = (
    "    const res = await fetch('train_data.json?t='+Date.now());\n"
    "    if(!res.ok) throw new Error('HTTP '+res.status);\n"
    "    const data = await res.json();"
)
t = t.replace(old_fetch, "    const data = __DATA__;", 1)
old_poster = (
    "async function loadPoster(){\n"
    "  await loadImg('latestImg', ['eval_latest.png?t='+Date.now()]);\n"
    "  await loadImg('posterImg', ['eval_poster.png?t='+Date.now()]);\n"
    "}"
)
new_poster = (
    "function loadPoster(){\n"
    "  const li=document.getElementById('latestImg'); if(li){li.src='poster_s7_klf4_top30.png';li.onerror=()=>{li.style.display='none';};}\n"
    "  const pi=document.getElementById('posterImg'); if(pi){pi.src='poster_s7_klf4_top30.png';pi.onerror=()=>{pi.style.display='none';};}\n"
    "}"
)
t = t.replace(old_poster, new_poster, 1)
t = t.replace("load();\nsyncAuto();", "load();", 1)
# 复制 s7 poster 到 dashboards 目录, html 用相对文件名引用
import shutil as _sh
_s7_poster = os.path.join(HERE, "s7_eval_poster.png")
if os.path.exists(_s7_poster):
    _sh.copy2(_s7_poster, os.path.join(os.path.dirname(OUT), "poster_s7_klf4_top30.png"))
    t = t.replace("li.src='eval_poster.png'", "li.src='poster_s7_klf4_top30.png'")
    t = t.replace("pi.src='eval_poster.png'", "pi.src='poster_s7_klf4_top30.png'")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w", encoding="utf-8").write(t)
rows = data.get("rows", [])
evs = [r for r in rows if r.get("is_eval")]
print("wrote", OUT)
print("s7 rows:", len(rows), "train:", len(rows)-len(evs), "eval:", len(evs))
print("last step:", rows[-1]["step"] if rows else None)