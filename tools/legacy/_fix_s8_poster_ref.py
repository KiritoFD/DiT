# -*- coding: utf-8 -*-
"""重建 s8 dashboard, poster 引用改为 poster_s8_klf4_clean_dino.png。"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "train_dashboard.html")
OUT = os.path.join(HERE, "dashboards", "s8_klf4_clean_dino.html")

data = json.load(open(os.path.join(HERE, "s8_train_data.json"), encoding="utf-8"))
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
    "  const li=document.getElementById('latestImg'); if(li){li.src='poster_s8_klf4_clean_dino.png';li.onerror=()=>{li.style.display='none';};}\n"
    "  const pi=document.getElementById('posterImg'); if(pi){pi.src='poster_s8_klf4_clean_dino.png';pi.onerror=()=>{pi.style.display='none';};}\n"
    "}"
)
t = t.replace(old_poster, new_poster, 1)
t = t.replace("load();\nsyncAuto();", "load();", 1)
open(OUT, "w", encoding="utf-8").write(t)
print("wrote", OUT)
