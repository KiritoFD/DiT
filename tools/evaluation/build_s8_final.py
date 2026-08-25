# -*- coding: utf-8 -*-
"""从最后一次 s8 完整数据快照重建 s8 dashboard (s8 已停训, train_data.json 已切到 s9)。
数据来源: 远程 s8 ckpt 目录的 log.txt + cpu_eval_state.json, 一次性拉取并固化。"""
import os, sys, json, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pull_monitor as pm

S8_RUN = "20260823-234546-s8-klf4-clean-dino"
S8_SERIES = "/root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino"
S8_RUN_DIR = f"{S8_SERIES}/{S8_RUN}"
S8_CKPT = f"{S8_RUN_DIR}/checkpoints"

SEP = "===_DSH_SEP_==="
script = (
    f"echo '{SEP}LOG'; "
    f"cat {S8_RUN_DIR}/log.txt 2>/dev/null; "
    f"echo '{SEP}EVAL'; "
    f"cat {S8_CKPT}/eval_auto_*.json 2>/dev/null; "
    f"echo '{SEP}EVAL_STATE'; "
    f"cat {S8_CKPT}/cpu_eval_state.json 2>/dev/null; "
    f"echo '{SEP}END'"
)
combined = pm._ssh(script, timeout=60)
parts = combined.split(SEP)
log_content = parts[1].strip() if len(parts) >= 2 else ""
eval_json = parts[2].strip() if len(parts) >= 3 else ""
eval_state_json = parts[3].strip() if len(parts) >= 4 else ""

rows = pm.parse(log_content or "")
pm.merge_eval_json(rows, eval_json or "")
rows.sort(key=lambda r: r["step"])

# 用 s8 的 batch/dataset 写入
pm.BATCH_SIZE = 224
pm.DATASET_SIZE = 106345
pm.write(rows, f"remote:{pm.REMOTE_HOST}:{S8_RUN_DIR}/log.txt")

# 备份为 s8_train_data.json (不覆盖 train_data.json, 那是 s9 的)
import shutil
shutil.copy(pm.OUT_JSON, os.path.join(HERE, "s8_train_data.json"))
print(f"s8 data saved: {len(rows)} rows")

# 生成 s8 dashboard (用 s8 数据, 不被 pull_monitor 覆盖)
TPL = os.path.join(HERE, "train_dashboard.html")
t = open(TPL, encoding="utf-8").read()
data = json.load(open(os.path.join(HERE, "s8_train_data.json"), encoding="utf-8"))
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
    "  const li=document.getElementById('latestImg'); if(li){li.src='s8_eval_poster.png';li.onerror=()=>{li.style.display='none';};}\n"
    "  const pi=document.getElementById('posterImg'); if(pi){pi.src='s8_eval_poster.png';pi.onerror=()=>{pi.style.display='none';};}\n"
    "}"
)
t = t.replace(old_poster, new_poster, 1)
t = t.replace("load();\nsyncAuto();", "load();", 1)

# 复制 s8 poster
s8_poster = os.path.join(HERE, "eval_poster.png")  # 上次 s8 的 poster
if os.path.exists(s8_poster):
    import shutil as _sh
    _sh.copy2(s8_poster, os.path.join(HERE, "dashboards", "s8_eval_poster.png"))

OUT = os.path.join(HERE, "dashboards", "s8_klf4_clean_dino.html")
open(OUT, "w", encoding="utf-8").write(t)

evs = [r for r in rows if r.get("is_eval")]
trs = [r for r in rows if not r.get("is_eval")]
print(f"wrote {OUT}")
print(f"  rows: {len(rows)} (train={len(trs)}, eval={len(evs)})")
if trs:
    print(f"  last train step: {trs[-1]['step']}, diff: {trs[-1].get('diff')}")
if evs:
    print(f"  evals: {[(r['step'], round(r.get('mse',0),4), round(r.get('ssim',0),4)) for r in evs[-5:]]}")