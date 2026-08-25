# -*- coding: utf-8 -*-
"""一次性: 拉取远程 s7 完整 log + eval_auto + cpu_eval_state → 生成 s7_history.json。
复用 pull_monitor.py 的 parse / merge_eval_json 逻辑。
"""
import os, sys, json, subprocess
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pull_monitor as pm

S7_RUN = "20260823-163710-s7-klf4-top30-diffonly"
S7_SERIES = "/root/Workspace/xy/DiT/5script/results/s7_klf4_top30"
S7_RUN_DIR = f"{S7_SERIES}/{S7_RUN}"
S7_CKPT = f"{S7_RUN_DIR}/checkpoints"

SEP = "===_DSH_SEP_==="
script = (
    f"echo '{SEP}LOG'; "
    f"cat {S7_RUN_DIR}/log.txt 2>/dev/null; "
    f"echo '{SEP}EVAL'; "
    f"cat {S7_CKPT}/eval_auto_*.json 2>/dev/null; "
    f"echo '{SEP}EVAL_STATE'; "
    f"cat {S7_CKPT}/cpu_eval_state.json 2>/dev/null; "
    f"echo '{SEP}EVAL_LOG'; "
    f"grep -E 'MSE=.*SSIM=' /root/Workspace/xy/DiT/auto_eval_cpu.log 2>/dev/null | tail -200; "
    f"echo '{SEP}END'"
)
combined = pm._ssh(script, timeout=60)
parts = combined.split(SEP)
log_content = parts[1].strip() if len(parts) >= 2 else ""
eval_json = parts[2].strip() if len(parts) >= 3 else ""
eval_state_json = parts[3].strip() if len(parts) >= 4 else ""
eval_log_lines = parts[4].strip() if len(parts) >= 5 else ""

print(f"log lines: {len(log_content.splitlines())}")
print(f"eval_json len: {len(eval_json)}")
print(f"eval_state len: {len(eval_state_json)}")
print(f"eval_log lines: {len(eval_log_lines.splitlines())}")

rows = pm.parse(log_content or "")
pm.merge_eval_json(rows, eval_json or "")
eval_log_ts = pm.parse_eval_log_ts(eval_log_lines or "")
existing_steps = {r["step"] for r in rows}
for step, (mse, ssim, ts) in eval_log_ts.items():
    if step not in existing_steps:
        rows.append({
            "step": step, "total": None, "diff": None,
            "stdmid": None, "x0lat": None, "stepsPerSec": None,
            "memCur": None, "memPeak": None,
            "mse": mse, "ssim": ssim, "ts": ts, "is_eval": True,
        })
        existing_steps.add(step)
    else:
        for r in rows:
            if r["step"] == step:
                r["is_eval"] = True
                r["mse"] = mse
                r["ssim"] = ssim
                if ts:
                    r["ts"] = ts
                break
rows.sort(key=lambda r: r["step"])

# 用 write 生成标准格式
pm.write(rows, f"remote:{pm.REMOTE_HOST}:{S7_RUN_DIR}/log.txt")

# train_data.json 现在是 s7 的；备份为 s7_history.json
import shutil
shutil.copy(pm.OUT_JSON, os.path.join(HERE, "s7_history.json"))

# 重新加载确认
d = json.load(open(os.path.join(HERE, "s7_history.json"), encoding="utf-8"))
evs = [r for r in d["rows"] if r.get("is_eval")]
trs = [r for r in d["rows"] if not r.get("is_eval")]
print(f"\ns7 history: {len(d['rows'])} rows (train={len(trs)}, eval={len(evs)})")
if trs:
    print(f"  last train step={trs[-1]['step']} diff={trs[-1].get('diff')}")
if evs:
    print(f"  eval steps: {[(r['step'], round(r['mse'],4) if r.get('mse') else None, round(r['ssim'],4) if r.get('ssim') else None) for r in evs]}")