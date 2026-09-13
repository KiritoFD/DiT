#!/bin/bash
# run_s6_resume.sh — 顺序 resume 前两份实验到真正早停/收敛
#   exp1_resume: s6_top6_diffonly 从 20000 续(epochs 无限制, 早停控制)
#   exp2_resume: s6_top6_struct_fp32 从 90000 续
set -u
cd /root/Workspace/xy/DiT
MAINLOG=run_s6_resume.log
echo "=== run_s6_resume start $(date) ===" >> "$MAINLOG"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAINLOG"; }

stop_cpu_eval(){
  pkill -f auto_eval_cpu.py 2>/dev/null
  sleep 3
  say "killed cpu eval"
}

start_cpu_eval(){
  local rd=$1
  local tag=${rd##*/}
  nohup /opt/conda/bin/python auto_eval_cpu.py --results-dir "5script/results/$rd" \
    --seen5-csv 5script/seen2_top6.csv --workers 4 --worker-threads 8 --interval 20 \
    > "cpu_eval_${tag}_resume.log" 2>&1 &
  say "cpu eval for $rd (pid $!)"
}

# ---- exp1 resume: diffonly 20000 -> 收敛 ----
say "===== RESUME EXP1 diffonly (from 20000) ====="
start_cpu_eval s6_top6_diffonly
/opt/conda/bin/python src/train.py --config resume_s6_diffonly.json
say "===== RESUME EXP1 DONE ====="
stop_cpu_eval

# ---- exp2 resume: struct_fp32 90000 -> 收敛 ----
say "===== RESUME EXP2 struct_fp32 (from 90000) ====="
start_cpu_eval s6_top6_struct_fp32
/opt/conda/bin/python src/train.py --config resume_s6_struct_fp32.json
say "===== RESUME EXP2 DONE ====="
stop_cpu_eval

say "===== RESUMES ALL COMPLETE $(date) ====="