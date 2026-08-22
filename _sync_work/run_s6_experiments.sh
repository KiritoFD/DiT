#!/bin/bash
# run_s6_experiments.sh — 顺序跑 4 份 S 模型 top6 实验（含 CPU eval 管理）
#   1) exp_s6_top6_diffonly          : 纯 diff loss, batch 192
#   2) exp_s6_top6_struct_fp32       : fp32 skel+canny 从零, subset=8, batch 48
#   3) exp_s6_top6_struct_fp32_full  : fp32 skel+canny 从零, 全 batch decode(subset=0), batch 8
#   4) exp_s6_top6_diff_then_struct  : resume exp1 最后 ckpt, 加 skel+canny (batch 192->48)
# 用法: 在 tmux 里 bash run_s6_experiments.sh
set -u
cd /root/Workspace/xy/DiT
MAINLOG=run_s6.log
echo "=== run_s6_experiments start $(date) ===" >> "$MAINLOG"

say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAINLOG"; }

stop_cpu_eval(){
  pkill -f auto_eval_cpu.py 2>/dev/null
  sleep 3
  say "killed old cpu eval (pgrep: $(pgrep -fc auto_eval_cpu.py 2>/dev/null || echo 0) left)"
}

start_cpu_eval(){
  local rd=$1
  local tag=${rd##*/}
  nohup /opt/conda/bin/python auto_eval_cpu.py --results-dir "5script/results/$rd" \
    --seen5-csv 5script/seen2_top6.csv --workers 4 --worker-threads 8 --interval 20 \
    > "cpu_eval_$tag.log" 2>&1 &
  say "cpu eval started for $rd (pid $!)"
}

# ---- 停掉当前 pixelfp32 的 cpu eval ----
stop_cpu_eval

# ---- 实验1: 纯 diff ----
say "===== EXP1 diffonly (batch 192) ====="
start_cpu_eval s6_top6_diffonly
/opt/conda/bin/python src/train.py --config exp_s6_top6_diffonly.json
say "===== EXP1 done ====="
stop_cpu_eval

# 实验4 阶段1 ckpt = exp1 最后 ckpt（diff 已收敛）
PHASE1=$(ls -t 5script/results/s6_top6_diffonly/*/checkpoints/*.pt 2>/dev/null | head -1)
say "phase1 ckpt: ${PHASE1:-NONE}"

# ---- 实验2: fp32 skel+canny 从零, subset=8 ----
say "===== EXP2 struct_fp32 (batch 48, subset 8) ====="
start_cpu_eval s6_top6_struct_fp32
/opt/conda/bin/python src/train.py --config exp_s6_top6_struct_fp32.json
say "===== EXP2 done ====="
stop_cpu_eval

# ---- 实验3: fp32 skel+canny 从零, 全 batch decode(subset=0) ----
say "===== EXP3 struct_fp32_full (batch 8, subset 0, 全量解码) ====="
start_cpu_eval s6_top6_struct_fp32_full
/opt/conda/bin/python src/train.py --config exp_s6_top6_struct_fp32_full.json
say "===== EXP3 done ====="
stop_cpu_eval

# ---- 实验4: diff 收敛后 resume + struct（batch 192->48 切换）----
say "===== EXP4 diff_then_struct ====="
if [ -n "$PHASE1" ]; then
  start_cpu_eval s6_top6_diff_then_struct
  /opt/conda/bin/python - <<PYEOF
import json
cfg = json.load(open('exp_s6_top6_diff_then_struct.json'))
cfg['resume_full'] = '$PHASE1'
json.dump(cfg, open('/tmp/exp4_final.json','w'), indent=2, ensure_ascii=False)
print('exp4 resume_full =', cfg['resume_full'], flush=True)
PYEOF
  /opt/conda/bin/python src/train.py --config /tmp/exp4_final.json
  say "===== EXP4 done ====="
  stop_cpu_eval
else
  say "!!! no phase1 ckpt found, skip exp4"
fi

say "===== ALL COMPLETE $(date) ====="