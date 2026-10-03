#!/bin/bash
# run_sty_scan_chain.sh — 风格 token 容量扫描串行链 (sty16 / sty32 / sty64)
# 每个: 从零训练 30k 步 + cpu_eval daemon 产出 eval_auto json
# 不 resume 旧 ckpt: 实测 key 表面兼容但 out_proj 会被旧值覆盖为非零,
#   随机 style token 会污染已训模型 (见 _chk_resume_compat.py)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

launch_exp() { # name cfg results_dir
  local name=$1 cfg=$2 res=$3
  log "===== LAUNCH $name (style_token_n from $cfg) ====="
  tmux kill-session -t "$name" 2>/dev/null
  pkill -f "train.py --config src/train/configs/$cfg" 2>/dev/null
  sleep 3
  tmux new-session -d -s "$name" \
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/$cfg.json 2>&1 | tee /tmp/${name}_train.log"
  # eval daemon 切到本实验
  tmux kill-session -t cpu_eval 2>/dev/null
  tmux new-session -d -s cpu_eval \
    "export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/$res --threads 32 > /tmp/cpu_eval_${name}.log 2>&1"
  sleep 12
  if tmux has-session -t "$name" 2>/dev/null; then
    log "  $name OK; GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  else
    log "  !! $name tmux FAILED"; tail -5 /tmp/${name}_train.log; return 1
  fi
  sleep 30
  grep -m1 "Building 2-Cond" /tmp/${name}_train.log >/dev/null \
    && log "  model built OK" || { log "  !! model build pending"; tail -5 /tmp/${name}_train.log; }
}

wait_exp() { # name
  local name=$1 n=0
  log "===== WAIT $name ====="
  while tmux has-session -t "$name" 2>/dev/null; do
    sleep 120
    n=$((n+1))
    if [ $((n % 15)) -eq 0 ]; then
      log "  $name 训练中: $(grep -oE 'step=[0-9]+' /tmp/${name}_train.log 2>/dev/null | tail -1)"
    fi
  done
  log "  $name 完成"
}

log "### 风格 token 容量扫描开始 (16/32/64, 各 30k, 从头训练) ###"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

launch_exp sty16 c41x_sty16 c41x_sty16
wait_exp sty16

launch_exp sty32 c41x_sty32 c41x_sty32
wait_exp sty32

launch_exp sty64 c41x_sty64 c41x_sty64
wait_exp sty64

log "### 三个实验全部完成 ###"
tmux ls
