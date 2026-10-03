#!/bin/bash
# run_v10b_5exp_chain.sh — v10b 五实验串行链 (B/C + 42号 3 个)
#
# 顺序:
#   1. v10b-repa    (REPA 强化 w0.5 + layers 8,11)      [方案B]
#   2. v10b-inject  (深层注入 4层 + scale 0.6 + drop 0.05) [方案C]
#   3. v10b-callig  (callig 链增强 dim256 + MLP + scale 1.5)
#   4. v10b-norepa  (REPA 关闭)
#   5. v10b-shallow (REPA 浅层 2,4 + w0.05)
#
# 每个实验: 独立 tmux 训练会话 + cpu_eval_daemon watch 切到对应 results_dir
# (flat eval_auto json, 与 v10b 基线曲线可比)。前一个训练结束 (早停/max_steps,
# tmux 会话自动退出) 后, chain 继续下一个。
#
# 用法: bash /root/Workspace/xy/DiT/_sync_work/run_v10b_5exp_chain.sh > /tmp/v10b_5exp_chain.log 2>&1
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
RES_ROOT=assets/results

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── 单实验: 起训练 + 切 daemon ──
launch_exp() { # name config results_dir
  local name=$1 cfg=$2 res=$3
  log "===== LAUNCH $name ($cfg) ====="
  # 清残留
  tmux kill-session -t "$name" 2>/dev/null
  pkill -f "train.py --config src/train/configs/$cfg" 2>/dev/null
  sleep 3
  # 训练 (独立 tmux, 自动早停后会话退出)
  tmux new-session -d -s "$name" \
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/$cfg.json 2>&1 | tee /tmp/${name}_train.log"
  # daemon 切到本实验 results_dir (保证 eval 正确)
  tmux kill-session -t cpu_eval 2>/dev/null
  tmux new-session -d -s cpu_eval \
    "export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/$RES_ROOT/$res --threads 32 > /tmp/cpu_eval_daemon_${name}.log 2>&1"
  sleep 8
  if tmux has-session -t "$name" 2>/dev/null; then
    log "  tmux $name OK; GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  else
    log "  !! $name tmux FAILED"; tail -5 /tmp/${name}_train.log; return 1
  fi
  # 确认模型构建成功 (Building 2-Cond 出现 = 进入训练)
  sleep 30
  grep -m1 "Building 2-Cond" /tmp/${name}_train.log >/dev/null && log "  model built OK" \
    || { log "  !! model build pending/failed"; tail -5 /tmp/${name}_train.log; }
}

# ── 等待训练结束 (tmux 会话消失 = 训练进程退出) ──
wait_exp() { # name
  local name=$1
  log "===== WAIT $name (训练至早停/max_steps) ====="
  local n=0
  while tmux has-session -t "$name" 2>/dev/null; do
    sleep 120
    n=$((n+1))
    if [ $((n % 15)) -eq 0 ]; then
      local step=$(grep -oE "step=[0-9]+" /tmp/${name}_train.log 2>/dev/null | tail -1)
      log "  $name 仍训练中 ($step)"
    fi
  done
  log "  $name 训练结束"
}

# ══════════ 串行执行 ══════════
log "### v10b 5 实验串行链开始 ###"
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader

launch_exp v10brepa   v10brepa_strong  v10brepa_strong
wait_exp  v10brepa

launch_exp v10binj    v10binject       v10binject
wait_exp  v10binj

launch_exp v10bcallig v10bcallig_strong v10bcallig_strong
wait_exp  v10bcallig

launch_exp v10bnorepa v10bnorepa       v10bnorepa
wait_exp  v10bnorepa

launch_exp v10bshal   v10bshallow_repa v10bshallow_repa
wait_exp  v10bshal

log "### 全部 5 实验完成 ###"
tmux ls