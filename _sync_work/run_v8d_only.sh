#!/bin/bash
# run_v8d_only.sh — 补跑 v8d (unfreeze-main)。首次因 param group bug 失败, 已修复。
# 等待 posttrain 链(当前跑 v8h)结束后, 从 A_main_final 起跑 v8d。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
PY=/opt/conda/envs/cu121/bin/python
A=assets/results/v8_3stage/A_main_final.pt
LOG=/tmp/v8d_retry.log

# 等 posttrain tmux 会话结束 (避免抢 GPU)
while tmux has-session -t posttrain 2>/dev/null; do
  echo "$(date '+%F %T') waiting for posttrain chain ..." >> $LOG
  sleep 300
done

echo "=== [v8d-retry] $(date '+%F %T') 启动 ===" >> $LOG
$PY src/train/train_controlnet.py \
  --config src/train/configs/v8d_unfreeze_main.json \
  --main-ckpt "$A" > /tmp/v8d.log 2>&1
echo "[v8d-retry] rc=$? $(date '+%F %T')" >> $LOG