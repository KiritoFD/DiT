#!/bin/bash
# _sync_work/monitor_round.sh — 单次监控快照，返回 JSON 格式状态
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT

now=$(date '+%F %T')
s30_log=/tmp/s30_train_pipe.log
s31_log=/tmp/s31_ctrl_pipe.log
s32_log=/tmp/s32_repa_pipe.log

stage="unknown"
if pgrep -f 'train.py.*s30' >/dev/null 2>&1; then stage="A"
elif pgrep -f 'train_controlnet.py.*s31' >/dev/null 2>&1; then stage="B"
elif pgrep -f 'train_repa.py' >/dev/null 2>&1; then stage="C"
else
  if [ -f "$s30_log" ]; then stage="A(stop)"; fi
  if [ -f "$s31_log" ]; then stage="B(stop)"; fi
  if [ -f "$s32_log" ]; then stage="C(stop)"; fi
fi

# S30
s30_step=""; s30_total=""; s30_sps=""; s30_es=""
if [ -f "$s30_log" ]; then
  s30_step=$(tail -200 "$s30_log" 2>/dev/null | grep -oE 'step=[0-9]+' | tail -1 | tr -d 'step=')
  s30_total=$(tail -200 "$s30_log" 2>/dev/null | grep -oE 'Total: [0-9.]+' | tail -1 | awk '{print $2}')
  s30_sps=$(tail -200 "$s30_log" 2>/dev/null | grep -oE 'Steps/Sec: [0-9.]+' | tail -1 | awk '{print $2}')
  s30_mem=$(tail -200 "$s30_log" 2>/dev/null | grep -oE 'Mem: [0-9.]+G' | tail -1 | awk '{print $2}')
  s30_es=$(grep -a 'early-stop' "$s30_log" 2>/dev/null | tail -1 | sed 's/.*\[early-stop\] //' | tr -d '\n' 2>/dev/null)
  s30_es_step=$(grep -a 'early-stop' "$s30_log" 2>/dev/null | tail -1 | grep -oE 'step [0-9]+' | awk '{print $2}')
fi

# S31
s31_step=""; s31_loss=""; s31_es=""
if [ -f "$s31_log" ]; then
  s31_step=$(tail -200 "$s31_log" 2>/dev/null | grep -oE 'step=[0-9]+' | tail -1 | tr -d 'step=')
  s31_loss=$(tail -200 "$s31_log" 2>/dev/null | grep -oE 'loss=[0-9.]+' | tail -1 | awk -F= '{print $2}')
  s31_es=$(grep -a 'early-stop' "$s31_log" 2>/dev/null | tail -1 | sed 's/.*\[early-stop\] //' | tr -d '\n' 2>/dev/null)
fi

# S32
s32_step=""; s32_loss=""; s32_all=""
if [ -f "$s32_log" ]; then
  s32_step=$(tail -200 "$s32_log" 2>/dev/null | grep -oE 'step=[0-9]+' | tail -1 | tr -d 'step=')
  s32_loss=$(tail -200 "$s32_log" 2>/dev/null | grep -oE 'loss_diff=|loss_total=' | tail -1)
  s32_all=$(tail -3 "$s32_log" 2>/dev/null | tr '\n' ';' | sed 's/;/ | /g')
fi

# GPU
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1 | tr -d ' MiB')

echo "{\"now\":\"$now\",\"stage\":\"$stage\",\"s30\":{\"step\":\"$s30_step\",\"total\":\"$s30_total\",\"sps\":\"$s30_sps\",\"mem\":\"$s30_mem\",\"es\":\"$s30_es\",\"es_step\":\"$s30_es_step\"},\"s31\":{\"step\":\"$s31_step\",\"loss\":\"$s31_loss\",\"es\":\"$s31_es\"},\"s32\":{\"step\":\"$s32_step\",\"loss\":\"$s32_loss\",\"tail\":\"$s32_all\"},\"gpu_mem_mb\":\"$gpu_mem\"}"