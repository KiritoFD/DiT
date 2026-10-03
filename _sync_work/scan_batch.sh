#!/usr/bin/env bash
# 扫出能跑的最大 batch (144 / 152), 每个 40 步, compile 开, 记录 nvidia-smi 峰值
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
mkdir -p exp-std/logs_smoke

trial() {
  local B="$1"
  local LOG="exp-std/logs_smoke/scan_B${B}.log"
  echo ""
  echo "=========== batch=$B ==========="
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  ( $PY -u src/train/train.py \
      --config src/train/configs/v50_A_space_xattn_style_adaLN_top10.json \
      --experiment-name "scan_B${B}" --results-dir exp-std/runs_smoke \
      --global-batch-size "$B" --skel-latent-shards-weights 1.0,0.0 \
      --max-steps 40 --lr 5e-5 > "$LOG" 2>&1 ) &
  local SP=$!
  local MAX=0 V
  while kill -0 $SP 2>/dev/null; do
    V=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    V=${V:-0}
    [ "$V" -gt "$MAX" ] 2>/dev/null && MAX=$V
    sleep 1
  done
  wait $SP
  echo "  nvidia-smi 峰值 = ${MAX} MiB / 24564 MiB"
  if grep -aq 'OutOfMemoryError' "$LOG"; then
    echo "  结果: ✗ OOM"
    grep -a 'OutOfMemoryError' "$LOG" | head -1 | cut -c1-200
  else
    echo "  结果: ✓ 通过"
    grep -aE 'Steps/Sec' "$LOG" | tail -2 | sed 's/\x1b\[[0-9;]*m//g'
    grep -a 'Mem:' "$LOG" | tail -1 | sed 's/\x1b\[[0-9;]*m//g'
  fi
  sleep 5
}

trial 144
trial 152

echo ""
echo "=== 汇总 ==="
for B in 128 144 152 160 192; do
  L="exp-std/logs_smoke/scan_B${B}.log"
  [ -f "$L" ] || L=$(ls -t exp-std/logs_smoke/*B${B}*.log 2>/dev/null | head -1)
  [ -f "${L:-}" ] || continue
  if grep -aq 'OutOfMemoryError' "$L"; then R="✗ OOM"; else R="✓ 通过"; fi
  printf "  batch=%-4s %s\n" "$B" "$R"
done
