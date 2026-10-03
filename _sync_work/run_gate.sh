#!/bin/bash
# 极端实验: 骨架时间门, 下限 0.15。先停掉低秩头那条。
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python

echo "=== $(date '+%F %T') stop previous gate run ==="
tmux kill-session -t gate 2>/dev/null || true
for p in $(pgrep -f 'v17_glyph_gate_f015'); do
  kill "$p" 2>/dev/null || true
done
sleep 8
if pgrep -f 'src.train.train' >/dev/null; then
  echo "训练进程还在, 不启动"
  pgrep -af 'src.train.train' | cut -c1-140
  exit 1
fi
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "=== $(date '+%F %T') train ==="
exec $PY -u -m src.train.train --config src/train/configs/v17_glyph_gate_f015.json
