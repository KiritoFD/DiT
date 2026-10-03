#!/usr/bin/env bash
# 查: 真实 batch / 当前显存 / 显存是否在波动 / GPU 时间线
set -u
cd /root/Workspace/xy/DiT || exit 1
date
echo
echo "=== [1] 真正在跑的 argv (看 batch size) ==="
pgrep -af 'src/train/train.py' | head -2
echo
echo "=== [2] 显存连续采样 (8 次 x 2 秒, 看是否波动) ==="
for i in 1 2 3 4 5 6 7 8; do
  nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
  sleep 2
done
echo
echo "=== [3] 日志里的 Mem / Steps/Sec 轨迹 ==="
LOG=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
echo "log=$LOG"
grep -a 'Steps/Sec' "$LOG" 2>/dev/null | tail -8
echo "--- alloc 行 ---"
grep -a 'alloc' "$LOG" 2>/dev/null | tail -8
echo
echo "=== [4] 配置里的 batch ==="
grep -a 'global_batch_size' src/train/configs/v50_A_space_xattn_style_adaLN_top10.json
echo
echo "=== [5] GPU 上的进程 ==="
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
