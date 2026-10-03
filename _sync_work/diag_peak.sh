#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== fix2 训练 (train loss + eval) ==="
grep -E 'step |eval\]' logs/skelnet_fm64_fix2.log 2>/dev/null | tail -30
echo
echo "=== 对照 fix1 (门控 t<=0.3) ==="
grep -E 'eval\]' logs/skelnet_fm64_fix.log 2>/dev/null | tail -6
echo
echo "=== 进程/GPU ==="
pgrep -af 'train_skelnet_fm64' | head -2
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
