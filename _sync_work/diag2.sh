#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== 进程 ==="
ps -eo pid,etime,pcpu,cmd | grep -E '[t]rain_skelnet|[e]val_pix32' | head -5
echo
echo "=== 训练日志 (末尾) ==="
tail -12 logs/skelnet_fm_pix32e.log 2>/dev/null
echo
echo "=== eval 历史 ==="
grep -E 'eval\]' logs/skelnet_fm_pix32e.log 2>/dev/null | tail -8
echo
echo "=== 评测队列日志 ==="
tail -6 /tmp/eval_pix32b.log 2>/dev/null || echo "(无)"
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader
