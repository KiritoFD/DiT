#!/bin/bash
echo "=== 1054273 是谁 ==="
ps -p 1054273 -o pid,etime,rss,cmd --no-headers 2>/dev/null || echo "  已退出"
echo
echo "=== 所有 python 进程 ==="
ps -eo pid,etime,rss,cmd | grep '[p]ython' | head -8
echo
echo "=== 训练日志开头 ==="
head -14 /tmp/pix64_train.log 2>/dev/null
echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null
