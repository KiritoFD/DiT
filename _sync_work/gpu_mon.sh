#!/bin/bash
# 常驻 GPU 显存监测 (1Hz, 带时间戳) -> /tmp/gpu_mon.log
pkill -f gpu_mon.sh 2>/dev/null
while true; do
    echo "$(date +%H%M%S) $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    sleep 1
done
