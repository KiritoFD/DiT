#!/bin/bash
# run_patch_now.sh — 训练间隙跑 DINO patch 分析 (低 GPU 占用, nice 低优先级)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
nice -n 10 /opt/conda/envs/cu121/bin/python /tmp/dino_patch_discrim.py 2>&1 | tee /tmp/dino_patch_result.log