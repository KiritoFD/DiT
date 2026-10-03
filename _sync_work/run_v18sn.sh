#!/bin/bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
if pgrep -f "src[.]train[.]train" >/dev/null; then echo "已有训练在跑"; exit 1; fi
echo "=== v18-skelnet 100k start ==="
exec $PY -u -m src.train.train --config src/train/configs/v18_skelnet_100k.json
