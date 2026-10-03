#!/bin/bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
if pgrep -f 'src[.]train[.]train' >/dev/null; then
  echo '已有训练在跑, 不启动'; exit 1
fi
echo "=== $(date '+%F %T') v20 deform-skel 100k ==="
exec $PY -u -m src.train.train --config src/train/configs/v20_deform_skel_100k.json
