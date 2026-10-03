#!/bin/bash
# 探针二: 三臂对比 (adaLN / xattn / K4) 单 batch 极限记忆
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python
for ARM in adaln k4; do
  echo "########## arm=$ARM ##########"
  $PY -u tools/injection_overfit_probe.py --arm "$ARM" --steps 500 2>&1 \
    | grep -aE 'freeze|batch|latent\]|loss: init|step  (  1| 250| 500)|探针二|平台期|判读' | tail -9
done
