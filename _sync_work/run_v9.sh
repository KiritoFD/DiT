#!/bin/bash
set -u
cd /root/Workspace/xy/DiT || exit 1
export CUDA_VISIBLE_DEVICES=0
PY=/opt/conda/envs/cu121/bin/python
echo "=== v9: 纯 grid_sample(去残差) + 全局仿射 + 距离场 + TV/Jacobian + latent 正则 ==="
exec $PY -u tools/train_deform_standalone.py \
  --init assets/deform_skel_v5.pt --steps 12000 --batch 8192 --group 48 \
  --residual 0 --width 96 --max-off 6 --blur 0 --dt-ch 1 \
  --w-tv 0.01 --w-fold 0.1 --w-tv-out 0.01 --w-tv-res 0.01 \
  --w-img 0 --diag-decode 1 --img-batch 128 --eval-every 2000 \
  --out assets/deform_skel_v9.pt
