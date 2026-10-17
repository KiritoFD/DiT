#!/usr/bin/env bash
set -euo pipefail
cd /root/Workspace/xy/DiT
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
run="assets/style_encoder_v89/$(date +%Y%m%d-%H%M%S)"
echo "V89_ENCODER_RUN=$run"
echo "Encoder only: 20k steps, batch512, CE+0.3SupCon+0.1STD; no DiT training"
exec flock -n logs/strong_style_queue.lock \
  /opt/conda/envs/cu121/bin/python -u tools/train_prototype_style_encoder.py \
  --out "$run" --steps 20000 --batch 512 --lr 0.0001 --eval-every 1000
