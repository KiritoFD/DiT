#!/bin/bash
# β 下游扫描: 每个 β 重新生成 predskel shards -> 喂冻结 v26 -> 报 ssim
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt

for BETA in 0.2 0.35 0.5; do
  echo "==================== β=$BETA ===================="
  $PY -u tools/gen_predskel_dit.py --set seen20   --beta $BETA 2>&1 | tail -1
  $PY -u tools/gen_predskel_dit.py --set strict84 --beta $BETA 2>&1 | tail -1
  $PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas " " \
      --pred-seen data/top10_style23/predskel_dit_seen20 \
      --pred-strict data/top10_style23/predskel_dit_strict84 \
      --out assets/results/_calib_dit_b$BETA 2>&1 | grep -A2 "predskel:"
done
echo "==================== 扫描完成 ===================="
