#!/bin/bash
# 训练结束后自动跑: 收尾膨胀宽度 W ∈ {3,5,7} px 的下游实测 (以效果选宽度, 不限 3px)
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
CKPT=assets/skelnet_pix64.pt

while pgrep -f "tools/train_skelnet_pix.py --steps" > /dev/null; do sleep 20; done
sleep 5
echo "=== 训练结束, 开始宽度扫描 (ckpt=$CKPT) ==="
ls -la $CKPT 2>/dev/null || { echo "!!! ckpt 不存在, 中止"; exit 1; }

for W in 3 5 7; do
  echo "==================== W=$W px ===================="
  $PY -u tools/train_skelnet_pix.py --dump --resume $CKPT --prep skel --res 64 \
      --dump-width $W --dump-tag w$W 2>&1 | grep -E '\[dump\]|\[cache\]'
  $PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
      --pred-seen data/top10_style23/predskel_pix_seen20_w$W \
      --pred-strict data/top10_style23/predskel_pix_strict84_w$W \
      --out assets/results/_calib_pix_w$W 2>&1 | grep "set=.*pred " | tail -2
done
echo "=== 宽度扫描完成 ==="
