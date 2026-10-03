#!/bin/bash
# 7px 对照: (a) H(w7) raw 直接喂 (b) H(w7)+renorm 归一到3px
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
H=assets/skelnet_dit_H_bridge_nog_w7.pt

echo "======== (a) w7 raw 直接喂 ========"
$PY -u tools/gen_predskel_dit.py --set seen20   --resume $H --beta 0.634 --tag w7raw 2>&1 | tail -1
$PY -u tools/gen_predskel_dit.py --set strict84 --resume $H --beta 0.634 --tag w7raw 2>&1 | tail -1
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_dit_seen20_w7raw \
    --pred-strict data/top10_style23/predskel_dit_strict84_w7raw \
    --out assets/results/_calib_dit_w7raw 2>&1 | grep "set=.*pred " | tail -2

echo "======== (b) w7 + renorm(归一3px) ========"
$PY -u tools/gen_predskel_dit.py --set seen20   --resume $H --beta 0.634 --renorm --tag w7ren 2>&1 | tail -1
$PY -u tools/gen_predskel_dit.py --set strict84 --resume $H --beta 0.634 --renorm --tag w7ren 2>&1 | tail -1
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_dit_seen20_w7ren \
    --pred-strict data/top10_style23/predskel_dit_strict84_w7ren \
    --out assets/results/_calib_dit_w7ren 2>&1 | grep "set=.*pred " | tail -2
echo W7_ARMS_DONE
