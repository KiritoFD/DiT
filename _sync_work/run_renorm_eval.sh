#!/bin/bash
# renorm 去曝光版: 生成 -> 下游实测
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
$PY -u tools/gen_predskel_dit.py --set seen20 --renorm 2>&1 | tail -1
$PY -u tools/gen_predskel_dit.py --set strict84 --renorm 2>&1 | tail -1
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_dit_seen20 \
    --pred-strict data/top10_style23/predskel_dit_strict84 \
    --out assets/results/_calib_dit_renorm 2>&1 | grep -E "set=.*pred " | tail -2
echo RENORM_EVAL_DONE
