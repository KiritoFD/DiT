#!/bin/bash
# 停掉已平台的训练 -> 用 best ckpt 出 predskel -> 冻结 v26 下游实测 + poster
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
V26=assets/results/v26_gtskel/20260929-103927-v26-gtskel/checkpoints/0030000.pt
CKPT=assets/skelnet_fm64.pt

pkill -f 'train_skelnet_[f]m' 2>/dev/null
pkill -f 'sweep_[w]idth' 2>/dev/null
sleep 6
echo "--- 训练尾部 ---"
grep -E 'eval\]|早停|DONE' /tmp/fm64.log | tail -3
echo "--- ckpt ---"
ls -la $CKPT

echo "===== dump W=3 (采样 -> 升采样4x -> 骨架化 -> 3px -> encode) ====="
$PY -u tools/train_skelnet_fm64.py --dump --resume $CKPT --dump-width 3 \
    --dump-tag w3 2>&1 | grep -E '\[dump\]|Traceback|Error'

echo "===== 下游: 冻结 v26 主干 ====="
$PY -u tools/run_skel_calibration.py --ckpt $V26 --alphas 0 \
    --pred-seen data/top10_style23/predskel_fm_seen20_w3 \
    --pred-strict data/top10_style23/predskel_fm_strict84_w3 \
    --out assets/results/_calib_fm_w3 2>&1 | grep -E 'set=.*pred ' | tail -2
echo "--- posters ---"
ls assets/results/_calib_fm_w3/posters/ 2>/dev/null | head -8
