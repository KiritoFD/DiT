#!/bin/bash
# 用**同一把尺子**量"什么都不做": 把 std 骨架当 pred 喂进同一个 harness (v32 主干)
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
V32=$(ls -1 assets/results/v32_stage2_img/*/checkpoints/0080000.pt | head -1)

$PY -u tools/run_skel_calibration.py --ckpt "$V32" --alphas 0 \
    --pred-seen   data/top10_style23/shards_std \
    --pred-strict data/50k_v2_glyph15k/shards_std \
    --out assets/results/_calib_stdas_pred_v32 2>&1 \
    | grep -E 'set=.*pred ' | tail -4

echo "--- 已有 poster 位置 ---"
ls -1 assets/results/_calib_s1_v32/posters/*.png 2>/dev/null | head -6
ls -1 assets/results/_calib_stdas_pred_v32/posters/*.png 2>/dev/null | head -4
echo STDAS_DONE
