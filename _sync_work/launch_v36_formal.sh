#!/bin/bash
# v36 union 正式训练 (全量训练集). 前置: predskel* 脏目录已隔离到 _trash/。
# ★ 评测条件必须用干净的标准骨架: --shards-std-eval data/50k_v2_glyph15k/shards_std
#   (strict84 csv 的 84/84 命中; 原来的 predskel_std84_e2e 是脏数据, 会把 e2e 打成 0.01)
# 铁律: 不设 expandable_segments; 训练占卡时不并发跑任何评测。
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
mkdir -p logs assets/results/v36_union
$PY -u tools/train_joint_v36.py \
  --csv assets/train_top10_style23_minusval.csv \
  --shards-img data/top10_style23/shards_img \
  --shards-std data/top10_style23/shards_std \
  --shards-gt  data/top10_style23/shards_gtskel_w3 \
  --shards-std-eval data/50k_v2_glyph15k/shards_std \
  --batch 24 --gen-lr 2e-5 \
  --max-steps 40000 --eval-every 2500 --ckpt-every 2500 --log-every 50 \
  --results-dir assets/results/v36_union \
  2>&1 | tee logs/v36_union.log
