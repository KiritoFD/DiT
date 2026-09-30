#!/usr/bin/env bash
# run_v26_gtskel.sh — v26-gtskel: 把 v25 的 std skel 换成 GT skel, 150k 从零训练
#
# 与 v25_stdskel 的唯一差异 (config 层面):
#   skel_latent_shards_dir      : data/top10_style23/shards_std   -> data/top10_style23/shards_aux_skel3
#   eval_skel_latent_shards_dir : data/50k_v2_glyph15k/shards_std -> data/top10_style23/gt_skel_eval_strict84
#   + 双口径 eval (seen_pred/strict_pred -> predskel shards)
# 其余逐项一致。
#
# 断点续训:  RESUME=<ckpt.pt> bash run_v26_gtskel.sh
cd /root/Workspace/xy/DiT
mkdir -p logs
tmux kill-session -t v26_gtskel 2>/dev/null || true

RESUME_ARG=""
if [ -n "${RESUME:-}" ]; then RESUME_ARG="--resume-full $RESUME"; fi
echo "RESUME_ARG = '$RESUME_ARG'"

tmux new-session -d -s v26_gtskel "cd /root/Workspace/xy/DiT && PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train --config src/train/configs/v26_gtskel.json $RESUME_ARG 2>&1 | tee /root/Workspace/xy/DiT/logs/v26_gtskel.log"
echo "tmux session 'v26_gtskel' launched!  tail -f logs/v26_gtskel.log"
