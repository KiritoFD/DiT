#!/bin/bash
set -u
cd /root/Workspace/xy/DiT || exit 1
CKPT=/root/Workspace/xy/DiT/assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60/20260916-100932-v12-pretrain-S-cat-fame-kxl-tj-px60/checkpoints/0055000.pt
LOGD=/root/Workspace/xy/DiT/logs/v12_pretrain_S_cat_fame_kxl_tj_px60
TS=$(date +%Y%m%d-%H%M%S)
tmux kill-session -t v12cont 2>/dev/null
pkill -f 'train.py --config src/train/configs/' 2>/dev/null
sleep 5
tmux new-session -d -s v12cont "export PYTHONPATH=/root/Workspace/xy/DiT; export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; /opt/conda/envs/cu121/bin/python -u src/train/train.py --config src/train/configs/v12_pretrain_S_cat_fame_kxl_tj_px60.json --resume-full $CKPT 2>&1 | tee $LOGD/train_resume_$TS.log"
sleep 3
tmux ls