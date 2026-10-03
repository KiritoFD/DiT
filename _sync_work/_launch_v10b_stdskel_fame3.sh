#!/bin/bash
# 拉起 v10b-stdskel-fame3 训练 + cpu_eval daemon (fame-3 = 楷/行/隶, g=std 骨架, glyph_drop 0.25)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
NAME=v10bstdskel3
CFG=v10b_stdskel_fame3
RES=v10b_stdskel_fame3

tmux kill-session -t "$NAME" 2>/dev/null
pkill -f "train.py --config src/train/configs/$CFG" 2>/dev/null
sleep 3
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/$CFG.json 2>&1 | tee /tmp/${NAME}_train.log"
# eval daemon 切到本实验 results_dir (pretrain_g: g=skel_latent_shards_dir=std 骨架, 与训练一致)
tmux kill-session -t cpu_eval 2>/dev/null
tmux new-session -d -s cpu_eval \
  "export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/$RES --threads 32 > /tmp/cpu_eval_daemon_${NAME}.log 2>&1"
sleep 10
echo "tmux sessions:"; tmux ls 2>/dev/null | grep -E "$NAME|cpu_eval"
echo "--- train log ---"; tail -4 /tmp/${NAME}_train.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
