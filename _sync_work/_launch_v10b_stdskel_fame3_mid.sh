#!/bin/bash
# 拉起 v10b-stdskel-fame3-mid 训练 + cpu_eval daemon
# = d01 + w_std_mid=0.05 (flow 等效 sqrt_alpha=1-t 已修; early_stop 关闭, 验收看 follow-IoU3)
# 顶掉 d01: d01 的单变量结论已得出 (drop 0.1≈0.25, canonical IoU3: d01@17500=0.197 vs f3@17500=0.224)
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
NAME=v10bstdskel3mid
CFG=v10b_stdskel_fame3_mid
RES=v10b_stdskel_fame3_mid

tmux kill-session -t v10bstdskel3d01 2>/dev/null
pkill -f "train.py --config src/train/configs/v10b_stdskel_fame3_d01" 2>/dev/null
sleep 3
tmux kill-session -t "$NAME" 2>/dev/null
pkill -f "train.py --config src/train/configs/$CFG" 2>/dev/null
sleep 2
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/$CFG.json 2>&1 | tee /tmp/${NAME}_train.log"
# eval daemon 切到本实验 results_dir (pretrain_g: g=std 骨架 shards, 与训练一致)
tmux kill-session -t cpu_eval 2>/dev/null
tmux new-session -d -s cpu_eval \
  "export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/eval/cpu_eval_daemon.py --mode pretrain_g --watch-root /root/Workspace/xy/DiT/assets/results/$RES --threads 32 > /tmp/cpu_eval_daemon_${NAME}.log 2>&1"
sleep 15
echo "tmux sessions:"; tmux ls 2>/dev/null | grep -E "$NAME|cpu_eval"
echo "--- train log ---"; tail -8 /tmp/${NAME}_train.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
