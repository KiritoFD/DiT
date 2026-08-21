#!/bin/bash
# 远程执行: 拉起 canny=0.5 skel=1.0 的续训练(B模型) + CPU eval 跟随新实验
set -u
cd ~/Workspace/xy/DiT
PY=/opt/conda/bin/python
CFG=exp_s5_2factor_B_canny05_pixelsk.json
RESUME="5script/results/s5_2factor_B_latentstruct_pixelsk_opt/20260818-004938-s5-2factor-B-latentC-pixelsk-opt/checkpoints/0115000.pt"
RES_DIR="5script/results/s5_2factor_B_canny05_pixelsk"
EXP_LOG="exp_s5-2factor-B-canny05-pixelsk.log"
EVAL_LOG="cpu_eval_s5-2factor-B-canny05-pixelsk.log"

tmux kill-session -t train_gpu 2>/dev/null
tmux kill-session -t eval_cpu 2>/dev/null
sleep 1

tmux new-session -d -s train_gpu -n train \
  "cd ~/Workspace/xy/DiT && $PY src/train.py --config $CFG --resume-full $RESUME 2>&1 | tee $EXP_LOG"

tmux new-session -d -s eval_cpu -n eval \
  "cd ~/Workspace/xy/DiT && $PY auto_eval_cpu.py --results-dir $RES_DIR --workers 8 --worker-threads 8 --interval 20 2>&1 | tee $EVAL_LOG"

echo "=== tmux ==="; tmux ls
sleep 20
echo "=== train log tail ==="; tail -6 $EXP_LOG 2>/dev/null
echo "=== eval log tail ==="; tail -4 $EVAL_LOG 2>/dev/null
echo "=== gpu ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader