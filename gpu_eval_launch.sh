#!/bin/bash
# 远程执行: 停训练 + 停CPU eval + 备份state + 启动GPU eval(tmux eval_gpu, --once)
set -u
cd ~/Workspace/xy/DiT
CKPT="5script/results/s5_2factor_B_latentstruct_pixelsk_opt/20260818-004938-s5-2factor-B-latentC-pixelsk-opt/checkpoints"
EVAL_LOG="cpu_eval_gpu_s5-2factor-B-latentC-pixelsk-opt.log"

tmux kill-session -t train_gpu 2>/dev/null
tmux kill-session -t eval_cpu 2>/dev/null
sleep 2
cp -f cpu_eval_state.json cpu_eval_state.bak.json 2>/dev/null

tmux kill-session -t eval_gpu 2>/dev/null
sleep 1
tmux new-session -d -s eval_gpu -n eval \
  "cd ~/Workspace/xy/DiT && /opt/conda/bin/python auto_eval_cpu.py --ckpt-dir $CKPT --device cuda --workers 1 --once 2>&1 | tee $EVAL_LOG"

echo "=== tmux ==="; tmux ls
echo "=== ckpts(pt) ==="; ls $CKPT/*.pt 2>/dev/null | wc -l
echo "=== eval jsons done ==="; ls $CKPT/eval_auto_*.json 2>/dev/null | wc -l
echo "=== nvidia-smi ==="; nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
sleep 8
echo "=== log tail ==="; tail -5 $EVAL_LOG 2>/dev/null