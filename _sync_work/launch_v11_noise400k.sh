#!/bin/bash
# launch_v11_noise400k.sh - 启动 400k 条件噪声增强训练与双轨 (origin + self-cond) 评测
set -e

cd /root/Workspace/xy/DiT
mkdir -p logs/v11_pretrain_M432_adaln4_sym_noise400k

# 清理可能存在的旧 tmux 会话
tmux kill-session -t v11_noise400k 2>/dev/null || true
tmux kill-session -t gpueval_v11_noise400k 2>/dev/null || true

TS=$(date +%Y%m%d-%H%M%S)
TRAIN_LOG="/root/Workspace/xy/DiT/logs/v11_pretrain_M432_adaln4_sym_noise400k/train_${TS}.log"
EVAL_LOG="/root/Workspace/xy/DiT/logs/v11_pretrain_M432_adaln4_sym_noise400k/gpu_eval.log"

CKPT="/root/Workspace/xy/DiT/assets/results/v11_pretrain_M432_adaln4_sym/20260913-164820-v11-pretrain-M432-adaln4-sym/checkpoints/0042500.pt"
CFG="src/train/configs/v11_pretrain_M432_adaln4_sym_noise400k.json"
RESULTS_DIR="assets/results/v11_pretrain_M432_adaln4_sym_noise400k"

echo "=== 启动训练任务 v11_noise400k ==="
TERM=xterm tmux new-session -d -s v11_noise400k \
  "bash -c 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; /opt/conda/envs/cu121/bin/python -u src/train/train.py --config ${CFG} --resume-full ${CKPT} --fresh-scheduler 1 2>&1 | tee ${TRAIN_LOG}'"

echo "=== 启动双轨评测监听任务 gpueval_v11_noise400k ==="
TERM=xterm tmux new-session -d -s gpueval_v11_noise400k \
  "bash -c 'cd /root/Workspace/xy/DiT && PYTHONPATH=/root/Workspace/xy/DiT /opt/conda/envs/cu121/bin/python -u tools/eval/gpu_eval_loop.py --results-dir ${RESULTS_DIR} --device cuda --poll 30 --strict-every 10000 --self-cond --blend-alpha 0.5 > ${EVAL_LOG} 2>&1'"

echo "=== 任务已提交 ==="
tmux ls
