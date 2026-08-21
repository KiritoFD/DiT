#!/bin/bash
# 启动训练：GPU 训练(train_gpu) + CPU 评测(auto_eval_cpu) 两个独立 tmux 会话。
# 训练只负责训练 + 保存 ckpt；评测全部由 auto_eval_cpu.py 在 CPU 端独立完成，
# eval 代码可随时修改而无需重启训练。
#
# 用法: ./launch_train.sh [config.json] [auto_eval_cpu 额外参数...]
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PY=/opt/conda/bin/python
CFG="${1:-exp_s5_2factor_B_latentstruct_pixelsk_opt.json}"
shift || true

TRAIN_ONLY=0
EVAL_ONLY=0
if [ "${1:-}" = "--train-only" ]; then
  TRAIN_ONLY=1
  shift
fi
if [ "${1:-}" = "--eval-only" ]; then
  EVAL_ONLY=1
  shift
fi

RES_DIR=$("$PY" -c "import json,sys; print(json.load(open('$CFG'))['results_dir'])")
EXP_NAME=$("$PY" -c "import json,sys; print(json.load(open('$CFG')).get('experiment_name','exp'))")
LOG="exp_${EXP_NAME}.log"
EVAL_LOG="cpu_eval_${EXP_NAME}.log"

if [ "$EVAL_ONLY" = "1" ]; then
  # 仅重启 CPU 评测进程（不触碰正在运行的训练）
  tmux kill-session -t eval_cpu 2>/dev/null
  sleep 1
  tmux new-session -d -s eval_cpu -n eval \
    "cd '$SCRIPT_DIR' && $PY auto_eval_cpu.py --results-dir $RES_DIR $* 2>&1 | tee $EVAL_LOG"
  echo "[launch] eval-only mode: 已启动 CPU 评测 (train_gpu 不受影响)"
  echo "[launch] CPU eval session : eval_cpu    (log: $EVAL_LOG)"
  echo "[launch] attach: tmux a -t eval_cpu"
  exit 0
fi

# 1) 清理旧会话（旧进程退出后 GPU/CPU 资源释放）
tmux kill-session -t train_gpu 2>/dev/null
tmux kill-session -t eval_cpu 2>/dev/null
sleep 1

# 2) GPU 训练：只训练 + 保存 ckpt
tmux new-session -d -s train_gpu -n train \
  "cd '$SCRIPT_DIR' && $PY src/train.py --config $CFG 2>&1 | tee $LOG"

# 3) CPU 评测：轮询新 ckpt，独立评测（指标 + show5 + seen5 展示）
if [ "$TRAIN_ONLY" = "1" ]; then
  echo "[launch] train-only mode: 不启动 CPU 评测进程"
else
  tmux new-session -d -s eval_cpu -n eval \
    "cd '$SCRIPT_DIR' && $PY auto_eval_cpu.py --results-dir $RES_DIR $* 2>&1 | tee $EVAL_LOG"
  echo "[launch] CPU eval session : eval_cpu    (log: $EVAL_LOG)"
fi
echo "[launch] results dir      : $RES_DIR"
echo "[launch] ckpt marker      : $RES_DIR/_active_ckpt_dir.txt"
echo "[launch] attach: tmux a -t train_gpu | tmux a -t eval_cpu"