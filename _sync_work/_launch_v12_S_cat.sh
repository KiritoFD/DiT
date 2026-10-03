#!/bin/bash
# v12: S/2 + factorized_cat(+glyph_vec) 从头训练
#
# 与 _launch_v10b_stdskel_fame3_variant.sh 的区别:
#   1) 不启动 gpu_eval_loop —— v12 config 已开 in_mem_eval=true (训练进程内评测),
#      再叠一个 gpu_eval_loop(--pause-train, SIGSTOP 训练) 会双重评测且互相打断。
#   2) 启动前先把 v11 的 best ckpt copy 到固定无时间戳路径 (remote.md §4)。
set -u

NAME=${1:?tmux name}
CFG=${2:?config name}
RES=${3:?results dir}
V11_BEST_SRC=${4:-}
V11_BEST_DST=${5:-}

cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/$RES
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGD"
mkdir -p /root/Workspace/xy/DiT/assets/registry/configs
cp /root/Workspace/xy/DiT/src/train/configs/$CFG.json \
   /root/Workspace/xy/DiT/assets/registry/configs/$CFG.json 2>/dev/null || true

# ---- 1. 保底: 把 v11 best ckpt 复制到固定路径 (不依赖时间戳目录) ----
if [ -n "$V11_BEST_SRC" ] && [ -f "$V11_BEST_SRC" ]; then
    cp -v "$V11_BEST_SRC" "$V11_BEST_DST"
    ls -la "$V11_BEST_DST"
else
    echo "[launch] WARN: v11 best ckpt not found at '$V11_BEST_SRC' (skip preserve)"
fi

# ---- 2. 停掉旧训练 ----
tmux kill-session -t "$NAME" 2>/dev/null
pkill -f "train.py --config src/train/configs/" 2>/dev/null
sleep 5
echo "[launch] after pkill:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ---- 3. 启动 v12 (in_mem_eval 在 config 里, 不再另起 eval daemon) ----
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; $PY -u src/train/train.py --config src/train/configs/$CFG.json 2>&1 | tee $LOGD/train_$TS.log"

sleep 20
echo "[launch] tmux:"; tmux ls 2>/dev/null
echo "[launch] log tail:"; tail -12 "$LOGD/train_$TS.log" 2>/dev/null
echo "[launch] gpu:"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "[launch] LOGFILE=$LOGD/train_$TS.log"
