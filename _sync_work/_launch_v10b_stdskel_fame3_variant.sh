#!/bin/bash
# 参数化拉起一个 fame3 变体训练 + cpu_eval daemon
# 用法: _launch_v10b_stdskel_fame3_variant.sh <tmux名> <config名> <results目录名> [resume_ckpt路径] [resume_lr]
# 日志: /root/Workspace/xy/DiT/logs/<results目录名>/train_<ts>.log + eval_<ts>.log (带时间戳, 不互相覆盖)
set -u
NAME=${1:?tmux name}
CFG=${2:?config name}
RES=${3:?results dir}
RESUME=${4:-}
RESUME_LR=${5:-}
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/$RES
TS=$(date +%Y%m%d-%H%M%S)
mkdir -p "$LOGD"
mkdir -p /root/Workspace/xy/DiT/assets/registry/configs
cp /root/Workspace/xy/DiT/src/train/configs/$CFG.json /root/Workspace/xy/DiT/assets/registry/configs/$CFG.json 2>/dev/null || true

tmux kill-session -t "$NAME" 2>/dev/null
pkill -f "train.py --config src/train/configs/" 2>/dev/null
sleep 3
RESUME_ARGS=""
if [ -n "$RESUME" ]; then
    RESUME_ARGS="--resume-full $RESUME --fresh-scheduler 1"
    if [ -n "$RESUME_LR" ]; then
        RESUME_ARGS="$RESUME_ARGS --resume-lr $RESUME_LR"
        echo "[launch] override base LR -> $RESUME_LR"
    fi
    echo "[launch] resume from $RESUME (fresh scheduler)"
fi
tmux new-session -d -s "$NAME" \
  "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=/root/Workspace/xy/DiT; $PY -u src/train/train.py --config src/train/configs/$CFG.json $RESUME_ARGS 2>&1 | tee $LOGD/train_$TS.log"
# eval: GPU in-mem 循环 (seen 每 ckpt / strict 每 10k, 均在 cuda). CPU 不再跑评测。
tmux kill-session -t "gpueval_$NAME" 2>/dev/null
tmux new-session -d -s "gpueval_$NAME" \
  "cd /root/Workspace/xy/DiT && PYTHONPATH=/root/Workspace/xy/DiT $PY -u tools/eval/gpu_eval_loop.py --results-dir assets/results/$RES --device cuda --poll 30 --strict-every 10000 > $LOGD/gpu_eval.log 2>&1"
sleep 5
echo "tmux:"; tmux ls 2>/dev/null | grep -E "$NAME|gpueval_"
tail -3 $LOGD/train_$TS.log 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
