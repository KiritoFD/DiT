#!/usr/bin/env bash
# s20 预训练（mid_common, S/2, v2 骨干, flow free-lunch）
#
#   bash run_s20_midcommon.sh
#
# 结构沿用 run_flow_full.sh：先起 CPU 指标 daemon，再起 train.py，最后收尾。
# 训练侧 in-process GPU eval 每 ckpt_every(2500) 步产出 PNG，
# daemon 计算 MSE/SSIM/skel_iou/LPIPS 并写 eval_auto_*.json 供早停读取。
set -u

cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# train.py 位于 src/train/，必须把仓库根加进 PYTHONPATH 才能 import src.*
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/bin/python

CFG="src/train/configs/s20_midcommon_s_flow_v2.json"
RESULTS_DIR="5script/results/s20_midcommon_s_flow_v2"
TAG="s20"

# daemon 在 src/eval/ 下（run_flow_full.sh 里写的是仓库根，路径是错的）
DAEMON="src/eval/eval_metrics_daemon.py"

mkdir -p "$RESULTS_DIR"

pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true
sleep 1

nohup $PY "$DAEMON" "$RESULTS_DIR" > "/tmp/${TAG}_eval_daemon.log" 2>&1 &
DAEMON_PID=$!
echo "[$(date '+%H:%M:%S')] eval daemon pid=$DAEMON_PID  ($DAEMON)"

nohup $PY src/train/train.py --config "$CFG" > "/tmp/${TAG}_train.log" 2>&1 &
TRAIN_PID=$!
echo "[$(date '+%H:%M:%S')] train pid=$TRAIN_PID"

echo "$TRAIN_PID" > "/tmp/${TAG}_train.pid"
echo "$DAEMON_PID" > "/tmp/${TAG}_daemon.pid"

wait $TRAIN_PID 2>/dev/null
echo "[$(date '+%H:%M:%S')] train process exited"

# 让 daemon 处理完最后一个 pending eval
sleep 30
kill $DAEMON_PID 2>/dev/null || true
pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true
echo "[$(date '+%H:%M:%S')] === s20 TRAINING COMPLETE ==="
