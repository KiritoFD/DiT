#!/bin/bash
# Serial full runs: s15(WS-Flow) -> s17(S-Flow) -> s14(WS-DDPM) -> s16(S-DDPM)
# Order: flow first, then ddpm. Each 200k steps, early stop enabled.
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/opt/conda/bin/python

run_full () {
    local CFG=$1
    local RESULTS_DIR=$2
    local TAG=$3

    echo "============================================"
    echo "[$(date '+%H:%M:%S')] === START $TAG ($CFG) ==="
    echo "============================================"

    pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true
    sleep 1

    nohup $PY eval_metrics_daemon.py "$RESULTS_DIR" > "/tmp/${TAG}_eval_daemon.log" 2>&1 &
    local DAEMON_PID=$!
    echo "[$TAG] eval daemon pid=$DAEMON_PID"

    nohup $PY train.py --config "$CFG" > "/tmp/${TAG}_full.log" 2>&1 &
    local TRAIN_PID=$!
    echo "[$TAG] train pid=$TRAIN_PID"

    wait $TRAIN_PID 2>/dev/null
    echo "[$TAG] train process exited"

    sleep 30
    kill $DAEMON_PID 2>/dev/null || true
    pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true

    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo "[$(date '+%H:%M:%S')] === END $TAG ==="
    echo ""
}

run_full "s15_ws_flow.json" "5script/results/s15_ws_flow" "s15"
run_full "s17_s_flow.json" "5script/results/s17_s_flow" "s17"
run_full "s14_ws_ddpm.json" "5script/results/s14_ws_ddpm" "s14"
run_full "s16_s_ddpm.json" "5script/results/s16_s_ddpm" "s16"

echo "============================================"
echo "[$(date '+%H:%M:%S')] ALL 4 FULL RUNS COMPLETE"
echo "============================================"