#!/bin/bash
# Serial full runs: s15(WS-Flow 200k) -> s17(S-Flow 200k)
# Each run: starts eval daemon, runs train.py to max_steps=200000,
# with early stopping. Daemon keeps running alongside training.
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

    # Start eval daemon (long-lived, stays for whole run)
    nohup $PY eval_metrics_daemon.py "$RESULTS_DIR" > "/tmp/${TAG}_eval_daemon.log" 2>&1 &
    local DAEMON_PID=$!
    echo "[$TAG] eval daemon pid=$DAEMON_PID"

    # Start training
    nohup $PY train.py --config "$CFG" > "/tmp/${TAG}_full.log" 2>&1 &
    local TRAIN_PID=$!
    echo "[$TAG] train pid=$TRAIN_PID"

    # Wait for training to finish (max_steps=200k or early stop)
    wait $TRAIN_PID 2>/dev/null
    echo "[$TAG] train process exited"

    # Let daemon finish any pending eval
    sleep 30
    kill $DAEMON_PID 2>/dev/null || true
    pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true

    echo "[$(date '+%H:%M:%S')] === END $TAG ==="
    echo ""
}

run_full "s15_ws_flow.json" "5script/results/s15_ws_flow" "s15"
run_full "s17_s_flow.json" "5script/results/s17_s_flow" "s17"

echo "============================================"
echo "[$(date '+%H:%M:%S')] ALL FLOW RUNS COMPLETE"
echo "============================================"