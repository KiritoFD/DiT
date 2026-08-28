#!/bin/bash
# Serial 1k-step validation: s14(WS-DDPM) -> s15(WS-Flow) -> s16(S-DDPM) -> s17(S-Flow)
# Each run: starts eval daemon for that series, runs train.py to max_steps=1000,
# waits for eval_auto_0001000.json (or timeout), then kills daemon and proceeds.
#
# Usage:  bash run_4tests_serial.sh
# Logs:   /tmp/s14_test.log, /tmp/s15_test.log, /tmp/s16_test.log, /tmp/s17_test.log
#         /tmp/s14_eval_daemon.log, etc.
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
PY=/opt/conda/bin/python

run_one () {
    local CFG=$1          # e.g. s14_ws_ddpm_test.json
    local RESULTS_DIR=$2  # e.g. 5script/results/s14_ws_ddpm
    local TAG=$3          # e.g. s14
    local DAEMON_PID=""

    echo "============================================"
    echo "[$(date '+%H:%M:%S')] === START $TAG ($CFG) ==="
    echo "============================================"

    # Clean any stale state
    pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true
    rm -rf "$RESULTS_DIR" 2>/dev/null || true
    sleep 1

    # Start eval daemon (CPU, separate process) for this series
    nohup $PY eval_metrics_daemon.py "$RESULTS_DIR" > "/tmp/${TAG}_eval_daemon.log" 2>&1 &
    DAEMON_PID=$!
    echo "[$TAG] eval daemon pid=$DAEMON_PID"

    # Start training
    nohup $PY train.py --config "$CFG" > "/tmp/${TAG}_test.log" 2>&1 &
    TRAIN_PID=$!
    echo "[$TAG] train pid=$TRAIN_PID"

    # Wait for training to finish (max_steps=1000 → ~6-8 min) or die
    # Then wait up to 5 min for eval_auto_0001000.json to appear (daemon needs to process)
    local DEADLINE=$(( $(date +%s) + 1800 ))  # 30 min hard cap per run
    while kill -0 $TRAIN_PID 2>/dev/null; do
        if [ $(date +%s) -gt $DEADLINE ]; then
            echo "[$TAG] TIMEOUT (train still running after 30min), killing"
            kill -9 $TRAIN_PID 2>/dev/null || true
            break
        fi
        sleep 15
    done
    wait $TRAIN_PID 2>/dev/null || true
    echo "[$TAG] train process exited"

    # Wait for eval_auto_0001000.json (daemon processes pending marker)
    local EVAL_WAIT=$(( $(date +%s) + 300 ))  # 5 min for CPU eval
    local EVAL_FILE=""
    while [ $(date +%s) -lt $EVAL_WAIT ]; do
        EVAL_FILE=$(ls "$RESULTS_DIR"/*/checkpoints/eval_auto_0001000.json 2>/dev/null | head -1 || true)
        if [ -n "$EVAL_FILE" ]; then
            echo "[$TAG] eval_auto_0001000.json found: $EVAL_FILE"
            cat "$EVAL_FILE"
            break
        fi
        sleep 10
    done
    if [ -z "$EVAL_FILE" ]; then
        echo "[$TAG] WARNING: eval_auto_0001000.json not found after 5min"
        ls -la "$RESULTS_DIR"/*/checkpoints/ 2>/dev/null | head -20
    fi

    # Kill daemon + free GPU
    kill $DAEMON_PID 2>/dev/null || true
    pkill -f "eval_metrics_daemon.py $RESULTS_DIR" 2>/dev/null || true
    sleep 2
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
    echo "[$(date '+%H:%M:%S')] === END $TAG ==="
    echo ""
}

run_one "s14_ws_ddpm_test.json" "5script/results/s14_ws_ddpm" "s14"
run_one "s17_s_flow_test.json"  "5script/results/s17_s_flow"  "s17"
run_one "s15_ws_flow_test.json" "5script/results/s15_ws_flow" "s15"
run_one "s16_s_ddpm_test.json"  "5script/results/s16_s_ddpm"  "s16"

echo "============================================"
echo "[$(date '+%H:%M:%S')] ALL 4 TESTS COMPLETE"
echo "============================================"
for t in s14 s15 s16 s17; do
    echo "--- $t eval_auto_0001000 ---"
    cat 5script/results/${t}_*/*/checkpoints/eval_auto_0001000.json 2>/dev/null || echo "  (not found)"
done