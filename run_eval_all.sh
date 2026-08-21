#!/bin/bash
set -u
# Critical: remote locale must be UTF-8 so Chinese paths load correctly.
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
cd /root/Workspace/xy/DiT
export XFORMERS_DISABLED=1
PY=/opt/conda/bin/python
CKPT_DIR=results/full_3cond/001-DiT-3Cond-XL-2/checkpoints
OUT_DIR=eval_pngs/test
LOG=eval_all_test.log
mkdir -p "$OUT_DIR"
for ck in $(ls "$CKPT_DIR"/*.pt | sort); do
    step=$(basename "$ck" .pt)
    out="$OUT_DIR/eval_step_${step}.png"
    if [ -f "$out" ]; then
        echo "[$(date +%H:%M:%S)] step $step already done, skip" | tee -a "$LOG"
        continue
    fi
    echo "[$(date +%H:%M:%S)] ===== eval(step $step) on test.csv =====" | tee -a "$LOG"
    "$PY" eval_full_3cond.py "$ck" "$out" 8 150 0 test.csv >> "$LOG" 2>&1
    echo "[$(date +%H:%M:%S)] step $step done (rc $?)" | tee -a "$LOG"
done
echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$LOG"
