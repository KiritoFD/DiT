#!/bin/bash
# 依次跑各 compile 模式, 结果汇总到 /tmp/stepbench.log
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOG=/tmp/stepbench.log
: > "$LOG"
for m in none default max-autotune reduce-overhead; do
    echo "[run] mode=$m  $(date +%H:%M:%S)" >> "$LOG"
    timeout 1800 "$PY" -u _sync_work/step_bench.py "$m" 2>&1 | grep -E "^RESULT" >> "$LOG"
    echo "[done] mode=$m  $(date +%H:%M:%S)" >> "$LOG"
done
echo "[ALL DONE] $(date +%H:%M:%S)" >> "$LOG"
