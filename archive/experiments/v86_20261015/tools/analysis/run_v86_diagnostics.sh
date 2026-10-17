#!/usr/bin/env bash
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=.
PY=/opt/conda/envs/cu121/bin/python
"$PY" tools/analysis/prepare_v86_diagnostic_runs.py
port=12661
for arm in control writer brush; do
  log="logs/v86_diag_${arm}_20261014.log"
  echo "START $arm $(date -Is)"
  MASTER_PORT=$port "$PY" -m src.train.train --config "src/train/configs/v86_diag_${arm}_5k.json" --num-workers 4 > "$log" 2>&1
  grep -q 'Reached max_steps=5000' "$log"
  echo "DONE $arm $(date -Is)"
  port=$((port+1))
done
"$PY" tools/analysis/summarize_v86_diagnostics.py
