#!/usr/bin/env bash
# Explicit launch only; no killing or attaching to existing training jobs.
# Usage: bash tools/analysis/run_strong_style_serial.sh [control A30 A60 A100 B25 B50 C D]
# Fresh v84A200k EMA arms: all10k, save/evaluate at5k and10k.
# Gradient ratios are measured on active auxiliary steps; C runs on 1/4 steps.
# Max coefficient=5; actual parameter-gradient ratios are recalibrated every10 steps.
# Ten consecutive underpowered probes abort rather than silently waste the budget.
# Logs report realized ratios at probe steps. Change configs only after checking them.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=.
PY=${PYTHON_BIN:-/opt/conda/envs/cu121/bin/python}
mkdir -p logs
# Protect against a second copy of this queue and against an occupied GPU.
exec 9>logs/strong_style_queue.lock
flock -n 9 || { echo 'Another strong-style queue is running'; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU already occupied; wait for existing jobs to finish before launching'; exit 1
fi
"$PY" tools/analysis/prepare_strong_style.py
"$PY" tools/analysis/calibrate_strong_style.py --device cpu
arms=("$@")
if ((${#arms[@]} == 0)); then arms=(control A30 A60 A100 B25 B50 C D); fi
for arm in "${arms[@]}"; do
  case "$arm" in control|A30|A60|A100|B25|B50|C|D) ;; *) echo "Unknown arm: $arm"; exit 1;; esac
  log="logs/strong_style_${arm}_$(date +%Y%m%d-%H%M%S).log"
  echo "START $arm $(date -Is) $log"
  MASTER_PORT=12781 "$PY" -m src.train.train --config "src/train/configs/v86_strong_${arm}.json" --num-workers 4 > "$log" 2>&1
  steps=10000
  grep -q "Reached max_steps=$steps" "$log" || { echo "Incomplete arm: $arm"; exit 1; }
  echo "DONE $arm $(date -Is)"
done
# No automatic winner/40k extension: independent content+style evaluation is required.
