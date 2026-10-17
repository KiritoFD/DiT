#!/usr/bin/env bash
# Explicit phase launcher: no automatic5k->50k->200k promotion.
set -euo pipefail
cd "$(dirname "$0")/../.."
MODE=${1:-strong}
PHASE=${2:-5k}
RESUME=${3:-}
case "$MODE" in strong|mild) ;; *) exit 2;; esac
case "$PHASE" in 5k|50k|200k) ;; *) exit 2;; esac
PY=${PYTHON_BIN:-/opt/conda/envs/cu121/bin/python}
CONFIG="src/train/configs/v90_20l/${MODE}_${PHASE}.json"
mkdir -p logs
exec 9>logs/strong_style_queue.lock
flock -n 9 || { echo 'GPU queue occupied; P1 left untouched'; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU occupied; refusing concurrent base training'; exit 1
fi
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 PYTHONPATH=.
STAMP=$(date +%Y%m%d-%H%M%S)
RESULTS="assets/results/v90_20l_${MODE}_${PHASE}_${STAMP}"
if [[ "$PHASE" == 5k ]]; then
  [[ -z "$RESUME" ]] || { echo '5k starts a fresh20L backbone'; exit 2; }
  "$PY" tools/analysis/validate_v90_launch.py --config "$CONFIG"
  MASTER_PORT=12922 "$PY" -u -m src.train.train --config "$CONFIG" --results-dir "$RESULTS" > "logs/v90_20l_${MODE}_${PHASE}_${STAMP}.log" 2>&1
else
  [[ -n "$RESUME" ]] || { echo 'Review prior phase, then explicitly provide its checkpoint'; exit 2; }
  "$PY" tools/analysis/validate_v90_launch.py --config "$CONFIG" --resume "$RESUME"
  MASTER_PORT=12923 "$PY" -u -m src.train.train --config "$CONFIG" --results-dir "$RESULTS" --resume-full "$RESUME" > "logs/v90_20l_${MODE}_${PHASE}_${STAMP}.log" 2>&1
fi
case "$PHASE" in 5k) STOP=5000;;50k) STOP=50000;;200k) STOP=200000;;esac
"$PY" tools/analysis/eval_v90_20l_phase.py --results "$RESULTS" --step "$STOP" --cache assets/std_shape_cache_256.pt > "logs/v90_20l_${MODE}_${PHASE}_${STAMP}_acceptance.log" 2>&1
