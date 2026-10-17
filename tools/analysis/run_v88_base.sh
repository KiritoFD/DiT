#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
PY=${PYTHON_BIN:-/opt/conda/envs/cu121/bin/python}
CONFIG=${1:-src/train/configs/v88_writer_glyph_base.json}
STAMP=$(date +%Y%m%d-%H%M%S)
mkdir -p logs
exec 9>logs/strong_style_queue.lock
flock -n 9 || { echo 'GPU queue lock occupied'; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU occupied; refusing concurrent training'; exit 1
fi
[[ -f "$CONFIG" ]] || { echo 'Prepare the plan with python tools/analysis/prepare_v88.py first'; exit 1; }
STEPS=$("$PY" -c 'import json,sys; n=json.load(open(sys.argv[1]))["max_steps"]; assert isinstance(n,int) and n>0; print(n)' "$CONFIG")
TRAIN_LOG="logs/v88_base_${STAMP}.log"
echo "CONFIG=$CONFIG STEPS=$STEPS TRAIN_LOG=$TRAIN_LOG"
# Do not regenerate/overwrite a reviewed config when starting a run.
"$PY" tools/analysis/smoke_v88.py --config "$CONFIG" --out "assets/v88_preflight/$STAMP" > "logs/v88_preflight_${STAMP}.log" 2>&1
echo "PREFLIGHT PASSED $(date -Is)"
MASTER_PORT=${MASTER_PORT:-12821} "$PY" -m src.train.train --config "$CONFIG" --num-workers 4 > "$TRAIN_LOG" 2>&1
grep -q "Reached max_steps=$STEPS" "$TRAIN_LOG"
echo "TRAIN DONE $(date -Is)"
