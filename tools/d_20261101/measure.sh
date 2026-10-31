#!/usr/bin/env bash
# Measure one D arm with a full nvidia-smi trace alongside the training log.
#
# We record nvidia-smi rather than trusting torch's max_memory_allocated(): on this
# stack that counter reported ~20 GiB for a run the card only ever held 7.6 GiB of,
# so it is not a usable ceiling. The nvidia-smi peak is the number that decides what
# fits on a 24 GiB card.
set -u
ARM="${1:?usage: measure.sh ARM [STEPS]}"
STEPS="${2:-40}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT" || exit 1
OUT="assets/d_20261101/preflight"
mkdir -p "$OUT"
PY=/opt/conda/envs/cu121/bin/python
GPU_LOG="$OUT/nvidia_smi_${ARM}.csv"
RUN_LOG="$OUT/run_${ARM}.log"
rm -f "$GPU_LOG" "$RUN_LOG"

nvidia-smi -lms 500 \
    --query-gpu=timestamp,index,name,memory.used,power.draw,utilization.gpu \
    --format=csv,noheader > "$GPU_LOG" 2>&1 &
SMI=$!
sleep 2

TORCHINDUCTOR_CACHE_DIR="$ROOT/assets/d_20261101/inductor_cache" \
    "$PY" -u -m tools.style_matrix.runtime.train \
    --spec "assets/d_20261101/specs/${ARM}.json" --stop "$STEPS" \
    --run "$OUT/${ARM}/run" > "$RUN_LOG" 2>&1
RC=$?

sleep 2
kill "$SMI" 2>/dev/null
wait "$SMI" 2>/dev/null

PEAK=$(awk -F', ' 'NF>=4 {gsub(/ MiB/,"",$4); if ($4+0>m) m=$4+0} END {print m+0}' "$GPU_LOG")
echo "=== $ARM rc=$RC  nvidia-smi peak = ${PEAK} MiB  ($(wc -l < "$GPU_LOG") samples)"
echo "--- last records (step_s is cumulative, so read the deltas):"
grep -o "{\"step\".*}' "$RUN_LOG" | tail -4
echo "--- logs: $GPU_LOG  $RUN_LOG"
