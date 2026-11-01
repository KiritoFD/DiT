#!/bin/bash
# GAR post-training driver: verify the frozen-K contract, then eval each saved decoder.
#
# The whole point of a GAR run is that the generator stays K exactly, so the audit runs
# FIRST on every checkpoint: if the generator moved, the run is not a decoder-only
# experiment and no metric below it means anything.
set -uo pipefail
cd /root/Workspace/xy/DiT || exit 1

P=/opt/conda/envs/cu121/bin/python
ROOT=assets/d_20261101
RUN=$ROOT/runs/GAR
EVAL=$ROOT/eval
LOGS=$ROOT/logs
STATE=$ROOT/state
PANEL=assets/u100_pretrain_20261024/validation_val100.csv
mkdir -p "$EVAL" "$LOGS" "$STATE"

log() { echo "[$(date +%H:%M:%S)] $*"; }

STEPS=${*:-"100 200 336 400 500 600 672"}

for step in $STEPS; do
  ck=$(printf '%s/checkpoints/%07d.pt' "$RUN" "$step")
  [ -f "$ck" ] || { log "step $step: no checkpoint at $ck, skip"; continue; }

  # ---- gate 1: frozen-K audit (aborts the arm if the generator changed) ----
  if [ ! -f "$STATE/GAR_verify_$step.done" ]; then
    log "step $step: frozen-K audit"
    "$P" tools/d_20261101/verify_d1b.py "$ck" > "$LOGS/verify_GAR_$step.log" 2>&1 \
      || { log "step $step: FROZEN-K AUDIT FAILED (see $LOGS/verify_GAR_$step.log)"; exit 1; }
    grep -E "IDENTICAL|CHANGED|norm" "$LOGS/verify_GAR_$step.log" | sed 's/^/    /'
    date > "$STATE/GAR_verify_$step.done"
  fi

  # ---- gate 2: panel eval with the fine-tuned decoder ----
  tag="GAR_s$(printf '%06d' "$step")"
  if [ -f "$EVAL/$tag/eval.json" ]; then log "step $step: eval exists, skip"; continue; fi
  log "step $step: eval -> $tag"
  "$P" -u -m tools.f_20261101.eval_arm --checkpoint "$ck" --tag "$tag" \
    --panel "$PANEL" --device cuda --out "$EVAL" --use-ckpt-decoder \
    > "$LOGS/eval_$tag.log" 2>&1 \
    || { log "step $step: eval FAILED (see $LOGS/eval_$tag.log)"; exit 1; }
  "$P" -c "
import json,sys
d=json.load(open('$EVAL/$tag/eval.json'))
print('    ', {k:v for k,v in d.items() if isinstance(v,(int,float))})" 2>/dev/null
done

log "GAR driver finished"
