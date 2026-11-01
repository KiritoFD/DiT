#!/bin/bash
# GARU driver: unfrozen-generator arm. Runs AFTER GARL so the two never share the card.
#
# Two gates before any GPU time is spent, in increasing cost order:
#   1. CPU pre-flight  -- does the spec really start from K, with the intended trainable
#      surface? This is the check that would have caught the D1/D2/D3 random-style-branch
#      bug, which is the reason this arm exists at all.
#   2. GPU routing probe -- does a real generator gradient appear? If it stayed zero the
#      arm would silently BE the decoder-only GAR arm.
#
# There is deliberately NO frozen-K audit here: this arm is not decoder-only, so the
# generator is SUPPOSED to move. Acceptance is a paired comparison against K.
set -uo pipefail
cd /root/Workspace/xy/DiT || exit 1

P=/opt/conda/envs/cu121/bin/python
ROOT=assets/d_20261101
SPEC=$ROOT/specs/GARU.json
RUN=$ROOT/runs/GARU
LOGS=$ROOT/logs
STATE=$ROOT/state
EVAL=$ROOT/eval
PANEL=assets/u100_pretrain_20261024/validation_val100.csv
STEPS=${1:-2500}
mkdir -p "$LOGS" "$STATE" "$EVAL" "$RUN"

log() { echo "[$(date +%H:%M:%S)] $*"; }
done_already() { [ -f "$STATE/$1.done" ]; }
mark() { date > "$STATE/$1.done"; log "stage $1 DONE"; }

# ---------------------------------------------------------------- gate 0: wait for GARL
if ! done_already garu_card_free; then
  log "waiting for the GARL run to release the GPU"
  while tmux has-session -t garl 2>/dev/null; do sleep 60; done
  sleep 20
  nvidia-smi --query-gpu=memory.used --format=csv,noheader
  mark garu_card_free
fi

# ---------------------------------------------------------------- gate 1: spec pre-flight
if ! done_already garu_preflight; then
  log "gate 1: CPU spec pre-flight (must start exactly at K)"
  PYTHONPATH=. CUDA_VISIBLE_DEVICES="" "$P" tools/d_20261101/check_spec_init.py "$SPEC" \
    > "$LOGS/garu_preflight.log" 2>&1 \
    || { log "PRE-FLIGHT FAILED (see $LOGS/garu_preflight.log)"; tail -20 "$LOGS/garu_preflight.log"; exit 1; }
  grep -E "as-built|trainable|OK:|FAIL" "$LOGS/garu_preflight.log" | sed 's/^/    /'
  mark garu_preflight
fi

# ---------------------------------------------------------------- gate 2: routing probe
if ! done_already garu_routing; then
  log "gate 2: GPU gradient-routing probe (GARU must move the generator)"
  PYTHONPATH=. "$P" tools/d_20261101/probe_grad_routing.py --cuda \
    > "$LOGS/garu_routing.log" 2>&1 \
    || { log "ROUTING PROBE FAILED (see $LOGS/garu_routing.log)"; tail -20 "$LOGS/garu_routing.log"; exit 1; }
  grep -E "generator grad|OK:|FAIL" "$LOGS/garu_routing.log" | sed 's/^/    /'
  mark garu_routing
fi

# ---------------------------------------------------------------- train
if ! done_already garu_train; then
  log "training GARU for $STEPS steps"
  TORCHINDUCTOR_CACHE_DIR="$ROOT/inductor_cache" PYTHONPATH=. \
    "$P" -u -m tools.style_matrix.runtime.train \
      --spec "$SPEC" --stop "$STEPS" --run "$RUN" \
      > "$LOGS/train_GARU.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { log "TRAIN FAILED rc=$rc (see $LOGS/train_GARU.log)"; exit 1; }
  mark garu_train
fi

# ---------------------------------------------------------------- eval the milestones
for step in 250 500 1000 1500 2000 2500; do
  [ "$step" -le "$STEPS" ] || continue
  ck=$(printf '%s/checkpoints/%07d.pt' "$RUN" "$step")
  [ -f "$ck" ] || { log "step $step: no checkpoint, skip"; continue; }
  tag="GARU_s$(printf '%06d' "$step")"
  if [ -f "$EVAL/$tag/eval.json" ]; then log "step $step: eval exists, skip"; continue; fi
  log "step $step: eval -> $tag"
  "$P" -u -m tools.f_20261101.eval_arm --checkpoint "$ck" --tag "$tag" \
    --panel "$PANEL" --device cuda --out "$EVAL" --use-ckpt-decoder \
    > "$LOGS/eval_$tag.log" 2>&1 \
    || { log "eval $tag FAILED (see $LOGS/eval_$tag.log)"; exit 1; }
  "$P" tools/d_20261101/gar_table.py >/dev/null 2>&1 || true
done

log "GARU driver finished"
