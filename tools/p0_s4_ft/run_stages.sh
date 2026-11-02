#!/bin/bash
# Serial P0 -> S4 driver: S4-W (3k) -> S4-FM (35k) -> S4-Tail (<=10k).
#
# Stage 1 is the only stage that performs the partial transfer from P0. Stages 2-3 chain
# by --resume so R2/I1 weights survive the boundary (the plan's check #4).
#
# Two gates before GPU time, in increasing cost order:
#   1. migration dry-run -- backbone + R2 must be byte-identical to P0's EMA, and only the
#      writer rows plus the fresh I1 may differ. Cheap, CPU-only, and it is the check that
#      makes "we inherited P0" a verified claim rather than an assumption.
#   2. init forward -- under the null writer the first samples must not be destroyed by
#      the freshly initialised I1 (the plan's check #2).
#
# Stages are resumable: each stage writes a .done marker, and re-running the script skips
# completed stages rather than restarting over them.
set -uo pipefail
cd /root/Workspace/xy/DiT || exit 1

P=/opt/conda/envs/cu121/bin/python
R=assets/p0_s4_ft
SPECS=$R/specs
RUNS=$R/runs
LOGS=$R/logs
STATE=$R/state
mkdir -p "$LOGS" "$STATE" "$RUNS"

log() { echo "[$(date +%H:%M:%S)] $*"; }
done_already() { [ -f "$STATE/$1.done" ]; }
mark() { date > "$STATE/$1.done"; log "stage $1 DONE"; }

# ---- gate 1: migration must be exact ---------------------------------------
log "gate 1: migration dry-run"
if ! PYTHONPATH=. timeout 1200 $P tools/p0_s4_ft/migrate_dryrun.py --device cpu \
        > "$LOGS/gate1_migration.log" 2>&1; then
  log "GATE 1 FAILED -- see $LOGS/gate1_migration.log"
  tail -20 "$LOGS/gate1_migration.log"
  exit 1
fi
if ! tail -3 "$LOGS/gate1_migration.log" | grep -q "OK: backbone + R2 byte-identical"; then
  log "GATE 1 FAILED (migration not exact) -- see $LOGS/gate1_migration.log"
  tail -20 "$LOGS/gate1_migration.log"
  exit 1
fi
grep -E "byte-identical|totals:" "$LOGS/gate1_migration.log" | tail -4
log "gate 1 PASSED"

# ---- gate 2: the fresh I1 must not wreck the null-writer output -------------
log "gate 2: init forward under null writer"
if ! PYTHONPATH=. timeout 1500 $P tools/p0_s4_ft/check_init_forward.py \
        --spec "$SPECS/S4-W.json" --out "$R/init_check.json" \
        > "$LOGS/gate2_init.log" 2>&1; then
  log "GATE 2 FAILED -- see $LOGS/gate2_init.log"
  tail -30 "$LOGS/gate2_init.log"
  exit 1
fi
tail -12 "$LOGS/gate2_init.log"
log "gate 2 PASSED"

# ---- serial stages ----------------------------------------------------------
run_stage() {
  local name=$1; local prev=$2
  if done_already "$name"; then log "skip $name (already done)"; return 0; fi

  local spec="$SPECS/$name.json"
  local run="$RUNS/$name"
  local steps
  steps=$($P -c "import json;print(json.load(open('$spec'))['config']['max_steps'])")
  local resume=""
  if [ -n "$prev" ] && [ -f "$prev" ]; then
    resume="--resume $prev"
    log "$name: resuming from $prev"
  else
    log "$name: fresh start from P0 partial transfer"
  fi

  log "$name: $steps steps -> $run"
  if ! PYTHONPATH=. timeout 86400 $P -m tools.style_matrix.runtime.train \
          --spec "$spec" --run "$run" $resume --stop "$steps" \
          > "$LOGS/$name.log" 2>&1; then
    log "$name FAILED -- see $LOGS/$name.log"
    tail -40 "$LOGS/$name.log"
    return 1
  fi
  local ck
  ck=$(printf "%s/checkpoints/%07d.pt" "$run" "$steps")
  if [ ! -f "$ck" ]; then
    log "$name finished but $ck is missing -- refusing to advance"
    tail -20 "$LOGS/$name.log"
    return 1
  fi
  mark "$name"
  log "$name: checkpoint $ck"
  LAST_CKPT=$ck
  return 0
}

LAST_CKPT=""
run_stage S4-W "" || exit 1
W_CKPT=$LAST_CKPT
run_stage S4-FM "$W_CKPT" || exit 1
FM_CKPT=$LAST_CKPT
run_stage S4-Tail "$FM_CKPT" || exit 1

log "ALL STAGES DONE"
log "final checkpoint: $LAST_CKPT"
