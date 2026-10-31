#!/bin/bash
# D-plan serial driver: D1 -> D2 -> D3, then eval.
#
# Each arm's horizon is derived from its measured throughput (see
# assets/d_20261101/preflight/), targeting ~2 h per arm rather than a guessed step
# count:
#   D1 2.83 s/step -> 2500 steps   (~2.0 h)   peak 19.8 GiB
#   D2 2.60 s/step -> 3000 steps   (~2.2 h)   peak 21.2 GiB
#   D3 0.64 s/step -> 11000 steps  (~2.0 h)   peak 22.4 GiB
#
# Stages are idempotent through state/ markers so a restart resumes.
#
#   tmux new -d -s d 'bash tools/d_20261101/run_d.sh > assets/d_20261101/logs/driver.log 2>&1'
set -uo pipefail
cd /root/Workspace/xy/DiT || exit 1

P=/opt/conda/envs/cu121/bin/python
ROOT=assets/d_20261101
SPECS=$ROOT/specs
RUNS=$ROOT/execution/runs
EVAL=$ROOT/eval
STATE=$ROOT/state
LOGS=$ROOT/logs
PANEL=assets/u100_pretrain_20261024/validation_val100.csv
K_A=assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt
mkdir -p "$SPECS" "$RUNS" "$EVAL" "$STATE" "$LOGS"

ONLY=${1:-}
log() { echo "[$(date +%H:%M:%S)] $*"; }
done_already() { [ -f "$STATE/$1.done" ]; }
mark() { date > "$STATE/$1.done"; log "stage $1 DONE"; }
should_run() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

train_arm() {   # train_arm <arm> <steps>
  local arm=$1 steps=$2
  local spec=$SPECS/$arm.json run=$RUNS/$arm
  if [ ! -f "$spec" ]; then log "$arm: missing spec $spec"; return 1; fi
  log "$arm: training $steps steps -> $run"
  TORCHINDUCTOR_CACHE_DIR="$ROOT/inductor_cache" \
    "$P" -u -m tools.style_matrix.runtime.train \
      --spec "$spec" --stop "$steps" --run "$run" \
      > "$LOGS/train_$arm.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    log "$arm: TRAIN FAILED rc=$rc (see $LOGS/train_$arm.log)"; return 1
  fi
  if [ ! -f "$run/checkpoints/$(printf '%07d' "$steps").pt" ]; then
    log "$arm: no final checkpoint at step $steps"; return 1
  fi
  log "$arm: done, $(ls "$run/checkpoints" | wc -l) checkpoints"
}

run_eval() {   # run_eval <tag> <checkpoint> [extra eval_arm args...]
  local tag=$1 ck=$2; shift 2
  if [ -f "$EVAL/$tag/eval.json" ]; then
    log "  skip eval $tag (eval.json exists)"; return 0
  fi
  "$P" -u -m tools.f_20261101.eval_arm --checkpoint "$ck" --tag "$tag" \
    --panel "$PANEL" --device cuda --out "$EVAL" "$@" > "$LOGS/eval_$tag.log" 2>&1 \
    || { log "  eval $tag FAILED (see $LOGS/eval_$tag.log)"; return 1; }
  log "  eval $tag ok"
}

# ---------------------------------------------------------------- preflight gate
if should_run s0_ready && ! done_already s0_ready; then
  log "s0: verifying inputs"
  for f in "$SPECS/D1.json" "$SPECS/D2.json" "$SPECS/D3.json" \
           "$ROOT/calibration_D1.json" "$ROOT/calibration_D2.json" \
           "$ROOT/stage_p2_hard.csv" "$K_A"; do
    [ -f "$f" ] || { log "s0: MISSING $f"; exit 1; }
  done
  # The hard-char CSV must keep the source schema or D3 silently trains on garbage.
  head -1 "$ROOT/stage_p2_hard.csv" | grep -q 'img_id,image_path,calligrapher' \
    || { log "s0: stage_p2_hard.csv header looks wrong"; exit 1; }
  log "s0: inputs ok ($(wc -l < "$ROOT/stage_p2_hard.csv") rows in hard csv)"
  mark s0_ready
fi

# ---------------------------------------------------------------- s1: D1
if should_run s1_d1 && ! done_already s1_d1; then
  log "s1: D1 decoder-only (G2J4, U-Net frozen, FM skipped)"
  train_arm D1 2500 && mark s1_d1 || exit 1
fi

# ---------------------------------------------------------------- s2: D2
if should_run s2_d2 && ! done_already s2_d2; then
  log "s2: D2 U-Net + decoder joint (J3, FM + weak image)"
  train_arm D2 3000 && mark s2_d2 || exit 1
fi

# ---------------------------------------------------------------- s3: D3
if should_run s3_d3 && ! done_already s3_d3; then
  log "s3: D3 hard-character FM (plain, oversampled train csv)"
  train_arm D3 11000 && mark s3_d3 || exit 1
fi

# ---------------------------------------------------------------- s4: eval
if should_run s4_eval && ! done_already s4_eval; then
  log "s4: evaluating D1/D2/D3 against the val100 panel"
  D1=$RUNS/D1/checkpoints/0002500.pt
  D2=$RUNS/D2/checkpoints/0003000.pt
  D3=$RUNS/D3/checkpoints/0011000.pt
  run_eval D1_2500 "$D1" --use-ckpt-decoder
  run_eval D2_3000 "$D2" --use-ckpt-decoder
  run_eval D3_11000 "$D3"
  # Same U-Net as D3, D1's decoder: tests whether the two fixes compose without
  # any extra training, since both sit on the same VAE latent contract.
  run_eval D3_11000_plus_D1dec "$D3" --decoder-from "$D1"
  mark s4_eval
fi

log "all stages finished"
