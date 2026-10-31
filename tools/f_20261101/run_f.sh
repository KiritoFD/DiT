#!/bin/bash
# F-plan serial driver (12 h budget). Every stage is idempotent: a marker file in
# state/ makes a completed stage skip, so re-running this script after an
# interruption resumes where it stopped.
#
#   bash tools/f_20261101/run_f.sh            # run all stages in order
#   bash tools/f_20261101/run_f.sh s4_f0      # run one stage (and its prerequisites
#                                             # are NOT run — use for debugging)
#
# Intended launch (survives logout):
#   tmux new -d -s f 'bash tools/f_20261101/run_f.sh > logs/f_driver.log 2>&1'
set -uo pipefail
cd /root/Workspace/xy/DiT || exit 1

P=/opt/conda/envs/cu121/bin/python
ROOT=assets/f_20261101
SPECS=$ROOT/specs
RUNS=$ROOT/runs
EVAL=$ROOT/eval
STATE=$ROOT/state
LOGS=$ROOT/logs
PANEL=assets/u100_pretrain_20261024/validation_val100.csv
SWAP_CSV=assets/p125_20261030/stage_p2.csv
K_A=assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt
K_P3A=assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt
mkdir -p "$SPECS" "$RUNS" "$EVAL" "$STATE" "$LOGS"

ONLY=${1:-}
log() { echo "[$(date +%H:%M:%S)] $*"; }
done_already() { [ -f "$STATE/$1.done" ]; }
mark() { date > "$STATE/$1.done"; log "stage $1 DONE"; }
should_run() { [ -z "$ONLY" ] || [ "$ONLY" = "$1" ]; }

# An eval stage is idempotent on its own output, so a driver restart never
# repeats an hour of sampling.
run_eval() {   # run_eval <tag> <checkpoint> [extra eval_arm args...]
  local tag=$1 ck=$2; shift 2
  if [ -f "$EVAL/$tag/eval.json" ]; then
    log "  skip eval $tag (eval.json exists)"; return 0
  fi
  $P -u -m tools.f_20261101.eval_arm --checkpoint "$ck" --tag "$tag" \
    --panel "$PANEL" --device cuda --out "$EVAL" "$@" > "$LOGS/eval_$tag.log" 2>&1 \
    || { log "  eval $tag FAILED (see $LOGS/eval_$tag.log)"; return 1; }
  log "  eval $tag ok"
}

run_swap() {   # run_swap <tag> <checkpoint> [extra t3 args...]
  local tag=$1 ck=$2; shift 2
  if [ -f "$EVAL/t3_$tag/t3_report.json" ]; then
    log "  skip swap $tag (report exists)"; return 0
  fi
  $P -u -m tools.diag.t3_writer_swap --checkpoint "$ck" --csv "$SWAP_CSV" \
    --tag "$tag" --device cuda --out "$EVAL" "$@" > "$LOGS/swap_$tag.log" 2>&1 \
    || { log "  swap $tag FAILED (see $LOGS/swap_$tag.log)"; return 1; }
  log "  swap $tag ok"
}

# ---------------------------------------------------------------- s0: pick K
if should_run s0_k_eval && ! done_already s0_k_eval; then
  log "s0: K selection — eval A@43429 vs P3A@20000"
  run_eval K_A "$K_A" --csv "$SWAP_CSV" --chars 20 --writers 10 --seeds 2 || exit 1
  run_eval K_P3A "$K_P3A" --csv "$SWAP_CSV" --chars 20 --writers 10 --seeds 2 || exit 1
  $P -u -m tools.f_20261101.pick pick-k \
    --evals "$EVAL/K_A/eval.json" "$EVAL/K_P3A/eval.json" \
    --out "$ROOT/k_choice.json" || exit 1
  mark s0_k_eval
fi
K_TAG=$($P -c "import json;print(json.load(open('$ROOT/k_choice.json'))['winner'])" 2>/dev/null)
K_CK=$K_A; [ "$K_TAG" = "K_P3A" ] && K_CK=$K_P3A
log "K = $K_TAG ($K_CK)"

# ------------------------------------------------- s1: site ablation on K
if should_run s1_ablation && ! done_already s1_ablation; then
  log "s1: inference ablations on K (none / off0 / off1 / off0+1)"
  for v in "none:" "off0:0" "off1:1" "off01:0,1"; do
    tag=${v%%:*}; sites=${v#*:}
    run_swap "abl_$tag" "$K_CK" --chars 30 --writers 10 --seeds 3 \
      --off-sites "$sites" || exit 1
  done
  # content check on the simplified structure we intend to ship
  run_eval K_abl_off01 "$K_CK" --skip-writer || exit 1
  mark s1_ablation
fi

# ------------------------------------------------------------ s2: specs
if should_run s2_specs && ! done_already s2_specs; then
  log "s2: build F0/F1 specs from K=$K_TAG"
  $P -u -m tools.f_20261101.build_specs --k "$K_CK" --out "$SPECS" \
    --stop 5000 --horizon 12000 || exit 1
  mark s2_specs
fi

# ------------------------------------------------- s3: smoke (freeze/EMA/resume)
if should_run s3_smoke && ! done_already s3_smoke; then
  log "s3: smoke — 60 steps, then resume 20 more; verify gains stay zero"
  SMOKE=$ROOT/smoke
  rm -rf "$SMOKE"
  $P -u -m tools.style_matrix.runtime.train --spec "$SPECS/F0.json" \
    --stop 60 --run "$SMOKE" > "$LOGS/smoke1.log" 2>&1 \
    || { log "s3 FAILED first pass"; tail -20 "$LOGS/smoke1.log"; exit 1; }
  $P -u -m tools.style_matrix.runtime.train --spec "$SPECS/F0.json" \
    --stop 80 --run "$SMOKE" --resume "$SMOKE/checkpoints/0000060.pt" \
    > "$LOGS/smoke2.log" 2>&1 \
    || { log "s3 FAILED resume pass"; tail -20 "$LOGS/smoke2.log"; exit 1; }
  $P -u -m tools.f_20261101.eval_arm --checkpoint "$SMOKE/checkpoints/0000080.pt" \
    --tag smoke --skip-writer --panel "$PANEL" --device cuda --out "$SMOKE/eval" \
    > "$LOGS/smoke_eval.log" 2>&1 || true
  grep -E '^\[sites\]' "$LOGS/smoke1.log" "$LOGS/smoke2.log" | sed 's/^/  /'
  $P - <<'PY'
import torch
ck = torch.load('assets/f_20261101/smoke/checkpoints/0000080.pt', map_location='cpu', weights_only=False)
disabled = ck.get('disabled_injection_sites') or []
assert disabled, 'checkpoint lost its disabled-site record'
bad, live = [], []
for name in ('delta', 'ema'):
    for k, v in ck[name].items():
        if 'style_injections' in k and k.endswith('.gain'):
            mx = float(v.abs().max())
            site = int(k.split('.')[1])
            live.append((name, k, round(mx, 4)))
            if site in disabled and mx != 0.0:
                bad.append((name, k, mx))
print('[smoke] disabled_injection_sites:', disabled)
for row in sorted(live):
    print('   gain', row)
assert not bad, f'a disabled site came back non-zero after resume: {bad}'
kept = [r for r in live if int(r[1].split('.')[1]) not in disabled and r[2] != 0.0]
assert kept, 'the surviving site lost its gain'
print(f'[smoke] PASS - disabled {disabled} stay 0.0 through save+resume; '
      f'surviving site keeps gain {kept[0][2]}')
PY
  [ $? -eq 0 ] || { log "s3 FAILED gain assertion"; exit 1; }
  mark s3_smoke
fi

# ---------------------------------------------------------------- s4: F0 5k
if should_run s4_f0 && ! done_already s4_f0; then
  log "s4: F0 (FM only) 5k"
  $P -u -m tools.style_matrix.runtime.train --spec "$SPECS/F0.json" \
    --stop 5000 --run "$RUNS/F0" > "$LOGS/F0.log" 2>&1 \
    || { log "s4 FAILED"; tail -30 "$LOGS/F0.log"; exit 1; }
  mark s4_f0
fi

# ---------------------------------------------------------------- s5: F1 5k
if should_run s5_f1 && ! done_already s5_f1; then
  log "s5: F1 (FM + weak E-G) 5k"
  $P -u -m tools.style_matrix.runtime.train --spec "$SPECS/F1.json" \
    --stop 5000 --run "$RUNS/F1" > "$LOGS/F1.log" 2>&1 \
    || { log "s5 FAILED"; tail -30 "$LOGS/F1.log"; exit 1; }
  grep -E 'eg_samples' "$LOGS/F1.log" | tail -3 | sed 's/^/  /'
  mark s5_f1
fi

# ------------------------------------------------------- s6: compare F0 vs F1
if should_run s6_eval_arms && ! done_already s6_eval_arms; then
  log "s6: eval F0@5000 vs F1@5000"
  for arm in F0 F1; do
    run_eval "${arm}_5000" "$RUNS/$arm/checkpoints/0050000.pt" \
      --csv "$SWAP_CSV" --chars 30 --writers 10 --seeds 3 || exit 1
  done
  $P -u -m tools.f_20261101.pick pick-arm \
    --f0 "$EVAL/F0_5000/eval.json" --f1 "$EVAL/F1_5000/eval.json" \
    --out "$ROOT/arm_choice.json" || exit 1
  mark s6_eval_arms
fi
WINNER=$($P -c "import json;print(json.load(open('$ROOT/arm_choice.json'))['winner'])" 2>/dev/null)
WINNER=${WINNER:-F0}
log "winner = $WINNER"

# --------------------------------------------------- s7: winner +7k (to 12k)
if should_run s7_extend && ! done_already s7_extend; then
  log "s7: extend $WINNER 5k -> 12k"
  $P -u -m tools.style_matrix.runtime.train --spec "$SPECS/$WINNER.json" \
    --stop 12000 --run "$RUNS/$WINNER" \
    --resume "$RUNS/$WINNER/checkpoints/0050000.pt" > "$LOGS/${WINNER}_ext.log" 2>&1 \
    || { log "s7 FAILED"; tail -30 "$LOGS/${WINNER}_ext.log"; exit 1; }
  mark s7_extend
fi

# --------------------------------------------------- s8: final acceptance
if should_run s8_final && ! done_already s8_final; then
  log "s8: final acceptance on $WINNER@12000"
  run_eval "${WINNER}_12000" "$RUNS/$WINNER/checkpoints/0120000.pt" \
    --csv "$SWAP_CSV" --chars 30 --writers 10 --seeds 3 || exit 1

  log "s8: inference sweep (CFG 1.0/1.3/1.6 x site2 gain 1.0/1.2)"
  for cfg in 1.0 1.3 1.6; do
    for gain in 1.0 1.2; do
      tag="sweep_cfg${cfg}_g${gain}"
      gs=(); [ "$gain" != "1.0" ] && gs=(--gain-scale "2:$gain")
      run_swap "$tag" "$RUNS/$WINNER/checkpoints/0120000.pt" \
        --chars 20 --writers 10 --seeds 2 --cfg "$cfg" "${gs[@]}" || log "  sweep $tag failed"
    done
  done
  mark s8_final
fi

log "ALL STAGES DONE"
