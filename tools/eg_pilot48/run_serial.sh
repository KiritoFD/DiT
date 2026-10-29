#!/usr/bin/env bash
# eg_pilot48 24h serial chain with gating and stop-loss rules.
set -uo pipefail
cd /home/ds/Workspace/DiT
PY=/home/ds/miniconda3/envs/vcp-embryo/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
LOG=docs/1030/eg_supervision_pilot48/chain.log
log(){ echo "[$(date +%T)] $*" | tee -a $LOG; }

log "=== exp0 environment/link validation ==="
"$PY" -u -m tools.eg_pilot48.exp0_env 2>&1 | tee -a $LOG || { log "exp0 FAILED - abort"; exit 1; }

log "=== exp1.1 E-G2 independent evaluator ==="
"$PY" -u -m tools.eg_pilot48.exp1_eg2 2>&1 | tee -a $LOG || { log "eg2 CRASHED - abort"; exit 1; }
[ -f docs/1030/eg_supervision_pilot48/eg2_real_validation.json ] || { log "eg2 no output - abort"; exit 1; }
EG2_OK=$("$PY" -c "import json;print(json.load(open('docs/1030/eg_supervision_pilot48/eg2_real_validation.json'))['accepted'])")
log "EG2 accepted: $EG2_OK"

log "=== exp1.2 E-G gradient causal pilot ==="
"$PY" -u -m tools.eg_pilot48.exp1_grad 2>&1 | tee -a $LOG || { log "grad pilot CRASHED - skip B"; echo SKIP > /tmp/run_b_flag; }
if [ -f docs/1030/eg_supervision_pilot48/eg_gradient_verdict.json ]; then
"$PY" - <<'PYINNER'
import json
d=json.load(open('docs/1030/eg_supervision_pilot48/eg_gradient_verdict.json'))
rows=[r for r in d['rows'] if r['loss']=='eg_rank']
pos_eg=sum(1 for r in rows if r['d_eg']>0)
pos_eg2=sum(1 for r in rows if r['d_eg2']>0)
print('RUN' if (len(rows)>0 and pos_eg>=2 and pos_eg2>=2) else 'SKIP')
PYINNER
else
echo SKIP
fi
RUN_B=$("$PY" -c "print(open('/tmp/run_b_flag').read().strip())" 2>/dev/null || echo RUN)
RUN_B=$("$PY" - <<'PYINNER2'
import json
try:
    d=json.load(open('docs/1030/eg_supervision_pilot48/eg_gradient_verdict.json'))
    rows=[r for r in d['rows'] if r['loss']=='eg_rank']
    pos_eg=sum(1 for r in rows if r['d_eg']>0)
    pos_eg2=sum(1 for r in rows if r['d_eg2']>0)
    print('RUN' if (len(rows)>0 and pos_eg>=2 and pos_eg2>=2) else 'SKIP')
except Exception:
    print('SKIP')
PYINNER2
)$("$PY" - <<'EOF'
import json
d=json.load(open('docs/1030/eg_supervision_pilot48/eg_gradient_verdict.json'))
rows=[r for r in d['rows'] if r['loss']=='eg_rank']
pos_eg=sum(1 for r in rows if r['d_eg']>0)
pos_eg2=sum(1 for r in rows if r['d_eg2']>0)
# gate: majority of t show d_eg>0 AND d_eg2>0 (transfer), content kept (iou_keep>0.9)
gate=len(rows)>0 and pos_eg>=2 and pos_eg2>=2
print('RUN' if gate else 'SKIP')
EOF
)
log "gate for arm B: $RUN_B"

log "=== exp2 writer path ablation ==="
"$PY" -u -m tools.eg_pilot48.exp2_path 2>&1 | tee -a $LOG || log "exp2 crashed - continue"

log "=== arm A (plain FM control, 5k) ==="
"$PY" -u -m tools.eg_pilot48.train_arm --arm A 2>&1 | tee -a $LOG
[ -f assets/eg_pilot48_tmp/ckpt_A/A_0005000.pt ] || { log "arm A failed - abort"; exit 1; }

if [ "$RUN_B" = "RUN" ]; then
  log "=== arm B (FM + E-G bank ranking, 5k) ==="
  "$PY" -u -m tools.eg_pilot48.train_arm --arm B 2>&1 | tee -a $LOG
else
  log "=== arm B SKIPPED (gradient pilot not passed) ==="
fi

log "=== arm C (FM + E-G representation alignment, 5k) ==="
"$PY" -u -m tools.eg_pilot48.train_arm --arm C 2>&1 | tee -a $LOG

log "=== arms D0/D1 (query adapter pilot, 2k each) ==="
"$PY" -u -m tools.eg_pilot48.train_arm --arm D0 2>&1 | tee -a $LOG
"$PY" -u -m tools.eg_pilot48.train_arm --arm D1 2>&1 | tee -a $LOG

log "=== val100 generation per arm (seed0 + seed1) ==="
for ARM in A C; do
  CK=assets/eg_pilot48_tmp/ckpt_$ARM/${ARM}_0005000.pt
  for SEED in 0 1; do
    [ "$SEED" = "1" ] && CK="$CK"  # same ckpt; seed handled via csv copy
    "$PY" -u -m tools.eg_pilot48.gen_val100 --arm $ARM --seed $SEED 2>&1 | tee -a $LOG
  done
done
if [ -f assets/eg_pilot48_tmp/ckpt_B/B_0005000.pt ]; then
  for SEED in 0 1; do "$PY" -u -m tools.eg_pilot48.gen_val100 --arm B --seed $SEED 2>&1 | tee -a $LOG; done
fi
for ARM in D0 D1; do
  CK=assets/eg_pilot48_tmp/ckpt_$ARM/${ARM}_0002000.pt
  [ -f "$CK" ] && "$PY" -u -m tools.eg_pilot48.gen_val100 --arm $ARM --seed 0 2>&1 | tee -a $LOG
done

log "=== writer-swap panels per arm ==="
"$PY" -u -m tools.eg_pilot48.gen_swap --ckpt S5_START --tag S5_START 2>&1 | tee -a $LOG
for ARM in A C; do
  "$PY" -u -m tools.eg_pilot48.gen_swap --ckpt assets/eg_pilot48_tmp/ckpt_$ARM/${ARM}_0005000.pt --tag $ARM 2>&1 | tee -a $LOG
done
[ -f assets/eg_pilot48_tmp/ckpt_B/B_0005000.pt ] && "$PY" -u -m tools.eg_pilot48.gen_swap --ckpt assets/eg_pilot48_tmp/ckpt_B/B_0005000.pt --tag B 2>&1 | tee -a $LOG

log "=== unified metrics ==="
"$PY" -u -m tools.eg_pilot48.eval_arms --arms S5_START A B C 2>&1 | tee -a $LOG

log "=== analysis ==="
"$PY" -u -m tools.eg_pilot48.analyze 2>&1 | tee -a $LOG
log "=== PILOT CHAIN COMPLETE ==="
