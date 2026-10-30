#!/usr/bin/env bash
set -euo pipefail
cd /home/ds/Workspace/DiT
PY=/home/ds/miniconda3/envs/vcp-embryo/bin/python
ROOT=assets/p125_20261030
STATE=docs/1031/state.json
echo "{\"start\":\"$(date +%T)\"}" > $STATE
echo "=== A: S5@85k E-G eval x3 seeds ==="
for s in 0 1 2; do
  nice -n 5 "$PY" -u -m tools.p125.eval_p125 --run assets/style_focus_85k_20261029/48_v3/runs/S5 --step 85000 \
    --protocol heldout187_seed$s --csv exp-std/csv/eval200_fixed.csv --arm S5@85k --out docs/1031/eg_metrics.csv
done
echo "=== C: writer-swap panel + CFG sweep ==="
"$PY" -u -m tools.p125.writer_swap --out $ROOT/swap_panel
echo "=== D: causal ablation ==="
"$PY" -u -m tools.p125.ablation --out docs/1031/ablation.json
echo "=== E1: E-G2 verifier training ==="
"$PY" -u -m tools.p125.eg2
echo "=== E2: latent gradient-transfer pilot ==="
"$PY" -u -m tools.p125.latent_pilot
echo "=== F: build + run 2k forks ==="
"$PY" -u -m tools.p125.forks
"$PY" -u -m tools.style_matrix.runtime.serial --plan $ROOT/forks48/plan_FMA.json --execute
"$PY" -u -m tools.style_matrix.runtime.serial --plan $ROOT/forks48/plan_FMB.json --execute
echo "{\"done\":\"$(date +%T)\"}" >> $STATE
echo ALL-1031-DONE
