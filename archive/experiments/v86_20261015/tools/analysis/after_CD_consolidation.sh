#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python
exec 8>logs/consolidation_waiter.lock
flock -n 8 || { echo 'Consolidation waiter already exists'; exit 1; }
echo "WAIT C/D completion $(date -Is)"
until grep -q '^DONE D ' logs/full_followups_10k_20261015.log; do
  pgrep -f '^bash tools/analysis/run_full_followups.sh B50 C D$' >/dev/null || {
    echo 'C/D queue failed or stopped; refusing to continue'; exit 1;
  }
  sleep 30
done
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do sleep 15; done
exec 9>logs/strong_style_queue.lock
flock -w 120 9 || { echo 'GPU queue lock unavailable'; exit 1; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo 'GPU occupied after lock acquisition; refusing concurrent launch'; exit 1
fi
"$PY" tools/analysis/prepare_consolidation.py
for arm in low15 matched80; do
  log="logs/v86_consolidate_${arm}_20261015.log"
  echo "START $arm $(date -Is)"
  MASTER_PORT=12811 "$PY" -m src.train.train --config "src/train/configs/v86_consolidate_${arm}.json" --num-workers 4 > "$log" 2>&1
  grep -q 'Reached max_steps=5000' "$log"
  echo "DONE $arm $(date -Is)"
done
sets=(--gen base=assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix
      --gen parent5k=assets/results/v86_full_fixed_B25/20261015-071058-v86-full-fixed-B25/eval_samples_ctrl/step0005000/eval200fix)
for arm in low15 matched80; do
  paths=(assets/results/v86_consolidate_"$arm"/*/eval_samples_ctrl/step0005000/eval200fix)
  [[ ${#paths[@]} == 1 && -d "${paths[0]}" ]] || { echo "Ambiguous output: $arm"; exit 1; }
  sets+=(--gen "$arm=${paths[0]}")
done
"$PY" tools/eval_gen_with_disc.py --device cpu "${sets[@]}" --out assets/B25_consolidation_independent_20261015.json > logs/B25_consolidation_independent_20261015.log 2>&1
echo "DIAGNOSTIC START $(date -Is)"
"$PY" tools/analysis/probe_B25_gradients.py > logs/B25_gradient_replay_20261015.log 2>&1
"$PY" tools/analysis/diagnose_B25_drift.py --device cuda --raw-critic > logs/B25_drift_gpu_20261015.log 2>&1
"$PY" tools/v85_compare.py \
  --set B25_0k=assets/runs/v84A_std_cond/20261013-072215-v84A-std-cond/eval_samples_ctrl/step0200000/eval200fix \
  --set B25_5k=assets/results/v86_full_fixed_B25/20261015-071058-v86-full-fixed-B25/eval_samples_ctrl/step0005000/eval200fix \
  --set B25_10k=assets/results/v86_full_fixed_B25/20261015-071058-v86-full-fixed-B25/eval_samples_ctrl/step0010000/eval200fix \
  --out assets/style_eval_B25_drift_20261015 > logs/B25_style_drift_20261015.log 2>&1
"$PY" -c 'import json; from pathlib import Path; root=Path("assets/style_eval_B25_drift_20261015"); names={"B25_0k","B25_5k","B25_10k"}; [(_ for _ in ()).throw(RuntimeError("incomplete style report")) for enc in ("DINO-S","CLIP-S") if not names <= set(json.loads((root/enc/"style_report.json").read_text())["sets"])]'
echo "DIAGNOSTIC DONE $(date -Is)"
