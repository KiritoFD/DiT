#!/usr/bin/env bash
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=.
PY=/opt/conda/envs/cu121/bin/python
wait_pid=${1:?Pass the existing owned Control training PID}
while kill -0 "$wait_pid" 2>/dev/null; do sleep 30; done
# The previous short Control must finish successfully before claiming its GPU.
grep -q 'Reached max_steps=5000' logs/v86_diag_control_20261014.log
test -f assets/writer_counterfactual_20261014/manifest.json
until test -f assets/writer_counterfactual_20261014/metrics.json; do sleep 30; done
"$PY" tools/analysis/prepare_v86A_writer_table.py
port=12671
for arm in table frozen; do
  echo "START $arm $(date -Is)"
  MASTER_PORT=$port "$PY" -m src.train.train --config "src/train/configs/v86A_writer_${arm}_20k.json" --num-workers 4 > "logs/v86A_writer_${arm}_20261014.log" 2>&1
  grep -q 'Reached max_steps=20000' "logs/v86A_writer_${arm}_20261014.log"
  echo "DONE $arm $(date -Is)"
  port=$((port+1))
done
