#!/bin/bash
# noise_all_v8dhi.sh — 对 v8d/v8h/v8i 落盘图 (best 附近 step) 算噪点指标
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT${PYTHONPATH:+:$PYTHONPATH}
PY=/opt/conda/envs/cu121/bin/python
OUT=assets/results/v8_3stage/noise_v8dhi
mkdir -p "$OUT"

calc() { # exp step tag
  local exp=$1 step=$2 tag=$3
  local dir="assets/results/v8_3stage/$exp/2026*/checkpoints/eval_samples_ctrl/step$step/$tag"
  local d=$(ls -d $dir 2>/dev/null | head -1)
  [ -z "$d" ] && { echo "[$exp@$step $tag] NO DIR"; return; }
  $PY src/eval/metrics_png.py --dir "$d" --tag "$tag" --n 100 --out "$OUT/${exp}_${step}_${tag}.json" > "$OUT/${exp}_${step}_${tag}.log" 2>&1
  echo "[$exp@$step $tag] rc=$?"
}

echo "=== v8d (unfreeze) ==="
calc v8d 0035000 ctrl
calc v8d 0035000 base
echo "=== v8h (repa-early) ==="
calc v8h 0040000 ctrl
calc v8h 0040000 base
echo "=== v8i (combo) ==="
calc v8i 0032500 ctrl
calc v8i 0032500 base
echo ALL_DONE