#!/usr/bin/env bash
# Evaluate T0 vs T1 at both checkpoints with the F4 writer-difference probe.
# Same extractor, same characters, same seed -> the four runs differ ONLY in
# the weights, so a difference between them is attributable to the arm.
set -eu
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/u100_p1p2_20261025/p2_train.csv

for arm in T0 T1; do
  case $arm in
    T0) RUN=assets/style_24h_20261026/ablations/T0/20261026-090007-Unet-flow ;;
    T1) RUN=assets/style_24h_20261026/ablations/T1/20261026-095423-Unet-flow ;;
  esac
  for step in 0002500 0005000; do
    if [ -s "assets/t_eval/${arm}_${step}/gt_vs_generated.json" ]; then
      continue
    fi
    echo "===== START ${arm}_${step} $(date -Iseconds) ====="
    $PY -u -m tools.stroke_interaction.f4_gt_vs_gen \
        --ckpt "$RUN/checkpoints/${step}.pt" \
        --csv "$CSV" \
        --out "assets/t_eval/${arm}_${step}" \
        --chars 40 --per-kind 20 --seed 11
    echo "===== END ${arm}_${step} rc=$? $(date -Iseconds) ====="
  done
done
echo "ALL_DONE $(date -Iseconds)"
