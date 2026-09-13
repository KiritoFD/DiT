#!/usr/bin/env bash
set -e
D=/root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/20260823-234546-s8-klf4-clean-dino/checkpoints
for f in "$D"/eval_auto_*.json; do
  echo "== $(basename "$f") =="
  cat "$f"
  echo
done