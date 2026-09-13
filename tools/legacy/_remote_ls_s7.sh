#!/usr/bin/env bash
set -e
cd /root/Workspace/xy/DiT/5script/results/s7_klf4_top30
for d in */; do
  echo "DIR: $d"
  ls "${d}checkpoints"/eval_auto_*.json 2>/dev/null | head
  ls "${d}checkpoints"/cpu_eval_state.json 2>/dev/null
done
echo "---LATEST LOG---"
find . -name log.txt | sort | tail -1