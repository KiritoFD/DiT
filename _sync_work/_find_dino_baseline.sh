#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== results dirs with eval_auto metrics ==="
for d in assets/results/*/; do
  r=$(basename "$d")
  n=$(ls "$d"*/checkpoints/eval_auto_*.json 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then
    echo "$r: $n eval points"
  fi
done
echo ""
echo "=== configs referencing dino/char embedding ==="
for c in src/train/configs/s*.json; do
  if grep -q -a 'dino\|char_embed\|freeze_char' "$c" 2>/dev/null; then
    echo "--- $c ---"
    grep -a -E 'char_embed_dim|freeze_char|char_dino|use_ids' "$c" | head -4
  fi
done
