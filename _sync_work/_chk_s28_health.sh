#!/bin/bash
cd /root/Workspace/xy/DiT/assets/results/s28_std_dino_pretrain
LATEST=$(ls -dt 2026*/ | head -1)
echo "run: $LATEST"
echo "=== last 6 step lines ==="
grep -a -E 'step=' "$LATEST/log.txt" | tail -6
echo "=== any errors/NaN ==="
grep -a -i -E 'error|traceback|nan|warning' "$LATEST/log.txt" | tail -5
echo "=== checkpoint dir ==="
ls "$LATEST/checkpoints/" 2>/dev/null | tail -8
echo "=== eval_auto ==="
ls "$LATEST/checkpoints/"eval_auto_*.json 2>/dev/null || echo none
