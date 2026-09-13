#!/bin/bash
cd /root/Workspace/xy/DiT
CKPT_DIR=assets/results/ctrl_skel/20260823-085301-ctrl-skel-top30-scratch/checkpoints

# Clean old state
rm -f $CKPT_DIR/cpu_eval_state.json

# GPU eval all 3 ckpts (batch=16, 100 samples, ~2min each)
for pt in $CKPT_DIR/*.pt; do
    echo "=== eval $pt ==="
    /opt/conda/bin/python auto_eval_ctrl.py \
        --ckpt-dir $CKPT_DIR \
        --eval-csv assets/eval100_top30.csv \
        --from-scratch --device cuda --batch 16 --once --steps 50
done
echo "=== ALL DONE ==="
ls -la $CKPT_DIR/eval_auto_*.json
