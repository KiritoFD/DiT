#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== s29_ctrl_gt_skel_1px runs ==="
ls -dt assets/results/s29_ctrl_gt_skel_1px/2026*/ 2>/dev/null
echo "  eval_auto_ctrl count:"
ls assets/results/s29_ctrl_gt_skel_1px/*/checkpoints/eval_auto_ctrl_*.json 2>/dev/null | wc -l
echo "=== v8_3stage runs ==="
ls -dt assets/results/v8_3stage/2026*/ 2>/dev/null
echo "  eval_auto count:"
ls assets/results/v8_3stage/*/checkpoints/eval_auto_*.json 2>/dev/null | wc -l
echo "=== s32c 是否还在跑 ==="
pgrep -af "v8a_s32\|s32c\|train.py.*s32" | grep -v grep | head -3
echo "=== 当前在跑的训练 config ==="
pgrep -af "train.py" | grep -v grep | head -5
