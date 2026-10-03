#!/bin/bash
cd /root/Workspace/xy/DiT
RUN=assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1pix/checkpoints
echo "=== s31 checkpoints dirs ==="
ls "$RUN" | grep -i eval
echo "=== eval_samples (non-ctrl) exists? ==="
ls "$RUN/eval_samples" 2>/dev/null | head -3
echo "=== look for gt under eval_samples_ctrl step dir parent ==="
ls "$RUN/eval_samples_ctrl/step0030000" 2>/dev/null
echo "=== find any gt files for s31 ==="
find "$RUN" -name "gt*.png" 2>/dev/null | head -3
echo "=== s30 (base) gt vs s31 sample alignment check: list gt0/sample0/ctrl0 exist? ==="
ls assets/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/eval_samples/step0030000/gt0.png
ls assets/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1pix/checkpoints/eval_samples_ctrl/step0030000/ctrl/ctrl0.png
