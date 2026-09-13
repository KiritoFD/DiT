#!/bin/bash
cd /root/Workspace/xy/DiT
EVAL_DIR=$(ls -d 5script/results/s15_ws_flow/*/checkpoints/eval_samples 2>/dev/null | head -1)
SEEN_DIR=$(ls -d 5script/results/s15_ws_flow/*/checkpoints/seen_samples 2>/dev/null | head -1)
CKPT_DIR=$(ls -d 5script/results/s15_ws_flow/*/checkpoints 2>/dev/null | head -1)
echo "EVAL_DIR=$EVAL_DIR"
echo "SEEN_DIR=$SEEN_DIR"
echo "CKPT_DIR=$CKPT_DIR"

# Eval poster (no seen row)
/opt/conda/bin/python tools/make_eval_poster_eval.py \
    --show5-dir "$EVAL_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --exp "s15 WS/2 Flow 3top30 (200k)" \
    --n-samples 10 --step-stride 10 \
    -o /tmp/s15_eval_poster.png

# Seen poster (separate)
/opt/conda/bin/python tools/make_seen_poster.py \
    --seen-dir "$SEEN_DIR" \
    --exp "s15 WS/2 Flow 3top30 (200k) SEEN" \
    -o /tmp/s15_seen_poster.png

ls -la /tmp/s15_eval_poster.png /tmp/s15_seen_poster.png