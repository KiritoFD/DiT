#!/bin/bash
cd /root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/opt/conda/envs/cu121/bin/python -u tools/poster_skelnet_now.py \
    --n 6 --stride 400 --out _ot_scratch/poster_now5.png 2>&1 \
    | grep -vE 'Warning|warn|pytree|pkg_resources|preload|skel-guard' | tail -20
