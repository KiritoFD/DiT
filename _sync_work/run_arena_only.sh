#!/usr/bin/env bash
# 只跑竞技场 (两遍编码已完成, latent 直接读 shard -> 起步 11s)
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_purestd
LOG="exp-std/logs_purestd/arena_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_purestd/arena_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python

$PY -u tools/latent_signal_arena.py \
    --n-triplets 20000 --n-unique 45000 --n-pairs 4000 \
    --swd-proj 128 --steps 300 --targets 8 --seeds 2 --enc-batch 64 \
    --flux-shards exp-std/data/shards_img_flux16 \
    --out-dir exp-std/signal_arena_full

echo
echo "########## summary.txt ##########"
cat exp-std/signal_arena_full/summary.txt 2>/dev/null
ls -la exp-std/signal_arena_full/ 2>/dev/null
echo "[arena] 完成 $(date '+%F %T')"
