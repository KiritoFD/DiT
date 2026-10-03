#!/usr/bin/env bash
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_purestd
LOG="exp-std/logs_purestd/vae_probe_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_purestd/vae_probe_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python
$PY -u tools/vae_space_probe.py --pixel-cap 8000 --n-pairs 4000 --probe-steps 400
echo "[probe] 完成 $(date '+%F %T')"
cat exp-std/signal_arena_full/vae_space_probe.csv 2>/dev/null
