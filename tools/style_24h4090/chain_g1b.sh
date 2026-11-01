#!/bin/bash
# G1 second half: runs AFTER the T-chain, on the same gpu.lock queue.
#
# Launch order matters: chain.sh takes the lock first and holds it for the whole
# T0->T1->T2 sequence, so this script just waits. If chain.sh finishes and this
# one gets the lock, it runs alone.
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
LOG=assets/style_24h_20261026
LOCK=$LOG/gpu.lock

echo "===== G1B queued $(date -Is) =====" | tee -a "$LOG/chain.log"
while true; do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$U" -lt 2500 ] && mkdir "$LOCK" 2>/dev/null; then
    echo "g1b: acquired GPU lock, used=${U}MiB $(date -Is)" | tee -a "$LOG/chain.log"
    break
  fi
  echo "waiting: used=${U}MiB lock=$( [ -d "$LOCK" ] && echo held || echo free ) $(date -Is)" \
    >> "$LOG/g1b_wait.log"
  sleep 60
done

trap 'rmdir "$LOCK" 2>/dev/null' EXIT

# Re-check after taking the lock; something outside our queues may have grabbed VRAM.
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$U" -ge 2500 ]; then
  echo "g1b ABORT: lock taken but GPU shows ${U}MiB" | tee -a "$LOG/chain.log"
  exit 1
fi

echo "===== START G1B $(date -Is) =====" | tee -a "$LOG/chain.log"
# --limit must match gt_latents.npz rows (100); the script self-corrects anyway.
$PY -u -m tools.style_24h4090.g1_gt_vs_gen \
  --ckpt assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt \
  --limit 100 --cfg 1.0 >> "$LOG/g1b.log" 2>&1
rc=$?
echo "===== END G1B rc=$rc $(date -Is) =====" | tee -a "$LOG/chain.log"
