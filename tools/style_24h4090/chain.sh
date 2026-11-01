#!/bin/bash
# T0/T1/T2 on 4090, serialized inside one tmux session.
# Waits for the A0 equal-exposure run to release the GPU instead of preempting it.
#
# gpu.lock is shared with chain_g1b.sh: both queues poll the same "GPU free"
# condition, so whoever gets the lock first runs to completion and the other
# waits. Without it they would both start the instant A0 exits and OOM.
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
LOG=assets/style_24h_20261026
LOCK=$LOG/gpu.lock

# ---- 1. wait for GPU and take the lock ----
for i in $(seq 1 1440); do
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$U" -lt 2500 ] && mkdir "$LOCK" 2>/dev/null; then
    echo "GPU free + lock acquired at $(date -Is) used=${U}MiB" | tee -a "$LOG/chain.log"
    break
  fi
  [ $((i % 10)) -eq 0 ] && echo "waiting: used=${U}MiB $(date -Is)" | tee -a "$LOG/chain.log"
  sleep 30
done

# Re-check after taking the lock: another process can grab VRAM between the
# check and the mkdir, and a 5k run that OOMs 2 minutes in wastes the slot and
# leaves a misleading "START" line in the log.
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$U" -ge 2500 ]; then
  echo "ABORT: lock taken but GPU now shows ${U}MiB in use" | tee -a "$LOG/chain.log"
  rmdir "$LOCK" 2>/dev/null
  exit 1
fi

trap 'rmdir "$LOCK" 2>/dev/null' EXIT

run () {
  local name="$1"; shift
  local log="$LOG/abl_${name}.log"
  echo "===== START $name $(date -Is) =====" | tee -a "$LOG/chain.log"
  "$@" > "$log" 2>&1
  local rc=$?
  cat "$log" >> "$LOG/chain.log"
  echo "===== END   $name rc=$rc $(date -Is) =====" | tee -a "$LOG/chain.log"
  if grep -q "OutOfMemoryError" "$log"; then
    echo "ABORT: $name hit OOM, not launching the remaining arms" | tee -a "$LOG/chain.log"
    exit 1
  fi
  [ $rc -ne 0 ] && echo "WARN: $name exited rc=$rc (no OOM)" | tee -a "$LOG/chain.log"
  return 0
}

for ARM in T0 T1 T2; do
  run "ABL_$ARM" $PY -u -m tools.style_24h4090.run_ablation --arm $ARM --steps 5000
done

echo "CHAIN DONE $(date -Is)" | tee -a "$LOG/chain.log"
