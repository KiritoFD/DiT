#!/bin/bash
# F3 on 48, serialized inside one tmux session. F1/F2 already completed.
set -u
cd /home/ds/Workspace/DiT || exit 1
PY=/home/ds/miniconda3/envs/vcp-embryo/bin/python
LOG=assets/style_24h48_20261026

run () {
  local name="$1"; shift
  echo "===== START $name $(date -Is) =====" | tee -a "$LOG/chain.log"
  "$@" >> "$LOG/chain.log" 2>&1
  echo "===== END   $name rc=$? $(date -Is) =====" | tee -a "$LOG/chain.log"
}

run F3_E_G $PY -u -m tools.style_24h48.f3_transfer --kind G
run F3_E_L $PY -u -m tools.style_24h48.f3_transfer --kind L

echo "CHAIN2 DONE $(date -Is)" | tee -a "$LOG/chain.log"
