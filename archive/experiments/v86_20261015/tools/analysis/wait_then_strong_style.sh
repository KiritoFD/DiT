#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec 8>logs/strong_style_waiter.lock
flock -n 8 || { echo 'Waiter already running'; exit 1; }
echo "WAIT current table+frozen queue $(date -Is)"
while ! grep -q '^DONE frozen ' logs/v86A_writer_table_chain_20261014.log; do
  if ! pgrep -f '^bash tools/analysis/run_v86A_writer_table.sh ' >/dev/null; then
    echo 'Previous queue exited without completing frozen arm; refusing launch'; exit 1
  fi
  sleep 30
done
grep -q 'Reached max_steps=20000' logs/v86A_writer_table_20261014.log
grep -q 'Reached max_steps=20000' logs/v86A_writer_frozen_20261014.log
# Let the owned predecessor finish releasing CUDA; never kill any process.
while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do sleep 30; done
echo "START serial $(date -Is)"
exec bash tools/analysis/run_strong_style_serial.sh
