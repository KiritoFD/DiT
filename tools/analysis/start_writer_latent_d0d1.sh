#!/usr/bin/env bash
set -euo pipefail
cd /root/Workspace/xy/DiT/scratch/writer_latent_d0d1_20261016_01
test ! -e assets/D0
test ! -e assets/D1
nohup bash tools/analysis/run_writer_latent_d0d1.sh > /root/Workspace/xy/DiT/logs/writer_latent_d0d1_20261016.log 2>&1 < /dev/null &
printf '%s\n' "$!" > assets/pipeline.pid
printf 'WRITER_LATENT_PIPELINE_PID=%s\n' "$!"
date -u
