#!/usr/bin/env bash
set -euo pipefail
work=/root/Workspace/xy/DiT/scratch/writer_pixel_d0d1_20261016_01
cd "$work"
test ! -e assets/D0
test ! -e assets/D1
nohup bash tools/analysis/run_writer_pixel_d0d1.sh > /root/Workspace/xy/DiT/logs/writer_pixel_d0d1_20261016.log 2>&1 < /dev/null &
printf '%s\n' "$!" > assets/pipeline.pid
printf 'WRITER_PIXEL_PIPELINE_PID=%s\n' "$!"
date -u
