#!/usr/bin/env bash
set -euo pipefail
base=/root/Workspace/xy/DiT
cd "$base/scratch/writer_pixel_d0d1_20261016_01"
nohup /opt/conda/envs/cu121/bin/python -u tools/analysis/run_allograph_audit.py --out "$base/assets/allograph_audit_20261016" --batch 8 --memory-fraction .12 > "$base/logs/allograph_audit_20261016.log" 2>&1 < /dev/null &
printf 'ALLOGRAPH_PID=%s\n' "$!"
