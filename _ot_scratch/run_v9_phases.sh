#!/bin/bash
# run_v9_phases.sh — 分阶段跑 v9 encode (幂等可断点续跑)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
for ph in img skel3 skel1; do
  echo "===PHASE-${ph}===" >> /tmp/enc_v9.log
  $PY /tmp/cpu_encode_v9.py --nproc 16 --threads 4 --batch 16 --phase "${ph}" >> /tmp/enc_v9.log 2>&1 || { echo "FAIL-${ph}" >> /tmp/enc_v9.log; break; }
done
echo "===ALLDONE-$(date +%T)===" >> /tmp/enc_v9.log