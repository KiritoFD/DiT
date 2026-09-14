#!/bin/bash
# runner_base_encode.sh - base encode, 4 phases sequential, nice.
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
for ph in std_skel img aux_skel3 aux_canny; do
  echo "=== PHASE $ph start $(date) ==="
  nice -n 15 $PY -u tools/cpu_encode_base.py --phase $ph --nproc 32 --threads 4 --batch 16
  rc=$?
  if [ $rc -ne 0 ]; then echo "PHASE_FAILED $ph"; fi
done
echo "ALL DONE $(date)" > logs/encode_base_done.txt
