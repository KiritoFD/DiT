#!/bin/bash
# runner_tj_cpu_v2.sh - v9-style multiprocess CPU encode, sequential phases, nice.
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
for ph in img aux_skel3 aux_canny std_skel; do
  echo "=== PHASE $ph start $(date) ==="
  nice -n 15 $PY -u tools/cpu_encode_tj.py --phase $ph --nproc 16 --threads 4 --batch 16
done
echo "ALL DONE $(date)" > logs/tj_cpu_done.txt
