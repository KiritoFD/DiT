#!/bin/bash
# bench_matrix.sh — CPU eval 参数矩阵 (每组 16 样本 ctrl 臂)
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
# A: 56 threads (双 socket), batch 16
taskset -c 0-63 $PY tools/diag/bench_cpu2.py 56 16 2>&1 | tail -1 | sed 's/^/A_56t_b16: /'
# B: 56 threads, batch 32
taskset -c 0-63 $PY tools/diag/bench_cpu2.py 56 32 2>&1 | tail -1 | sed 's/^/B_56t_b32: /'
# C: 32 threads 单 socket, batch 32
taskset -c 32-63 $PY tools/diag/bench_cpu2.py 32 32 2>&1 | tail -1 | sed 's/^/C_32t_b32: /'
echo MATRIX_DONE
