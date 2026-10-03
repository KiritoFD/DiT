#!/bin/bash
# env_ab.sh — 分配器选型 A/B: baseline vs jemalloc vs MALLOC env (forward 时间, 32t)
OUT=/tmp/env_ab.log
PY=/opt/conda/envs/cu121/bin/python
cd /root/Workspace/xy/DiT
{
echo "=== 环境检查 ==="
which numactl || echo "numactl MISSING"
ls -la /usr/lib/x86_64-linux-gnu/libjemalloc.so* 2>/dev/null || echo "jemalloc so MISSING"
dpkg -l | grep -i jemalloc || true
echo "=== A/B forward (profile_cpu2 的 S2' 部分, 每组独立进程) ==="
echo "--- A1 baseline ---"
taskset -c 32-63 $PY tools/diag/profile_cpu2.py 32 2>&1 | grep -E "eager CFG forward"
echo "--- A2 jemalloc ---"
if [ -f /opt/conda/envs/cu121/lib/libjemalloclocal.so.2 ]; then
  LD_PRELOAD=/opt/conda/envs/cu121/lib/libjemalloclocal.so.2 taskset -c 32-63 $PY tools/diag/profile_cpu2.py 32 2>&1 | grep -E "eager CFG forward"
else
  echo "jemalloc 不可用, 跳过"
fi
echo "--- A3 MALLOC env ---"
MALLOC_MMAP_THRESHOLD_=2147483648 MALLOC_TRIM_THRESHOLD_=2147483648 taskset -c 32-63 $PY tools/diag/profile_cpu2.py 32 2>&1 | grep -E "eager CFG forward"
echo "=== 16样本端到端复测 (选型验证: jemalloc) ==="
if [ -f /opt/conda/envs/cu121/lib/libjemalloclocal.so.2 ]; then
  LD_PRELOAD=/opt/conda/envs/cu121/lib/libjemalloclocal.so.2 taskset -c 32-63 $PY tools/diag/bench_cpu2.py 32 16 2>&1 | tail -1
fi
echo "ENV_AB_DONE"
} > $OUT 2>&1
