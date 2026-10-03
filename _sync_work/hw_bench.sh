#!/bin/bash
# hw_bench.sh — CPU/RAM 硬件档案 + 全套 microbench (一次跑完落盘)
OUT=/tmp/hw_bench.log
PY=/opt/conda/envs/cu121/bin/python
cd /root/Workspace/xy/DiT

{
echo "########## S0 硬件档案 ##########"
echo "--- lscpu ---"; lscpu
echo "--- lscpu 缓存 ---"; lscpu -C 2>/dev/null
echo "--- lscpu 拓扑(前40行) ---"; lscpu -e=CPU,NODE,SOCKET,CORE,ONLINE 2>/dev/null | head -40
echo "--- NUMA ---"; numactl -H 2>/dev/null || cat /sys/devices/system/node/node*/meminfo 2>/dev/null | head -10
echo "--- 内存 ---"; free -g; cat /proc/meminfo | grep -E "MemTotal|HugePages_Total|Hugepagesize|AnonHugePages"
echo "--- THP ---"; cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
echo "--- glibc ---"; ldd --version | head -1
echo "--- 内核 ---"; uname -r; cat /etc/os-release | head -2
echo "--- 频率策略/当前频率(样本) ---"
for f in /sys/devices/system/cpu/cpu32/cpufreq/scaling_governor /sys/devices/system/cpu/cpu32/cpufreq/scaling_cur_freq /sys/devices/system/cpu/cpu32/cpufreq/scaling_max_freq; do echo "$f = $(cat $f 2>/dev/null)"; done
echo "--- 负载/进程 ---"; uptime; ps aux --sort=-%cpu | head -6 | awk '{printf "%s%% cpu %s%% mem %s\n", $3, $4, $11}'
echo "--- torch 环境 ---"
$PY -c "import torch; print('torch', torch.__version__, '| cap', torch.backends.cpu.get_cpu_capability(), '| mkldnn', torch.backends.mkldnn.is_available(), '| threads', torch.get_num_threads())"
echo
echo "########## S1 GEMM 线程扩展性 (socket1 逐级绑核) ##########"
for T in 1 2 4 8 16 32; do
  END=$((31+T))
  taskset -c 32-$END $PY tools/diag/hw_gemm.py $T 2>/dev/null
done
echo "--- 双 socket 56 线程 ---"
taskset -c 0-63 $PY tools/diag/hw_gemm.py 56 2>/dev/null
echo
echo "########## S2 内存带宽 / 分配 ##########"
taskset -c 32-63 $PY tools/diag/hw_mem.py 32 2>/dev/null
taskset -c 0-63 $PY tools/diag/hw_mem.py 56 2>/dev/null
echo
echo "########## S3 forward 基线 (32t, 默认 malloc) ##########"
taskset -c 32-63 $PY tools/diag/profile_cpu2.py 32 2>&1 | grep -vE "Warning|warn|pkg_resources|\[load\]|\[data\]|STAGE"
echo
echo "########## S4 forward A/B: malloc 阈值调高 (T1) ##########"
MALLOC_MMAP_THRESHOLD_=268435456 MALLOC_TRIM_THRESHOLD_=268435456 taskset -c 32-63 $PY tools/diag/profile_cpu2.py 32 2>&1 | grep -E "eager CFG forward|eager main-only|heun_batch|GEMM 合计"
echo
echo "########## S5 attn_impl 对照 + out= 复用 (T2/T3) ##########"
taskset -c 32-63 $PY tools/diag/profile_cpu3.py 32 2>&1 | grep -vE "Warning|warn|pkg_resources|\[load\]|\[data\]"
echo
echo "HW_BENCH_DONE"
} > $OUT 2>&1
