#!/usr/bin/env bash
# 1) 焊 expandable_segments 代码闸门  2) 验证  3) 找 A 路可复用的 ckpt  4) 启动 A/B
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python

echo "################ [1] 焊闸门 ################"
$PY -u _sync_work/alloc_gate_patch.py 2>&1 | tail -8
echo
echo "################ [2] 验证闸门 + 传参生效 ################"
grep -n 'alloc-gate' src/train/train.py | head -5
echo "--- 实测: 带脏环境变量跑一次, 应打印剔除并继续 ---"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY -c "
import sys; sys.path.insert(0,'/root/Workspace/xy/DiT')
import os
exec(open('src/train/train.py',encoding='utf-8').read().split('from src.model import')[0].split('from src.train.cli')[0])
print('  环境变量现状:', os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '<已剔除 ✓>'))
" 2>&1 | tail -5

echo
echo "################ [3] 找 A 路可复用的 ckpt ################"
CK=$(ls -t exp-std/runs_AB/*v50-A*/checkpoints/*.pt 2>/dev/null | head -1)
echo "找到: ${CK:-<无>}"
if [ -n "${CK:-}" ]; then
  ST=$(basename "$CK" .pt | sed 's/^0*//')
  # cosine(warmup=3000, total=100000, lr0=5e-5, min_ratio=0.1) 在 step=ST 处的值
  LR=$($PY -c "
import math
st=$ST; lr0=5e-5; warm=3000; tot=100000; minr=0.1
p=(st-warm)/max(tot-warm,1); p=min(max(p,0),1)
print('%.3e' % (lr0*(minr+(1-minr)*0.5*(1+math.cos(math.pi*p)))))")
  echo "  step=$ST  该点 cosine lr=$LR"
else
  ST=""; LR=""
fi

echo
echo "################ [4] 启动 (后台 setsid) ################"
mkdir -p exp-std/logs_AB
if [ -n "${CK:-}" ]; then
  RESUME_CKPT="$CK" RESUME_LR="$LR" nohup setsid bash _sync_work/launch_AB_100k.sh \
      > exp-std/logs_AB/launch_r3.log 2>&1 < /dev/null &
else
  nohup setsid bash _sync_work/launch_AB_100k.sh \
      > exp-std/logs_AB/launch_r3.log 2>&1 < /dev/null &
fi
echo "  已启动, 等 150 秒看起步"
sleep 150
echo
echo "--- launcher 日志 ---"
grep -aE 'resume|preflight|run\]|FAILED|FATAL|alloc-gate' exp-std/logs_AB/launch_r3.log | head -12
echo "--- GPU ---"
nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
echo "--- A 日志尾 ---"
A=$(ls -t exp-std/logs_AB/A_*.log 2>/dev/null | head -1)
tail -4 "$A" 2>/dev/null | sed 's/\x1b\[[0-9;]*m//g'
