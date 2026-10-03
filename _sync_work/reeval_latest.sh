#!/bin/bash
# 只重评"我们刚跑的 run": 找今天最新的 ckpt -> 统一口径评 eval200+seen -> 出 poster
cd /root/Workspace/xy/DiT || exit 1

echo "########## 1. 今天的 ckpt (按时间) ##########"
find exp-std assets/results exp -name '*.pt' -newermt '2026-10-03 00:00' 2>/dev/null \
  | xargs -r ls -lt 2>/dev/null | head -12

echo
echo "########## 2. 今天的 run 目录 ##########"
find exp-std assets/results exp -maxdepth 3 -name 'checkpoints' -type d \
  -newermt '2026-10-03 00:00' 2>/dev/null | head -8

echo
echo "########## 3. 最近日志 ##########"
ls -lt _sync_work/*.log logs/*.log 2>/dev/null | head -4

CK=$(find exp-std assets/results exp -name '*.pt' -newermt '2026-10-03 00:00' 2>/dev/null \
     | xargs -r ls -t 2>/dev/null | grep -vE 'best_|_bak' | head -1)
echo
echo ">>> 选中 ckpt = $CK"
if [ -z "$CK" ]; then echo "没找到今天的 ckpt"; exit 1; fi

echo
echo "########## 4. 重评 (eval200 修正条件 + seen) ##########"
PYTHONPATH=. HF_HUB_OFFLINE=1 timeout 2400 /opt/conda/envs/cu121/bin/python -u \
  tools/reeval_ckpts.py --ckpt "$CK" --tag latest 2>&1 \
  | grep -vE 'Warning|warn|pkg_resources|FutureWarning|_torch_pytree|_register_pytree' | tail -30

echo
echo "########## 5. 产物 (poster/指标) ##########"
find exp-std/reeval -newermt '-40 minutes' -type f 2>/dev/null | head -20
echo "--- master ---"
cat exp-std/eval_master.csv 2>/dev/null | head -6
