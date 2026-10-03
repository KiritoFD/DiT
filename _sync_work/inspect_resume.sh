#!/bin/bash
# 续训前: 取当前 lr / 已记录曲线 / 最新 ckpt
cd /root/Workspace/xy/DiT || exit 1
D=exp-std/runs_purestd

echo "########## 1. 日志 ##########"
ls -lt exp-std/logs_purestd/ 2>/dev/null | head -6
L=$(ls -t exp-std/logs_purestd/*.log 2>/dev/null | head -1)
echo "最新日志 = $L"
echo "--- 尾 8 行 ---"; tail -8 "$L" 2>/dev/null

echo
echo "########## 2. 最近的 lr ##########"
grep -oE "lr[ =:]+[0-9]\.[0-9e-]+" "$L" 2>/dev/null | tail -4
grep -oE "lr=[0-9.e-]+" "$L" 2>/dev/null | tail -4

echo
echo "########## 3. 已记录的评测曲线 (bak_oldcols, 每个=一次 eval) ##########"
for f in $(ls -t $D/eval_stdskel_summary.csv.bak_oldcols* 2>/dev/null); do
  printf '%-70s ' "$(basename "$f")"; tail -1 "$f"
done

echo
echo "########## 4. 最新 ckpt ##########"
ls -t $D/*p1.0/checkpoints/*.pt 2>/dev/null | head -3
echo "最新目录 = $(cat $D/_active_ckpt_dir.txt 2>/dev/null)"
