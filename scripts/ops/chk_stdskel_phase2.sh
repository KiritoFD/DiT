#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== Sp 定义 (L940-960) ==="
sed -n '940,960p' src/model/dit.py
echo
echo "=== c41x 最新 eval json 完整字段 ==="
F=$(ls 5script/results/v10b_stdskel_fame3_c41x/*/checkpoints/eval_auto_105000.json 2>/dev/null | head -1)
[ -z "$F" ] && F=$(ls 5script/results/v10b_stdskel_fame3_c41x/*/checkpoints/eval_auto_*.json | tail -1)
/opt/conda/bin/python -c "
import json
j = json.load(open('$F'))
for k in sorted(j):
    v = j[k]
    if isinstance(v, (dict, list)):
        v = str(v)[:80]
    print(f'  {k}: {v}')"
echo
echo "=== std_skel 生成脚本/工具 ==="
ls tools/ 2>/dev/null | grep -iE "std|skel" | head
grep -rn "std_skel1_latents" tools/ src/ --include="*.py" -l 2>/dev/null | head -5
echo
echo "=== fame3 数据规模 ==="
wc -l 5script/train_fame3_clean_v8.csv 2>/dev/null
head -2 5script/train_fame3_clean_v8.csv 2>/dev/null | cut -c1-120
echo
echo "=== 训练日志里 c41x 关键 (REPA/loss/EMA) ==="
grep -E "step=0(04|08|12)0000\)" /tmp/v10bstdskel3c41x_train.log 2>/dev/null | tail -3 | cut -c1-190