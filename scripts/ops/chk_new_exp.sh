#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== 全部 stdskel results 目录 (按时间) ==="
ls -dt assets/results/*stdskel* 2>/dev/null
echo
echo "=== 每个目录的 ckpt 数 / 最新 eval 点 ==="
for d in $(ls -dt assets/results/*stdskel* 2>/dev/null); do
  n=$(ls $d/*/checkpoints/*.pt 2>/dev/null | wc -l)
  last_ckpt=$(ls $d/*/checkpoints/[0-9]*.pt 2>/dev/null | sed 's|.*/||; s|\.pt||' | sort -n | tail -1)
  ne=$(ls $d/*/checkpoints/eval_auto_*.json 2>/dev/null | wc -l)
  echo "$d : ${n}ckpts last=$last_ckpt evals=$ne"
done
echo
echo "=== 最新 config v10b_stdskel_fame3_c41x.json ==="
/opt/conda/bin/python -c "
import json
c = json.load(open('src/train/configs/v10b_stdskel_fame3_c41x.json', encoding='utf-8'))
for k in sorted(c):
    v = str(c[k])
    if len(v) > 60: v = v[:60]+'...'
    print(f'  {k}: {v}')"
echo
echo "=== 各变体 config 差异 (相对 base c41) ==="
/opt/conda/bin/python -c "
import json
base = json.load(open('src/train/configs/v10b_stdskel_fame3_c41.json', encoding='utf-8'))
import glob
for p in sorted(glob.glob('src/train/configs/v10b_stdskel_fame3_*.json')):
    if 'c41.json' in p: continue
    c = json.load(open(p, encoding='utf-8'))
    diffs = {k: (base.get(k,'<absent>'), v) for k, v in c.items() if base.get(k) != v and not k.startswith('_')}
    print(p.split('/')[-1], ':', diffs)"