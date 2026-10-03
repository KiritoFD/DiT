#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== results 目录 ==="
ls -1 assets/results/ | head -40
echo
echo "=== 用 std 骨架做条件的训练配置 ==="
grep -l "shards_std\|std_skel" src/train/configs/*.json 2>/dev/null | head -12
echo
echo "=== 各 resolved_config 里的 skel 条件目录 (前 30 个) ==="
for f in $(ls -1 assets/results/*/resolved_config.json 2>/dev/null | head -30); do
  d=$(/opt/conda/envs/cu121/bin/python -c "
import json,sys
try:
    c=json.load(open('$f'))
    print(c.get('skel_latent_shards_dir','-'), '|', c.get('model','-'), '|', c.get('experiment_name','-'))
except Exception as e:
    print('ERR')
" 2>/dev/null)
  echo "  $(dirname $f | xargs basename): $d"
done
