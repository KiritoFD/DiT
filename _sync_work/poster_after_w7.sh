#!/bin/bash
# 等 w7 臂跑完 -> 出变体对比 poster -> 转 jpg
set -u
cd /root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
while pgrep -f run_w7_arms > /dev/null; do sleep 10; done
sleep 10
echo "=== w7 臂结束, 开始出 poster ==="
$PY -u tools/poster_skel_variants.py 2>&1 | grep -vE "Warning|warn|pytree|pkg_resources" | tail -14
$PY -u tools/poster_to_jpg.py 900 _ot_scratch/skel_variants.png /tmp/skv.jpg 2>&1 | tail -1
echo POSTER_DONE
