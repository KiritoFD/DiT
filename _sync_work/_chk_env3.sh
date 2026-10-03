#!/bin/bash
echo "=== base conda list torch/xformers ==="
/opt/conda/bin/conda list -n base 2>/dev/null | grep -a -i -E 'torch|xformers|flash|cuda' | head -20
echo ""
echo "=== base pip show torch ==="
/opt/conda/bin/pip show torch 2>/dev/null | grep -a -E 'Name|Version|Location|Editable'
echo ""
echo "=== 所有 conda env 目录 ==="
ls -la /opt/conda/envs/ 2>/dev/null
echo ""
echo "=== cu121 实际位置搜索 ==="
find /opt/conda -maxdepth 3 -name 'python*' -path '*cu121*' 2>/dev/null | head
find / -maxdepth 4 -type d -name 'cu121' 2>/dev/null | head -5
echo ""
echo "=== torch2 env python 位置 ==="
ls /opt/conda/envs/torch2/ 2>/dev/null
echo ""
echo "=== 当前 s28 训练进程用的 python 和 torch ==="
ls -la /proc/$(pgrep -f 'train.py --config.*s28' | head -1)/exe 2>/dev/null
echo ""
echo "=== base pip 历史 (torch 相关) ==="
/opt/conda/bin/pip list 2>/dev/null | grep -a -i -E 'torch|xformers|flash' | head
