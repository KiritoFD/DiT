#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
echo "=== 1. ZeroAdaLNInjection 定义 (legacy/controlnet.py) ==="
L=$(grep -n 'class ZeroAdaLNInjection' src/model/legacy/controlnet.py 2>/dev/null | head -1 | cut -d: -f1)
echo "行号=$L"
if [ -n "$L" ]; then sed -n "${L},$((L+55))p" src/model/legacy/controlnet.py; fi

echo
echo "=== 2. 注入器构造处 (dit.py 1640-1660) ==="
sed -n '1640,1662p' src/model/dit.py

echo
echo "=== 3. forward 里注入器的消费点 ==="
grep -n 'glyph_injections\|_inj_map\|glyph_inject_at\|g_tok' src/model/dit.py | sed -n '1,40p'

echo
echo "=== 4. forward 里 block 循环附近 (找注入调用) ==="
L2=$(grep -n 'for i, block in enumerate' src/model/dit.py | head -1 | cut -d: -f1)
echo "循环行号=$L2"
if [ -n "$L2" ]; then sed -n "$((L2-6)),$((L2+40))p" src/model/dit.py; fi
