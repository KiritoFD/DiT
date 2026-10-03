#!/bin/bash
cd /root/Workspace/xy/DiT || exit 1
echo "=== ZeroAdaLNInjection / 注入类定义 ==="
grep -nE 'class ZeroAdaLN|class .*Inject|class .*Injection' src/model/dit.py | head -10
echo
echo "=== adaln 注入在 forward 里的调用点 ==="
grep -nE 'glyph_inject|inj_|ZeroAdaLN|self\.glyph_injectors|modulat' src/model/dit.py | head -30
echo
echo "=== ZeroAdaLNInjection 类体 ==="
sed -n "$(grep -n 'class ZeroAdaLN' src/model/dit.py | head -1 | cut -d: -f1),+45p" src/model/dit.py
