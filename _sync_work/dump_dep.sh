#!/usr/bin/env bash
# 取 ZeroAdaLNInjection 的精确实现 + 全部 legacy.controlnet 引用点
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "════ [1] legacy/controlnet.py 里的类/函数清单 ════"
grep -nE '^(class |def )' src/model/legacy/controlnet.py | sed 's/^/  /'

echo
echo "════ [2] ZeroAdaLNInjection + zero_init_linear 全文 ════"
sed -n '60,100p' src/model/legacy/controlnet.py | cat -n | sed 's/^/  /'

echo
echo "════ [3] src/model/__init__.py 的 legacy 引用上下文 (20~55) ════"
sed -n '20,55p' src/model/__init__.py | cat -n | sed 's/^/  /'

echo
echo "════ [4] 全仓 'legacy' 引用点 (排除 legacy 目录自身) ════"
grep -rn "legacy" --include=*.py src tools 2>/dev/null | grep -v '/legacy/' | head -20 | sed 's/^/  /'

echo
echo "════ [5] dit.py 里那处 import 的上下文 (1670~1685) ════"
sed -n '1670,1685p' src/model/dit.py | cat -n | sed 's/^/  /'

echo
echo "════ [6] 谁在用 ZeroAdaLNInjection / ControlNetDiT / load_main_model ════"
for n in ZeroAdaLNInjection ControlNetDiT ControlConditionEncoder load_main_model zero_init_linear; do
  c=$(grep -rn "\b$n\b" --include=*.py src tools 2>/dev/null | grep -v '/legacy/' | wc -l)
  printf '  %-26s 非 legacy 引用数: %s\n' "$n" "$c"
  grep -rn "\b$n\b" --include=*.py src tools 2>/dev/null | grep -v '/legacy/' | head -3 | sed 's/^/      /'
done
