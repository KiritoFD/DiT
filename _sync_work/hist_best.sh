#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== MANIFEST 表头 ==="
head -20 assets/results/MANIFEST.md
echo
echo "=== 含 v13 的行 ==="
grep -i 'v13' assets/results/MANIFEST.md | head -12
echo
echo "=== 含 cos 或 _e 的行 ==="
grep -iE 'cos|_e\b|c41x' assets/results/MANIFEST.md | head -14
echo
echo "=== configs 里带 cos / c41x 的 ==="
ls -1 src/train/configs/ | grep -iE 'cos|c41x' | head -14
