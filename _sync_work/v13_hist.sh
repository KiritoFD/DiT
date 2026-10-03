#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== results 里的 v13 ==="
ls -d assets/results/*v13* 2>/dev/null
for d in assets/results/*v13*; do
  echo "--- $d"
  ls "$d" | head -6
  for f in "$d"/*summary*.csv "$d"/*/eval*.csv; do
    [ -f "$f" ] && { echo "  [$f]"; head -2 "$f"; }
  done
done
echo
echo "=== 实验记录里 v13 的成绩 ==="
grep -rniE 'v13|strict 0\.5[0-9]' docs/04_experiments/*.md 2>/dev/null | head -8
grep -rniE 'v13' docs/experiments/*.md 2>/dev/null | head -8
echo
echo "=== 谁是最好的 std-skel 主干 (docs 里) ==="
grep -rniE 'best|最好|最佳' docs/STATUS_2026-09-30.md 2>/dev/null | head -10
