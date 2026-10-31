#!/bin/bash
# Enumerate every eval panel dir on this host with its g/gt counts and step.
cd /root/Workspace/xy/DiT || exit 1
find assets scratch -type d -name eval200fix 2>/dev/null | while read -r d; do
  n=$(ls "$d"/g*.png 2>/dev/null | grep -cE '/g[0-9]+\.png$')
  gt=$(ls "$d"/gt*.png 2>/dev/null | grep -cE '/gt[0-9]+\.png$')
  [ "$n" -ge 20 ] || continue
  printf '%4d g %4d gt  %s\n' "$n" "$gt" "$d"
done | sort -k1,1nr | awk '{print}' | head -40
