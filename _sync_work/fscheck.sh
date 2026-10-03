#!/bin/bash
cd /root/Workspace/xy/DiT
C=${1:-伊秉绶}
L=$(ls -1t logs/v15_series/v15_fewshot/${C}_train_*.log | head -1)
echo "LOG=$L"
grep -E 'set=fewshot' "$L" | tail -12
grep -E '\(step=' "$L" | tail -2