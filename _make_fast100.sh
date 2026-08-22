#!/bin/bash
# 从 final_eval.csv 切固定 100 张子集（表头 + 前 100 行），供快速 eval
cd /root/Workspace/xy/DiT
head -n 101 final_eval.csv > fast100.csv
echo "rows: $(wc -l < fast100.csv) (含表头)"
head -3 fast100.csv
