#!/bin/bash
# fs6 stage 3: 找"甜点区"的边界，然后把赢家设置铺到其余 4 个主题。
#   stage 2 结论(沈周-行, 1000 步): lr 1e-4 0.5398 / 3e-4 0.5505 / 1e-3 0.5532，
#   两个 init 在 1e-3 已经并轨(0.5532 vs 0.5533)，且 eval 峰值就在最后一步 -> 还没到头。
#   A) 沈周-行: lr 3e-3 @1000（往上探边界） + lr 1e-3 @3000（往长探边界）
#   B) 其余 4 主题: 用已验证安全的 lr 1e-3 @2000，init 用 row_pt（起点更高）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
R=_sync_work/run_fs6.sh

bash $R train 沈周-行 row_pt 0.003 1000 s1k
bash $R train 沈周-行 row_pt 0.001 3000 s3k

for t in 伊秉绶-行 傅山-行 伊秉绶-隶 徐渭-行; do
  bash $R train "$t" row_pt 0.001 2000 s2k
done
/opt/conda/envs/cu121/bin/python _sync_work/fs6_report.py
