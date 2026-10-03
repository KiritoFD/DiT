#!/bin/bash
cd /root/Workspace/xy/DiT
echo "=== train_fame_clean.csv: 路径分布 ==="
cut -d, -f1 assets/train_fame_clean.csv | tail -n +2 | sed 's|/[^/]*$||' | sort | uniq -c
echo ""
echo "=== 有多少行指向 clean 目录 ==="
grep -c 'clean' assets/train_fame_clean.csv 2>/dev/null
echo "=== eval clean 路径分布 ==="
cut -d, -f1 assets/eval_fame_strict_clean.csv | tail -n +2 | sed 's|/[^/]*$||' | sort | uniq -c
echo ""
echo "=== 现有 latent shards 构建方式 ==="
ls -la data/latents/final_latents_fame/ 2>/dev/null | head -5
echo "=== 找 latent 构建脚本 ==="
grep -rl 'latent_shards\|npz\|shard_' tools/*.py 2>/dev/null | head -5
echo "=== build_fame_dataset 或类似 ==="
ls tools/build_*.py | head -10
