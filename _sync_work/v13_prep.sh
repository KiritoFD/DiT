#!/bin/bash
# v13 前置：等 VAE 编码跑完 -> 重建 DINO 缓存（REPA teacher）
#
# ⚠ 为什么必须重建 DINO 缓存
#   新数据集文件名是 6 位补零（data/50k/imgs/000000.png），extract_img_id 解析出
#   0,1,2,...；而旧的 data/dino_cache/base_sym_v1 是按**旧 id**（77.png -> 77）建的。
#   直接复用会让新 id 77 命中旧 id 77 的特征 —— **那是另一张图**，REPA teacher 静默错位。
#   而且新数据经过描满/去噪处理，图本身也变了，旧特征本就不可用。
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python

echo "[v13prep] ===== START $(date) ====="

# 1. 等 VAE 编码（img + std）完成
echo "[v13prep] 等 VAE 编码完成..."
for i in $(seq 1 240); do
    if grep -q "^\[std\] DONE" logs/_enc50k_gpu.log 2>/dev/null; then
        echo "[v13prep] 编码完成"
        break
    fi
    sleep 15
done
grep -E "DONE|skip" logs/_enc50k_gpu.log

# 2. 重建 DINO 缓存（51,036 张，GPU，约 10-20 分钟）
if [ -f data/dino_cache/50k_v1/meta.json ]; then
    echo "[v13prep] DINO 缓存已存在，跳过"
else
    echo "[v13prep] --- build_dino_cache ---"
    $PY tools/build_dino_cache.py \
        --csv assets/train_50k.csv \
        --out data/dino_cache/50k_v1 \
        --batch-size 128 --workers 8 --device cuda
fi

echo "[v13prep] ===== ALL DONE $(date) ====="
ls -la data/dino_cache/50k_v1/ 2>/dev/null
