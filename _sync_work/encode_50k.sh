#!/bin/bash
# 50k 数据集的多进程 CPU VAE 编码（GPU 被 xattn 占用，走 CPU）
# 48 进程 x 2 线程, nice 10 —— 不挤正在跑的训练
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
NP=48
TH=2

echo "[enc50k] ===== START $(date) ====="

echo "[enc50k] --- img ---"
$PY tools/cpu_encode_50k.py --csv assets/train_50k.csv --col image_path \
    --out data/50k/shards_img --nproc $NP --threads $TH --batch 8

echo "[enc50k] --- std ---"
$PY tools/cpu_encode_50k.py --csv assets/train_50k.csv --col std_path \
    --out data/50k/shards_std --nproc $NP --threads $TH --batch 8

echo "[enc50k] ===== ALL DONE $(date) ====="
