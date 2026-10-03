#!/bin/bash
# launch_sym2.sh — 等 image latent 编码真正结束后, 自动拉起 sym2 从头训练.
#
# 依赖: data/latents/final_latents_base_sym 必须编码完整, 否则训练会因缺 shard 静默丢样本。
set -u
cd /root/Workspace/xy/DiT || exit 1

echo "[chain] $(date '+%F %T') 等待 encode_base_sym 结束 ..."
for i in $(seq 1 180); do
  if ! pgrep -f encode_base_sym > /dev/null 2>&1; then
    echo "[chain] encode 进程已退出 (等待 $((i*20))s)"
    break
  fi
  sleep 20
done

# 再确认 shard 数与日志尾部
echo "[chain] encode 日志尾部:"
tail -3 /tmp/enc2.log
NSH=$(ls data/latents/final_latents_base_sym 2>/dev/null | wc -l)
echo "[chain] latent shards = $NSH"

# csv 行数 vs shard 覆盖数 粗校验 (每 shard 5000)
ROWS=$(python3 -c "import csv;print(sum(1 for _ in csv.DictReader(open('assets/train_base_sym.csv',encoding='utf-8'))))" 2>/dev/null || echo 0)
echo "[chain] csv rows = $ROWS  (期望 shards >= $((ROWS/5000)) )"
if [ "$NSH" -lt $((ROWS/5000)) ]; then
  echo "[chain] !! shard 数不足, 放弃启动, 请人工检查"
  exit 2
fi

sleep 5
export PYTHONPATH=/root/Workspace/xy/DiT
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHINDUCTOR_CACHE_DIR=/root/Workspace/xy/.inductor_cache
mkdir -p logs/v11_pretrain_Sp2_base_sym2
echo "[chain] $(date '+%F %T') 启动 sym2 从头训练"
exec /opt/conda/envs/cu121/bin/python -u src/train/train.py \
  --config src/train/configs/v11_pretrain_Sp2_base_sym2.json
