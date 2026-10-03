#!/usr/bin/env bash
# 全吃满流水线: 停训练 -> 全量编码 16ch latent(GT + std) -> 全预算信号竞技场
# 日志: exp-std/logs_purestd/flux_pipeline_<ts>.log   (结果另外落在 summary.txt, 不依赖 stdout)
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_purestd
LOG="exp-std/logs_purestd/flux_pipeline_$(date +%Y%m%d-%H%M%S).log"
# 自记日志: 用 tmux/setsid 启动时无需外部重定向, 日志路径自己写进 head
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_purestd/flux_pipeline_latest.log
echo "logfile=$LOG  start=$(date '+%F %T')"
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1

echo "########## 1. 停训练 + 停旧竞技场/编码 ##########"
pkill -f 'src/train/train.py' && echo "train.py 已停" || echo "无 train.py"
pkill -f 'latent_signal_arena.py' && echo "旧竞技场已停" || echo "无旧竞技场"
pkill -f 'encode_flux_latents.py' && echo "旧编码器已停" || echo "无旧编码器"
sleep 10
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
rm -rf exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16   # 清掉半成品

echo
echo "########## 2. 全量编码 GT 真迹图 -> 16ch (batch 64, 防 OOM) ##########"
$PY tools/encode_flux_latents.py --img-dir data/top10_style23/imgs \
    --out-dir exp-std/data/shards_img_flux16 --batch 64 --workers 16 --shard-size 5120

echo
echo "########## 3. 全量编码 std 条件图 -> 16ch ##########"
$PY tools/encode_flux_latents.py --img-dir data/top10_style23/std \
    --out-dir exp-std/data/shards_std_flux16 --batch 64 --workers 16 --shard-size 5120

echo
echo "########## 4. 全预算信号竞技场 ##########"
$PY tools/latent_signal_arena.py \
    --n-triplets 20000 --n-unique 45000 --n-pairs 4000 \
    --swd-proj 128 --steps 300 --targets 8 --seeds 2 --enc-batch 64 \
    --flux-shards exp-std/data/shards_img_flux16 \
    --out-dir exp-std/signal_arena_full

echo
echo "########## 5. 结果 ##########"
cat exp-std/signal_arena_full/summary.txt 2>/dev/null
ls -la exp-std/signal_arena_full/ 2>/dev/null
echo "[pipeline] 全部完成 $(date '+%F %T')"
