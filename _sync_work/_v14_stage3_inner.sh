#!/bin/bash
# v14 stage3: 承接 stage2 160k ckpt, 冻主干微调 87 表 (内部脚本, 由 _launch_v14.sh nohup 调用)
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
TS=$(date +%Y%m%d-%H%M%S)
LOGDIR=logs/v14_serial
mkdir -p "$LOGDIR"

CKPT=assets/results/v14_style87_s2/20260918-234236-v14-style87-stage2/checkpoints/0160000.pt
if [ ! -f "$CKPT" ]; then echo "FATAL: no ckpt $CKPT"; exit 1; fi

echo "[s3] 承接: $CKPT"
echo "[s3] 开始 $(date)"
$PY -u src/train/train.py \
    --config src/train/configs/v14_style87_stage3.json \
    --resume-full "$CKPT" \
    2>&1 | tee "$LOGDIR/v14_s3_$TS.log"
echo "[s3] 结束 $(date)"

# ratio 评测
CK3=$(ls -t assets/results/v14_style87_s3/*/checkpoints/*.pt 2>/dev/null | head -1 || true)
if [ -n "$CK3" ]; then
    echo "[ratio] ckpt=$CK3"
    $PY tools/eval_diversity.py --ckpt "$CK3" \
        --inter-csv assets/train_50k_v2.csv \
        --inter-skel-shards data/50k/shards_std \
        --callig-script-map assets/callig_script_id_map.json \
        --skip-intra --inter-char 8 --inter-callig 12 --cfg 1.0 \
        --device cpu \
        2>&1 | tee "$LOGDIR/v14_ratio_$TS.log" || true
    echo "[ratio] 判据: ratio_style 基线(v13 45表)=1.18 → 目标显著 >1.18"
fi
echo "[s3] 全部完成 $(date)"
