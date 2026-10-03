#!/bin/bash
# ?????? loss ????? (stage1 ? shards -> stage2 ? probe -> stage3 ?? v17)
# ? stage1/2 ???? GPU ?? ??????????, ? GPU ?????
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
PY=/opt/conda/envs/cu121/bin/python
CSV=assets/train_fame-kxl-tj-px60.csv
IMG_ROOT=data/fame-kxl-tj-px60/imgs
SKEL1=data/fame-kxl-tj-px60/inst_skel1
SKEL3=data/fame-kxl-tj-px60/inst_skel3
OUT=data/aux/inst_skel_latents_px60
PROBE=assets/structure_probes/latent_skel_probe_v1/best.pt

if pgrep -f 'train.py --config' > /dev/null; then
    echo "[pipeline] ??????, GPU ??? -> ?? stage1 (???????)???????????"
    exit 1
fi

echo "=== stage1: GT ? -> 1px ?? -> 3px ?? PNG -> VAE encode -> shards ($OUT) ==="
$PY tools/build_skel_latents.py \
    --csv $CSV --img-root $IMG_ROOT \
    --skel1-dir $SKEL1 --skel3-dir $SKEL3 \
    --latent-out $OUT 2>&1 | tee /tmp/inst_skel_stage1.log

echo "=== stage2: ? LatentSkelProbe (img latent -> inst skel latent, ??????? loss) ==="
mkdir -p "$(dirname $PROBE)"
$PY tools/train_latent_skel_probe.py \
    --latent-shards-dir data/fame-kxl-tj-px60/shards_img \
    --skel-shards-dir $OUT \
    --out $PROBE --width 64 --depth 3 --epochs 6 --batch 256 --lr 3e-4 \
    2>&1 | tee /tmp/inst_skel_stage2.log

echo "=== stage3: ?? v17 (resume from v12 last ckpt) ==="
CKPT=$(ls -t /root/Workspace/xy/DiT/assets/results/v12_pretrain_S_cat_fame_kxl_tj_px60/*/checkpoints/[0-9]*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "[pipeline] ??? v12 ckpt"; exit 1; fi
echo "[pipeline] resume from: $CKPT"
TS=$(date +%Y%m%d-%H%M%S)
LOGD=/root/Workspace/xy/DiT/logs/v17_pretrain_S_cat_instskel_fame_kxl_tj_px60
mkdir -p $LOGD
mkdir -p /root/Workspace/xy/DiT/assets/registry/configs
cp /root/Workspace/xy/DiT/src/train/configs/v17_pretrain_S_cat_instskel_fame_kxl_tj_px60.json \
   /root/Workspace/xy/DiT/assets/registry/configs/ 2>/dev/null || true
tmux kill-session -t v17cont 2>/dev/null
tmux new-session -d -s v17cont "export PYTHONPATH=/root/Workspace/xy/DiT; export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor; $PY -u src/train/train.py --config src/train/configs/v17_pretrain_S_cat_instskel_fame_kxl_tj_px60.json --resume-full $CKPT 2>&1 | tee $LOGD/train_v17_$TS.log"
sleep 5
tmux ls
echo "[pipeline] v17 ?? tmux:v17cont, ??: $LOGD/train_v17_$TS.log"