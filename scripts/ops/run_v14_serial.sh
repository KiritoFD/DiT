#!/bin/bash
# ============================================================================
# v14 style87 三阶段串行 (一个 tmux 前台顺序跑完)
#   prep   : 87 书家×书体词表 + DINO CLS + 层级 SupCon 表 (快, ~3min)
#   stage2 : 冻结 87 表, 训主干 250k 步 (wd=0.1, ~17h @4.09 step/s)
#   stage3 : 冻结主干, 微调 87 表 40k 步 (lr=3e-5, 锚定 λ=0.005, ~2.7h)
#   ratio  : CPU 测 ratio_style (判据: 基线 1.18 → 目标 >1.18 向 5-10 靠)
#
# 启动 (远端):
#   tmux new-session -d -s v14 \
#     "bash /root/Workspace/xy/DiT/scripts/ops/run_v14_serial.sh 2>&1 | tee /tmp/v14_serial.log"
#
# 前置 (已完成, 本脚本会跳过):
#   assets/callig_script_id_map.json   (87 类词表)
#   assets/dino_cls_50k.npz            (DINO CLS 51036)
#   assets/callig_script_emb_pretrained.pt  (87×128 SupCon 表)
# ============================================================================
set -euo pipefail
cd /root/Workspace/xy/DiT
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
TS0=$(date +%Y%m%d-%H%M%S)
LOGDIR=logs/v14_serial
mkdir -p "$LOGDIR"

# ---- GPU 空闲检查 ----
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$GPU_USED" -gt 500 ]; then
    echo "[FATAL] GPU 非空闲 (used=${GPU_USED}MiB)。先停所有训练再跑本脚本。"
    exit 1
fi
echo "[serial] GPU 空闲 (${GPU_USED}MiB) ✓"

# ---- prep 资产检查 (已提前跑好则跳过) ----
if [ -f assets/callig_script_id_map.json ] && \
   [ -f assets/dino_cls_50k.npz ] && \
   [ -f assets/callig_script_emb_pretrained.pt ]; then
    echo "[serial] prep 资产已就绪, 跳过 prep"
else
    echo "[serial] ===== prep (建词表+DINO+SupCon表) ====="
    bash scripts/ops/run_v14_style87_3stage.sh prep
fi

# ---- stage2: 冻 87 表, 训主干 250k ----
echo "[serial] ===== stage2 开始 $(date) ====="
$PY -u src/train/train.py --config src/train/configs/v14_style87_stage2.json \
    2>&1 | tee "$LOGDIR/v14_s2_$TS0.log"
echo "[serial] ===== stage2 结束 $(date) ====="

# ---- 找 stage2 最新 ckpt ----
CK2=$(ls -t assets/results/v14_style87_s2/*/checkpoints/*.pt 2>/dev/null | head -1 || true)
if [ -z "$CK2" ]; then
    echo "[serial] FATAL: stage2 无 ckpt, 终止"
    exit 1
fi
echo "[serial] stage2 best ckpt: $CK2"

# ---- stage3: 冻主干, 微调 87 表 40k ----
echo "[serial] ===== stage3 开始 $(date) ====="
$PY -u src/train/train.py --config src/train/configs/v14_style87_stage3.json \
    --resume-full "$CK2" --fresh-scheduler \
    2>&1 | tee "$LOGDIR/v14_s3_$TS0.log"
echo "[serial] ===== stage3 结束 $(date) ====="

# ---- ratio 评测 (判据) ----
CK3=$(ls -t assets/results/v14_style87_s3/*/checkpoints/*.pt 2>/dev/null | head -1 || true)
if [ -n "$CK3" ]; then
    echo "[serial] ===== ratio 评测 (ckpt=$CK3) ====="
    $PY tools/eval_diversity.py --ckpt "$CK3" \
        --inter-csv assets/train_50k_v2.csv \
        --inter-skel-shards data/50k/shards_std \
        --callig-script-map assets/callig_script_id_map.json \
        --skip-intra --inter-char 8 --inter-callig 12 --cfg 1.0 \
        --device cpu \
        2>&1 | tee "$LOGDIR/v14_ratio_$TS0.log" || true
    echo "[serial] ratio_style 应显著 > 1.18 (基线), 向 5-10 靠"
else
    echo "[serial] WARN: stage3 无 ckpt, 跳过 ratio"
fi

echo "[serial] ===== 全部完成 $(date) ====="
echo "[serial] stage2 eval: assets/results/v14_style87_s2/*/eval_stdskel_summary.csv"
echo "[serial] stage3 eval: assets/results/v14_style87_s3/*/eval_stdskel_summary.csv"
