#!/bin/bash
# weight_decay=0.1 对照实验
# 唯一变量: weight_decay 0.02 -> 0.1（依据 Gu et al. arXiv 2310.02664 的实测）
# 其余全部继承 v13_base_50k.json（数据/模型/LR/schedule/eval 均同）
set -u
cd /root/Workspace/xy/DiT || exit 1
export PYTHONPATH=/root/Workspace/xy/DiT
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor
PY=/opt/conda/envs/cu121/bin/python
LOGD=/root/Workspace/xy/DiT/logs/v13_series/v13_wd01
mkdir -p "$LOGD"
TS=$(date +%Y%m%d-%H%M%S)

# ⚠ 续跑：从最大的 ckpt 恢复（第一次跑到 108,300 时让给风格实验）
#   --resume-full 会恢复 step 计数 / LR / optimizer / EMA
CK=$(ls -t /root/Workspace/xy/DiT/assets/results/v13_wd01/*/checkpoints/*.pt 2>/dev/null | head -1)
if [ -z "$CK" ]; then
    echo "[wd01] 找不到续跑 ckpt，退出"
    exit 1
fi

echo "[wd01] ===== RESUME $(date) ====="
echo "[wd01] ckpt   = $CK"
echo "[wd01] config = v13_base_50k_wd01.json (wd=0.1)"
# ⚠ 必须显式 --results-dir: 配置名派生出的目录名和 base 一样(v13-base-50k),
#   不指定会把结果混进 base 的实验目录(实测踩过)
$PY -u src/train/train.py --config src/train/configs/v13_base_50k_wd01.json \
    --resume-full "$CK" \
    --results-dir assets/results/v13_wd01 \
    2>&1 | tee "$LOGD/train_$TS.log"
echo "[wd01] ===== END rc=$? $(date) ====="
