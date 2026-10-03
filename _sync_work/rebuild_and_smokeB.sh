#!/usr/bin/env bash
# 1) 重建 512 维 PCA 正交 K4 资产 (384->512 线性插值后再正交化)
# 2) 只冒烟 B 路 (关 compile 省 87 秒)
set -u
cd /root/Workspace/xy/DiT || exit 1
PY=/opt/conda/envs/cu121/bin/python
export PYTHONPATH=/root/Workspace/xy/DiT
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1

echo "########## 1. 重建 512 维 PCA 正交 K4 ##########"
CUDA_VISIBLE_DEVICES= $PY -u tools/build_multistyle_pca_k4.py \
    --dim 512 --out assets/pca_multistyle_k4_top10_d512.pt 2>&1 | tail -10
echo
echo "########## 2. 形状核对 ##########"
CUDA_VISIBLE_DEVICES= $PY - <<'PY'
import torch as th
for p in ("assets/pca_multistyle_k4_top10_d512.pt",):
    d = th.load(p, map_location="cpu", weights_only=False)
    print(p)
    for k in ("embedding", "centroids", "pair_mean"):
        if k in d:
            print("   ", k, tuple(d[k].shape))
    c = d["centroids"].float()
    n = c / c.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    s = n @ n.transpose(1, 2)
    off = ~th.eye(c.shape[1], dtype=th.bool)
    print(f"    正交性: mean cos={float(s[:, off].mean()):.6f} "
          f"max|cos|={float(s[:, off].abs().max()):.6f}")
print("期望: embedding (23, 2048) = 23 x (4*512); pair_mean (23, 512); mean cos ≈ 0")
PY
echo
echo "########## 3. 冒烟 B 路 ##########"
LOG=exp-std/logs_smoke/smoke_B2.log
mkdir -p exp-std/logs_smoke
timeout 900 $PY -u src/train/train.py \
    --config src/train/configs/v51_B_joint_kv_k4_top10.json \
    --experiment-name smoke_B2 --results-dir exp-std/runs_smoke \
    --global-batch-size 32 --skel-latent-shards-weights 1.0,0.0 \
    --max-steps 40 --lr 5e-5 --compile false > "$LOG" 2>&1
RC=$?
echo "rc=$RC  log=$LOG"
echo "--- 关键行 ---"
grep -aE 'style-ortho|style-anchor|callig-emb|Traceback|RuntimeError|ValueError|Error|错误' "$LOG" | head -20
echo "--- 步进 ---"
grep -aE 'step=|Reached max_steps|Done' "$LOG" | tail -5
if [ "$RC" -ne 0 ]; then echo "--- 尾部 ---"; tail -20 "$LOG"; fi
