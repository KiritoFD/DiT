#!/usr/bin/env bash
# 造 23 槽位的 K=4 风格质心表 (从 REPA 的 patch 缓存池化 -> build_multistyle_k4)
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_purestd
LOG="exp-std/logs_purestd/k4_top10_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_purestd/k4_top10_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python

echo "### 1. patch 缓存 -> CLS npz (top10) ###"
$PY -u tools/dino_cache_to_cls_npz.py --cache data/dino_cache/top10_v1 \
    --csv exp-std/csv/train.csv --out assets/dino_feat_top10.npz --chunk 2048

echo
echo "### 2. K=4 质心 (23 槽位) ###"
$PY -u tools/build_multistyle_k4.py \
    --npz assets/dino_feat_top10.npz \
    --map exp-std/csv/callig_script_id_map_top10.json \
    --out assets/multistyle_k4_top10.pt --k 4

echo
echo "### 3. 产物体检 ###"
$PY - <<'PY'
import torch as th
d = th.load("assets/multistyle_k4_top10.pt", map_location="cpu", weights_only=False)
print("keys:", list(d.keys()))
for k in ("embedding", "centroids", "pair_mean"):
    if k in d:
        v = d[k]
        print(f"  {k}: {tuple(v.shape)}  dtype={v.dtype}  "
              f"norm_mean={float(v.reshape(v.shape[0], -1).norm(dim=1).mean()):.3f}")
if "counts" in d:
    print("  counts:", d["counts"].tolist())
if "centroids" in d:
    c = d["centroids"]                      # (P,K,D)
    import torch.nn.functional as F
    cn = F.normalize(c, dim=-1)
    for i in range(min(3, c.shape[0])):
        sim = cn[i] @ cn[i].T
        off = (sim.sum() - sim.diag().sum()) / (c.shape[1] * (c.shape[1] - 1))
        print(f"  pair{i} 的 {c.shape[1]} 个质心互相 cos 均值 = {float(off):.3f} "
              f"(越接近 0 越'分化', 越接近 1 越'塌缩')")
PY
echo "[done] $(date '+%F %T')"
