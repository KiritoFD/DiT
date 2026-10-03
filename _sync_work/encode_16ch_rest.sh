#!/usr/bin/env bash
# 换 16ch VAE 收尾: 把剩下两类条件图编码成 16ch
#  ① GT 真迹骨架 w7 (38,583, 条件/中间监督用)   ② eval200 修正后的 std 条件图 (187, 评测用)
cd /root/Workspace/xy/DiT || exit 1
mkdir -p exp-std/logs_purestd
LOG="exp-std/logs_purestd/encode16_rest_$(date +%Y%m%d-%H%M%S).log"
exec > "$LOG" 2>&1
ln -sf "$(basename "$LOG")" exp-std/logs_purestd/encode16_rest_latest.log
echo "logfile=$LOG start=$(date '+%F %T')"
export PYTHONPATH=/root/Workspace/xy/DiT
export HF_HUB_OFFLINE=1
PY=/opt/conda/envs/cu121/bin/python

echo "### 目录命名确认 ###"
ls data/top10_style23/gt_skel_png_w7 | head -3
ls exp-std/data/std_fixed_eval200 | head -3

echo
echo "### ① GT 真迹骨架 w7 -> 16ch ###"
$PY -u tools/encode_flux_latents.py --img-dir data/top10_style23/gt_skel_png_w7 \
    --out-dir exp-std/data/shards_gtskel_flux16 --batch 64 --workers 16 --shard-size 5120

echo
echo "### ② eval200 修正 std 条件 -> 16ch ###"
$PY -u tools/encode_flux_latents.py --img-dir exp-std/data/std_fixed_eval200 \
    --out-dir exp-std/data/shards_std_flux16_fixed_eval200 --batch 64 --workers 8 --shard-size 5120

echo
echo "### 产物 ###"
for d in exp-std/data/shards_gtskel_flux16 exp-std/data/shards_std_flux16_fixed_eval200 \
         exp-std/data/shards_img_flux16 exp-std/data/shards_std_flux16; do
  echo "  $d : $(ls "$d" 2>/dev/null | wc -l) 文件, $(du -sh "$d" 2>/dev/null | cut -f1)"
done
$PY - <<'PY'
import json, glob, numpy as np
for d in ("exp-std/data/shards_gtskel_flux16", "exp-std/data/shards_std_flux16_fixed_eval200",
          "exp-std/data/shards_img_flux16", "exp-std/data/shards_std_flux16"):
    fs = sorted(glob.glob(d + "/shard_*.npz"))
    n = 0
    for f in fs:
        z = np.load(f)
        n += z["img_ids"].shape[0]
    if fs:
        z = np.load(fs[0])
        print(f"  {d}: {len(fs)} shard, 总 {n} 条, latents{z['latents'].shape} {z['latents'].dtype}")
PY
echo "[done] $(date '+%F %T')"
