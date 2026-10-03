#!/bin/bash
cd /root/Workspace/xy/DiT
for d in data/top10_style23/imgs data/top10_style23/std data/top10_style23/shards_std; do
  echo "=== $d ==="
  ls $d 2>/dev/null | head -3
  echo "   文件数: $(ls $d 2>/dev/null | wc -l)"
done
echo
echo "=== 原始图尺寸/样例 ==="
/opt/conda/envs/cu121/bin/python - <<'PY'
import glob, os
import numpy as np
from PIL import Image
for d in ["data/top10_style23/imgs", "data/top10_style23/std"]:
    fs = sorted(glob.glob(d + "/*"))[:3]
    for f in fs:
        try:
            a = np.asarray(Image.open(f).convert("L"))
            print(f"  {f}  shape={a.shape}  dtype={a.dtype}  min/max={a.min()}/{a.max()}"
                  f"  暗像素占比={(a<128).mean():.4f}")
        except Exception as e:
            print(f"  {f}  读取失败 {e}")
PY
echo
echo "=== gt_skel 生成脚本 (本地仓库) ==="
ls tools/ | grep -iE 'gt_skel|build_gt' 
