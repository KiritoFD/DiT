#!/bin/bash
# 查落盘数据集里的退化样本 (纯黑块/空骨架)
cd /root/Workspace/xy/DiT
/opt/conda/envs/cu121/bin/python - <<'PY'
import numpy as np
for split in ("train", "val", "seen20", "strict84"):
    z = np.load(f"data/top10_style23/skel64/{split}.npz")
    img = z["img_ink"].reshape(len(z["ids"]), -1).mean(1) / 255.0
    gt = z["gt_skel"].reshape(len(z["ids"]), -1)
    g = z["std_skel"].reshape(len(z["ids"]), -1)
    print(f"=== {split}  n={len(img)} ===")
    print(f"  原迹墨比: p1={np.percentile(img,1):.4f} p50={np.percentile(img,50):.4f} "
          f"p99={np.percentile(img,99):.4f} max={img.max():.4f}")
    print(f"  原迹墨比 >0.50 (疑黑块): {(img>0.50).sum()} | >0.80: {(img>0.80).sum()}")
    print(f"  GT骨架为空: {(gt.sum(1)==0).sum()} | std骨架为空: {(g.sum(1)==0).sum()}")
PY
