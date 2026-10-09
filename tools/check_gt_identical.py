import os
from PIL import Image
import numpy as np

# 对比 eval_full_metrics/gt_0..9 与 eval_ours200fix/gt0..9
moyi_m = "assets/server_48_evals/moyi_eval_full_metrics"
moyi_200 = "assets/server_48_evals/moyi_12ch_eval200fix"

print("=== 检验 eval_full_metrics/gt_i 与 eval_ours200fix/gti 的一致性 ===")
for i in range(10):
    p_m = os.path.join(moyi_m, f"gt_{i}.png")
    p_200 = os.path.join(moyi_200, f"gt{i}.png")
    if os.path.exists(p_m) and os.path.exists(p_200):
        im1 = np.array(Image.open(p_m).convert("L"))
        im2 = np.array(Image.open(p_200).convert("L"))
        diff = np.abs(im1.astype(float) - im2.astype(float)).mean()
        print(f"  [{i}] diff={diff:.4f} (diff < 1 说明是完全相同的原图)")
