import os
import sys
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# 检查 dit_b_aug_v66route 的 strict_40000_0..9
dit_b_dir = "assets/server_48_evals/dit_b_aug_v66route_eval"
print("dit_b 图片检查:")
for i in range(10):
    p = os.path.join(dit_b_dir, f"strict_40000_{i}.png")
    if os.path.exists(p):
        im = Image.open(p)
        print(f"  strict_40000_{i}.png size={im.size}")

# 检查 moyi_eval_full_metrics 里的 10 张图
moyi_m_dir = "assets/server_48_evals/moyi_eval_full_metrics"
print("moyi_eval_full_metrics 图片检查:")
for i in range(10):
    p12 = os.path.join(moyi_m_dir, f"moyi_12_50k_{i}.png")
    p4 = os.path.join(moyi_m_dir, f"moyi_4_80k_{i}.png")
    pgt = os.path.join(moyi_m_dir, f"gt_{i}.png")
    print(f"  sample #{i}: 12ch exists={os.path.exists(p12)}, 4ch exists={os.path.exists(p4)}, gt exists={os.path.exists(pgt)}")
