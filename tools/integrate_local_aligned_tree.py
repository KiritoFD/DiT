import os
import shutil
from PIL import Image

base = "assets/aligned_eval200_top10"

# 1. 整合 48_moyi_12ch
dst_m12 = os.path.join(base, "48_moyi_12ch")
os.makedirs(dst_m12, exist_ok=True)
src_m12 = "assets/server_48_evals/moyi_12ch_eval200fix"
for i in range(10):
    shutil.copy2(os.path.join(src_m12, f"g{i}.png"), os.path.join(dst_m12, f"{i:02d}.png"))

# 2. 整合 48_moyi_4ch
dst_m4 = os.path.join(base, "48_moyi_4ch")
os.makedirs(dst_m4, exist_ok=True)
src_m4 = "assets/server_48_evals/moyi_eval_full_metrics"
for i in range(10):
    shutil.copy2(os.path.join(src_m4, f"moyi_4_80k_{i}.png"), os.path.join(dst_m4, f"{i:02d}.png"))

# 3. 整合 48_dit_b_aug_v66route
dst_b = os.path.join(base, "48_dit_b_aug_v66route")
os.makedirs(dst_b, exist_ok=True)
src_b = "assets/server_48_evals/dit_b_aug_v66route_eval"
for i in range(10):
    shutil.copy2(os.path.join(src_b, f"strict_40000_{i}.png"), os.path.join(dst_b, f"{i:02d}.png"))

print("=== 验证 aligned_eval200_top10 目录下所有模型 ===")
models = sorted(os.listdir(base))
for m in models:
    mp = os.path.join(base, m)
    if os.path.isdir(mp):
        pngs = sorted([f for f in os.listdir(mp) if f.endswith(".png")])
        print(f"  [{m}]: {len(pngs)} 张 PNG -> {pngs}")

print("\n✓ 本地对齐树状目录准备完毕！")
