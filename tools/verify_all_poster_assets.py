import os
import glob
from PIL import Image

indices = [32, 4, 130, 162, 140, 114, 54, 144, 84, 172]
chars = ["冠", "豈", "出", "典", "悟", "照", "鼓", "呼", "兩", "其"]

print("=== 检查本地所有对比模型的图片可用性 ===")

# 1. v13 / v23 / v66 / v68 / v70 在 eval200_outputs 或 assets 中的路径
# 我们先看 eval200_outputs
e200_dirs = glob.glob("eval200_outputs/*")
print("eval200_outputs 子目录:", [os.path.basename(d) for d in e200_dirs])

# 2. 检查 v54
v54_dir = "assets/v54_eval200fix"
print("v54 目录存在:", os.path.exists(v54_dir))
for idx in indices:
    gp = os.path.join(v54_dir, f"g{idx}.png")
    gtp = os.path.join(v54_dir, f"gt{idx}.png")
    if not os.path.exists(gp):
        print(f"  v54 缺失 g{idx}.png")

# 3. 检查 48 moyi_12ch
moyi_dir = "assets/server_48_evals/moyi_12ch_eval200fix"
print("moyi_12ch 目录存在:", os.path.exists(moyi_dir))
for idx in indices:
    gp = os.path.join(moyi_dir, f"g{idx}.png")
    if not os.path.exists(gp):
        print(f"  moyi_12ch 缺失 g{idx}.png")

# 4. 检查 48 dit_b_aug_v66route
dit_b_dir = "assets/server_48_evals/dit_b_aug_v66route_eval"
print("dit_b 目录存在:", os.path.exists(dit_b_dir))
for col_i in range(10):
    gp = os.path.join(dit_b_dir, f"strict_40000_{col_i}.png")
    if not os.path.exists(gp):
        print(f"  dit_b 缺失 strict_40000_{col_i}.png")

print("✓ 预检完成！")
