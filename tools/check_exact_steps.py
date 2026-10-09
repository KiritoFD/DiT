import os
import glob
import csv

base = "/root/Workspace/xy/DiT"

# 检查 v10b strict50 里的 50 个 id
v10_dir = os.path.join(base, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50")
v10_ids = []
if os.path.exists(v10_dir):
    for f in os.listdir(v10_dir):
        if f.startswith("g") and not f.startswith("gt") and f.endswith(".png"):
            idx_str = f[1:-4]
            if idx_str.isdigit():
                v10_ids.append(int(idx_str))
v10_ids = sorted(v10_ids)
print(f"v10b 实际包含的 strict id ({len(v10_ids)} 个): {v10_ids[:10]} ... {v10_ids[-5:]}")

# 检查 v68 的最新 step 目录
v68_parent = os.path.join(base, "assets/results/v68_aug_sp_c2ot/20261006-180035-v68-aug-sp-c2ot/eval_samples_ctrl")
v68_steps = sorted(glob.glob(os.path.join(v68_parent, "step*")))
print(f"v68 (180035) 包含的 step 目录 ({len(v68_steps)} 个): {[os.path.basename(s) for s in v68_steps[-5:]]}")

v68_parent2 = os.path.join(base, "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl")
if os.path.exists(v68_parent2):
    v68_steps2 = sorted(glob.glob(os.path.join(v68_parent2, "step*")))
    print(f"v68 (192021) 包含的 step 目录 ({len(v68_steps2)} 个): {[os.path.basename(s) for s in v68_steps2[-5:]]}")

# 检查 v70 的最新 step 目录
v70_parent = os.path.join(base, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl")
v70_steps = sorted(glob.glob(os.path.join(v70_parent, "step*")))
print(f"v70 包含的 step 目录 ({len(v70_steps)} 个): {[os.path.basename(s) for s in v70_steps]}")
