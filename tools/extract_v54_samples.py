import os
import shutil

src_base = "/root/Workspace/xy/DiT/assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_samples_ctrl/step0100000/eval200fix"
dst_base = "/root/Workspace/xy/DiT/exp_milestones/v54_eval200fix"
os.makedirs(dst_base, exist_ok=True)

test_indices = [32, 4, 130, 162, 140, 114, 54, 144, 84, 172]
print(f"提取 v54 step100k 中的 10 个测试样本: {test_indices}")

for idx in test_indices:
    for prefix in ["g", "gt"]:
        fn = f"{prefix}{idx}.png"
        src_f = os.path.join(src_base, fn)
        dst_f = os.path.join(dst_base, fn)
        if os.path.exists(src_f):
            shutil.copy2(src_f, dst_f)
            print(f"  copied {fn}")
        else:
            print(f"  missing {fn}")

# 同时复制 eval_auto_100000.json
json_src = "/root/Workspace/xy/DiT/assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/checkpoints/eval_auto_100000.json"
if os.path.exists(json_src):
    shutil.copy2(json_src, os.path.join(dst_base, "eval_auto_100000.json"))
