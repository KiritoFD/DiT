import os
import glob
import json

base = "/root/Workspace/xy/DiT"

# 检查各阶段在 4090 上 eval200 的 g0..g9
paths = {
    "v54": "assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_samples_ctrl/step0100000/eval200fix",
    "v66": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix",
    "v68": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix",
    "v70": "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0010000/eval200fix",
    "eval200_outputs_v13": "exp_milestones/eval200_outputs/02_v13",
    "eval200_outputs_v21": "exp_milestones/eval200_outputs/03_v21",
    "eval200_outputs_v23": "exp_milestones/eval200_outputs/04_v23"
}

print("=== 检查 4090 上各模型 g0..g9 的存在性 ===")
for name, rel_p in paths.items():
    full_p = os.path.join(base, rel_p)
    exists = os.path.exists(full_p)
    if exists:
        g_count = sum(1 for i in range(10) if os.path.exists(os.path.join(full_p, f"g{i}.png")))
        print(f"[{name}] 存在: {full_p} | g0..g9 命中数: {g_count}/10")
    else:
        print(f"[{name}] 不存在: {full_p}")
