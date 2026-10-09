import os
import shutil
import tarfile

base = "/root/Workspace/xy/DiT"
out_dir = "/root/Workspace/xy/DiT/exp_milestones/eval200_30_gathered"
os.makedirs(out_dir, exist_ok=True)

target_30 = [
    55, 128, 129, 22, 115, 116, 27, 172, 60, 8,   # Top 10
    46, 30, 133, 77, 164, 59, 91, 38, 160, 120,   # Mid 10
    94, 17, 43, 2, 123, 173, 177, 13, 96, 49      # Worst 10
]

stages = {
    "02_v13": "exp_milestones/eval200_outputs/02_v13",
    "03_v21": "exp_milestones/eval200_outputs/03_v21",
    "04_v23": "exp_milestones/eval200_outputs/04_v23",
    "v54": "assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_samples_ctrl/step0100000/eval200fix",
    "05_v66": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix",
    "06_v68": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix",
    "07_v70": "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0030000/eval200fix",
    "gt": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix"
}

for sname, rel_p in stages.items():
    s_dst = os.path.join(out_dir, sname)
    os.makedirs(s_dst, exist_ok=True)
    src_p = os.path.join(base, rel_p)
    print(f"收集 {sname} 从 {src_p}...")
    
    for idx in target_30:
        cands = [
            f"g{idx}.png",
            f"{idx:02d}.png",
            f"{idx}.png",
            f"gt{idx}.png" if sname == "gt" else None
        ]
        # 寻找匹配
        found = False
        if os.path.exists(src_p):
            for c in cands:
                if c and os.path.exists(os.path.join(src_p, c)):
                    shutil.copy2(os.path.join(src_p, c), os.path.join(s_dst, f"{idx}.png"))
                    found = True
                    break
        if not found:
            print(f"  ⚠ 未找到 {sname} 样本 {idx}")

# 打包
tar_path = "/root/Workspace/xy/DiT/eval200_30_gathered.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(out_dir, arcname="eval200_30_gathered")

print(f"✓ 收集完毕并打包为: {tar_path}")
