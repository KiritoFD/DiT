import os
import shutil
import tarfile

base = "/root/Workspace/xy/DiT"
gather_dir = "/root/Workspace/xy/DiT/eval_export"
os.makedirs(gather_dir, exist_ok=True)

stages = {
    "v68": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix",
    "v66": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix",
    "v54": "assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_samples_ctrl/step0100000/eval200fix",
    "v70": "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0030000/eval200fix",
    "gt": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix",
    "v13": "exp_milestones/eval200_outputs/02_v13",
    "v21": "exp_milestones/eval200_outputs/03_v21",
    "v23": "exp_milestones/eval200_outputs/04_v23",
}

for sname, rel_p in stages.items():
    src_p = os.path.join(base, rel_p)
    dst_p = os.path.join(gather_dir, sname)
    os.makedirs(dst_p, exist_ok=True)
    print(f"正在打包 {sname} 从 {src_p}...")
    
    if not os.path.exists(src_p):
        print(f"  ⚠ 目录不存在: {src_p}")
        continue
        
    copied = 0
    for idx in range(187):
        if sname == "gt":
            fn = f"gt{idx}.png"
        else:
            fn = f"g{idx}.png"
            
        src_f = os.path.join(src_p, fn)
        if not os.path.exists(src_f):
            # 尝试其他命名
            for alt in [f"{idx}.png", f"{idx:02d}.png"]:
                alt_f = os.path.join(src_p, alt)
                if os.path.exists(alt_f):
                    src_f = alt_f
                    break
                    
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(dst_p, f"{idx}.png"))
            copied += 1
            
    print(f"  -> 成功复制 {copied}/187 张到 {dst_p}")

# 打包
tar_path = "/root/Workspace/xy/DiT/eval_all_models.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(gather_dir, arcname="eval")

print(f"✓ 打包完成: {tar_path} (大小: {os.path.getsize(tar_path)/1024/1024:.2f} MB)")
