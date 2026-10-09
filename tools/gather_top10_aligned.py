import os
import shutil

base = "/root/Workspace/xy/DiT"
out_dir = "/root/Workspace/xy/DiT/exp_milestones/aligned_eval200_top10"
os.makedirs(out_dir, exist_ok=True)

# 阶段映射
stages = {
    "02_v13": "exp_milestones/eval200_outputs/02_v13",
    "03_v21": "exp_milestones/eval200_outputs/03_v21",
    "04_v23": "exp_milestones/eval200_outputs/04_v23",
    "v54": "assets/results/v54_minimal_tables_noskel/20261005-000439-v54-minimal-tables-S-noskel-freeze/eval_samples_ctrl/step0100000/eval200fix",
    "05_v66": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix",
    "06_v68": "assets/results/v68_aug_sp_c2ot/20261006-192021-v68-aug-sp-c2ot/eval_samples_ctrl/step0200000/eval200fix",
    "07_v70": "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0010000/eval200fix",
    "gt": "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix"
}

for sname, rel_p in stages.items():
    s_dst = os.path.join(out_dir, sname)
    os.makedirs(s_dst, exist_ok=True)
    src_p = os.path.join(base, rel_p)
    print(f"处理 {sname} -> {src_p}")
    
    for i in range(10):
        # 寻找对应的图片
        # 可能是 00.png, 00_*.png, g0.png, gt0.png
        cands = [
            f"g{i}.png",
            f"{i:02d}.png",
            f"gt{i}.png" if sname == "gt" else None
        ]
        # 也可能是 00_*.png
        if os.path.exists(src_p):
            for f in os.listdir(src_p):
                if f.startswith(f"{i:02d}_") and f.endswith(".png"):
                    cands.append(f)
                elif f == f"g{i}.png":
                    cands.append(f)
                elif sname == "gt" and f == f"gt{i}.png":
                    cands.append(f)
        
        found = False
        for c in cands:
            if c and os.path.exists(os.path.join(src_p, c)):
                shutil.copy2(os.path.join(src_p, c), os.path.join(s_dst, f"{i:02d}.png"))
                found = True
                break
        if not found:
            print(f"  警告: {sname} 样本 {i} 未找到匹配图片")

print("全部提取完成！")
