import os
import glob
import csv
import json

base = "/root/Workspace/xy/DiT"

# 1. 检查各阶段最新 checkpoint 对应的 strict 采样目录
strict_dirs = {
    "01_v10b": os.path.join(base, "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e/eval_samples_ctrl/step0390000/strict50"),
    "02_v13": os.path.join(base, "assets/results/v13_wd01/20260918-210256-v13-base-50k/eval_samples_ctrl/step0125000/strict50"),
    "03_v21": os.path.join(base, "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k/eval_samples_ctrl/step0155000/strict"),
    "04_v23": os.path.join(base, "_archive/20261003_twostage/assets_results/v23_splitnorm/eval_samples_ctrl/step0085000/strict"),
    "05_v66": os.path.join(base, "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN/eval_samples_ctrl/step0150000/eval200fix"),
    "06_v68": os.path.join(base, "assets/results/v68_aug_sp_c2ot/20261006-180035-v68-aug-sp-c2ot/eval_samples_ctrl/step0015000/eval200fix"),
    "07_v70": os.path.join(base, "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot/eval_samples_ctrl/step0005000/eval200fix")
}

print("=== 检查各阶段原始训练保存的 STRICT 采样图目录 ===")
for sid, sdir in strict_dirs.items():
    if not os.path.exists(sdir):
        # 尝试查找备用目录
        print(f"[{sid}] 目录不存在: {sdir}")
        parent = os.path.dirname(os.path.dirname(sdir)) if "step" in sdir else os.path.dirname(sdir)
        matches = glob.glob(os.path.join(parent, "step*/strict*")) + glob.glob(os.path.join(parent, "step*/eval200*"))
        print(f"   备用候选项 ({len(matches)} 个): {matches[-3:] if matches else 'None'}")
    else:
        pngs = [f for f in os.listdir(sdir) if f.endswith(".png")]
        print(f"[{sid}] 存在: {sdir} ({len(pngs)} 张 PNG)")
        # 查看文件名规则
        sample_g = [f for f in pngs if f.startswith("g") and not f.startswith("gt")][:5]
        sample_gt = [f for f in pngs if f.startswith("gt")][:5]
        print(f"   g samples: {sample_g}")
        print(f"   gt samples: {sample_gt}")
