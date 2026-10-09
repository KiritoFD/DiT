import os
import glob
import csv
import json

base = "/root/Workspace/xy/DiT"

# 检查各个阶段的 eval csv / samples 命名
stages = [
    ("v10b", "assets/results/v10b_stdskel_fame3_c41x_cos_e/20260909-193018-v10b-stdskel-fame3-c41x-cos-e"),
    ("v13", "assets/results/v13_wd01/20260918-210256-v13-base-50k"),
    ("v21", "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k"),
    ("v23", "_archive/20261003_twostage/assets_results/v23_splitnorm"),
    ("v66", "assets/results/v66_tables_condroute2456/20261006-024102-v66-tables-condroute2456-adaLN"),
    ("v68", "assets/results/v68_aug_sp_c2ot/20261006-180035-v68-aug-sp-c2ot"),
    ("v70", "assets/results/v70_aug_sp_stdskel_c2ot/20261007-141413-v70-aug-sp-stdskel-c2ot")
]

print("=== 检查各阶段历史目录中的具体 eval 文件夹与文件名 ===")
for name, rel_path in stages:
    p = os.path.join(base, rel_path)
    if not os.path.exists(p):
        # 尝试 glob
        matches = glob.glob(os.path.join(base, rel_path + "*"))
        if matches:
            p = matches[0]
        else:
            print(f"[{name}] 目录不存在: {p}")
            continue

    print(f"\n--- [{name}] {p} ---")
    # 查找所有的 eval 子目录
    eval_dirs = glob.glob(os.path.join(p, "**/*eval*"), recursive=True)
    poster_dirs = glob.glob(os.path.join(p, "**/*poster*"), recursive=True)
    all_dirs = set(d for d in eval_dirs + poster_dirs if os.path.isdir(d))
    for d in sorted(all_dirs):
        files = [f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            print(f"  Dir: {os.path.relpath(d, p)} ({len(pngs)} pngs) -> sample: {pngs[:5]}")
