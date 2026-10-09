import os
import glob

base = "/root/Workspace/xy/DiT"
targets = {
    "v10b": "assets/results/v10b_stdskel_fame3_c41x_cos_e",
    "v13": "assets/results/v13_wd01",
    "v21": "archive_experiments/results_archive_failed/failed_skelnet_runs/v21_skelnet_200k",
    "v23": "_archive/20261003_twostage/assets_results/v23_splitnorm",
    "v66": "assets/results/v66_tables_condroute2456",
    "v68": "assets/results/v68_aug_sp_c2ot",
    "v70": "assets/results/v70_aug_sp_stdskel_c2ot"
}

for name, rel in targets.items():
    p = os.path.join(base, rel)
    print(f"\n=== [{name}] {p} ===")
    png_dirs = set()
    for root, dirs, files in os.walk(p):
        pngs = [f for f in files if f.endswith(".png")]
        if pngs:
            png_dirs.add((root, len(pngs)))
    # 打印包含 png 的子目录
    sorted_dirs = sorted(list(png_dirs), key=lambda x: x[0])
    for d, count in sorted_dirs[:10]:
        print(f"  {os.path.relpath(d, p)}: {count} 张, 样本: {os.listdir(d)[:4]}")
    if len(sorted_dirs) > 10:
        print(f"  ... 共有 {len(sorted_dirs)} 个包含 PNG 的子目录，包括最后一个: {sorted_dirs[-1]}")
