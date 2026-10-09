import os
import glob

base = "/root/Workspace/xy/DiT"
patterns = [
    "assets/results/v10*",
    "assets/results/v13*",
    "assets/results/*v13*",
    "archive_experiments/results_archive_failed/failed_skelnet_runs/v21*",
    "_archive/20261003_twostage/assets_results/v23*",
    "assets/results/v23*",
    "assets/results/v66*",
    "assets/results/v68*",
    "assets/results/v70*",
    "assets/full_eval_raw/*",
    "assets/ink_eval_raw/*",
    "exp-std/reeval/*"
]

print("=== 扫描所有历史阶段的原始评测输出与采样图 ===")
for p in patterns:
    full_p = os.path.join(base, p)
    matches = glob.glob(full_p)
    for m in matches:
        if not os.path.isdir(m):
            continue
        pngs = glob.glob(os.path.join(m, "**/*.png"), recursive=True)
        csvs = glob.glob(os.path.join(m, "**/*.csv"), recursive=True)
        if pngs or csvs:
            print(f"[{os.path.basename(m)}]")
            print(f"  path: {m}")
            print(f"  pngs: {len(pngs)}, csvs: {len(csvs)}")
            # 找到包含 eval/sample 的具体子目录
            subdirs = set(os.path.dirname(p) for p in pngs[:10])
            print(f"  sample dirs: {list(subdirs)[:3]}")
