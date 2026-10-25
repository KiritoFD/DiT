import json
import sys

paths = [
    "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/capacity_audit_20261017/exposure_matched_exact/report.json",
    "/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/assets/capacity_audit_20261017/exposure_matched/report.json"
]

all_models = {}
times = [0.1, 0.25, 0.4, 0.6, 0.8, 0.95]

for p in paths:
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        for m, info in d["models"].items():
            if m not in all_models:
                all_models[m] = info["buckets"]
    except Exception as e:
        print(f"Error reading {p}: {e}")

header = f"{'Model':<16} {'Exposure':<10} {'Split':<6} " + " ".join([f"t={t:<5}" for t in times]) + " Overall"
print(header)
print("-" * len(header))

exposure_map = {
    "l20_384_180k": "34.56M",
    "l20_256_120k": "34.56M",
    "l20_384_200k": "38.40M",
    "l20_256_130k": "37.44M",
    "l20_256_140k": "40.32M",
}

for m in ["l20_384_180k", "l20_256_120k", "l20_384_200k", "l20_256_130k", "l20_256_140k"]:
    if m not in all_models:
        continue
    b = all_models[m]
    exp = exposure_map.get(m, "unknown")
    for split in ["train", "val"]:
        row = []
        for t in times:
            val = b.get(f"{split}|t={t}", {}).get("mean", 0.0)
            row.append(f"{val:.5f}")
        overall = b.get(f"split={split}", {}).get("mean", 0.0)
        print(f"{m:<16} {exp:<10} {split:<6} " + " ".join(row) + f" {overall:.5f}")
