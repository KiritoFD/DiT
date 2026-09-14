#!/opt/conda/envs/cu121/bin/python
import os
BASE = "/root/Workspace/xy/HCSU"
for dataset in ["bei", "tie"]:
    d = os.path.join(BASE, dataset)
    print(f"\n=== {dataset} ===")
    count = 0
    for root, dirs, files in os.walk(d):
        for f in files:
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, d)
            count += 1
            if count <= 10:
                print(f"  {rel}")
    print(f"Total: {count} files")
