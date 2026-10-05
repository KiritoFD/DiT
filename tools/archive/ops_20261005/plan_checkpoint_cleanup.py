import os
import re
import sys

root_dirs = [
    "/root/Workspace/xy/DiT/data/archive/results_legacy",
    "/root/Workspace/xy/DiT/assets/results",
    "/root/Workspace/xy/DiT/_archive",
    "/root/Workspace/xy/DiT/exp-std",
    "/root/Workspace/xy/DiT/baseline",
]

total_files_before = 0
total_bytes_before = 0
total_files_to_delete = 0
total_bytes_to_delete = 0

plan = []

for root in root_dirs:
    if not os.path.exists(root):
        continue
    for dirpath, dirnames, filenames in os.walk(root):
        pts = [f for f in filenames if f.endswith((".pt", ".pth", ".ckpt"))]
        if len(pts) > 2:

            def sort_key(fn):
                nums = re.findall(r"\d+", fn)
                if nums:
                    return int(nums[-1])
                return 0

            special = [
                f
                for f in pts
                if any(k in f.lower() for k in ["best", "latest", "final"])
            ]
            numbered = [f for f in pts if f not in special]
            numbered.sort(key=sort_key)

            to_keep = set()
            for s in special:
                to_keep.add(s)
            if numbered:
                to_keep.add(numbered[-1])  # highest step / final
                if len(to_keep) < 2 and len(numbered) > 1:
                    to_keep.add(
                        numbered[0] if len(numbered) == 2 else numbered[-2]
                    )

            to_delete = [f for f in pts if f not in to_keep]

            dir_bytes = sum(os.path.getsize(os.path.join(dirpath, f)) for f in pts)
            del_bytes = sum(
                os.path.getsize(os.path.join(dirpath, f)) for f in to_delete
            )

            total_files_before += len(pts)
            total_bytes_before += dir_bytes
            total_files_to_delete += len(to_delete)
            total_bytes_to_delete += del_bytes

            plan.append(
                {
                    "dir": dirpath,
                    "total": len(pts),
                    "keep": sorted(list(to_keep)),
                    "delete": sorted(to_delete),
                    "freed_gb": del_bytes / (1024**3),
                }
            )

print("=" * 80)
print("Summary of Checkpoint Pruning Plan:")
print(f"  Total checkpoint files in multi-ckpt runs (>2): {total_files_before}")
print(
    f"  Checkpoints to keep: {total_files_before - total_files_to_delete} (1-2 per run)"
)
print(f"  Checkpoints to delete: {total_files_to_delete}")
print(f"  Total disk space to reclaim: {total_bytes_to_delete / (1024**3):.2f} GB!")
print("=" * 80)
print("Top 25 runs by reclaimable space:")
plan.sort(key=lambda x: x["freed_gb"], reverse=True)
for item in plan[:25]:
    k_str = str(item["keep"][:2])
    print(
        f"{item['freed_gb']:5.2f} GB | Keep: {k_str:<32} | Del {len(item['delete']):2d} | {item['dir']}"
    )

# Save full deletion list to file for safety
with open("/root/Workspace/xy/ckpt_prune_plan.txt", "w", encoding="utf-8") as f:
    for item in plan:
        f.write(f"DIR: {item['dir']}\n")
        f.write(f"KEEP: {', '.join(item['keep'])}\n")
        for d in item["delete"]:
            f.write(f"DELETE: {os.path.join(item['dir'], d)}\n")
        f.write("\n")
print("\nFull deletion plan saved to: /root/Workspace/xy/ckpt_prune_plan.txt")
