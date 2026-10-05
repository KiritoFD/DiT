import os
import sys
import time

plan_file = "/root/Workspace/xy/ckpt_prune_plan.txt"

if not os.path.exists(plan_file):
    print("Plan file not found:", plan_file)
    sys.exit(1)

print("=" * 80)
print("Executing Checkpoint Pruning according to plan...")
print("=" * 80)

deleted_count = 0
deleted_bytes = 0
failed_count = 0

t0 = time.time()

with open(plan_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith("DELETE: "):
            filepath = line[len("DELETE: ") :].strip()
            if os.path.exists(filepath):
                try:
                    sz = os.path.getsize(filepath)
                    os.remove(filepath)
                    deleted_count += 1
                    deleted_bytes += sz
                    if deleted_count % 100 == 0:
                        print(
                            f"Deleted {deleted_count} files, freed {deleted_bytes / (1024**3):.2f} GB..."
                        )
                except Exception as e:
                    print(f"Error deleting {filepath}: {e}")
                    failed_count += 1

dt = time.time() - t0
freed_gb = deleted_bytes / (1024**3)

print("=" * 80)
print(f"Checkpoint Pruning Finished in {dt:.1f}s!")
print(f"  Successfully deleted: {deleted_count:,} intermediate checkpoints")
print(f"  Failed: {failed_count}")
print(f"  Total disk space recovered: {freed_gb:.2f} GB!")
print("=" * 80)
