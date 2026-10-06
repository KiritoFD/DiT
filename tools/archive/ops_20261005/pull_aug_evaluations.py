import os
import subprocess
import sys
import tarfile

target_dir = "docs/106moyi/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h/evaluations"
os.makedirs(target_dir, exist_ok=True)
local_tar = "docs/106moyi/evaluations_aug.tar.gz"

print("=" * 80)
print("Packing and pulling evaluation results from Machine 48...")
print("=" * 80)

# Pack evaluations on 48
pack_cmd = [
    "ssh", "-o", "ConnectTimeout=10", "48",
    "cd /home/ds/Workspace/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h && "
    "tar -czf /tmp/evaluations_aug.tar.gz evaluations && ls -lh /tmp/evaluations_aug.tar.gz"
]
subprocess.run(pack_cmd, check=True)

# Pull tar to local
dl_cmd = ["scp", "-o", "ConnectTimeout=15", "48:/tmp/evaluations_aug.tar.gz", local_tar]
subprocess.run(dl_cmd, check=True)

# Extract
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall("docs/106moyi/DiT/experiments/capacity_ladder/results/tier3_b_aug/20261006-025311-cap_tier3_b_aug_10h")

os.remove(local_tar)
print("Evaluations unpacked into docs/106moyi successfully!")
print("=" * 80)
