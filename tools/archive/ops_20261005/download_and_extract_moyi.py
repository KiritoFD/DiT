import os
import subprocess
import sys
import tarfile

target_dir = "docs/106moyi"
os.makedirs(target_dir, exist_ok=True)
local_tar = os.path.join(target_dir, "moyi_history_clean.tar.gz")

print("=" * 80)
print(f"Downloading {local_tar} from 48...")
print("=" * 80)

# Try direct scp first
cmd = ["scp", "-o", "ConnectTimeout=15", "48:/tmp/moyi_history_clean.tar.gz", local_tar]
res = subprocess.run(cmd)

if res.returncode != 0:
    print("Direct scp failed, routing via 4090...")
    # Route via 4090
    cmd_via_4090 = [
        "ssh", "-o", "ConnectTimeout=15", "4090",
        "scp -o ConnectTimeout=5 ds@10.222.120.101:/tmp/moyi_history_clean.tar.gz /root/Workspace/xy/ && ls -lh /root/Workspace/xy/moyi_history_clean.tar.gz"
    ]
    subprocess.run(cmd_via_4090, check=True)
    cmd_dl = [
        "scp", "-o", "ConnectTimeout=15", "4090:/root/Workspace/xy/moyi_history_clean.tar.gz", local_tar
    ]
    subprocess.run(cmd_dl, check=True)

print(f"Download complete: {os.path.getsize(local_tar) / (1024**2):.2f} MB!")
print("Extracting archive into docs/106moyi...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(target_dir)

os.remove(local_tar)
print("Extraction finished and temporary tarball removed!")
print("=" * 80)
