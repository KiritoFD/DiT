import os
import subprocess
import sys
import tarfile

LOCAL_TARGET = "docs/106moyi"
os.makedirs(LOCAL_TARGET, exist_ok=True)

print("=" * 80)
print("【正在从 48 机打包拉取所有实验数据、JSON与图像至 docs/106moyi】")
print("=" * 80)

# 1. 在 48 机上打包排除 pt/pth/ckpt/npz
make_tar_cmd = [
    "ssh",
    "-o",
    "ConnectTimeout=10",
    "48",
    (
        "cd /home/ds/Workspace && "
        "tar -czf /tmp/moyi_history.tar.gz "
        "--exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='*.npz' "
        "moyi/results moyi/logs DiT/experiments && "
        "ls -lh /tmp/moyi_history.tar.gz"
    ),
]

print("-> 正在 48 机打包历史实验产物...")
res = subprocess.run(make_tar_cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

# 2. 下载 tar 包到本地
local_tar = os.path.join(LOCAL_TARGET, "moyi_history.tar.gz")
scp_cmd = [
    "scp",
    "-o",
    "ConnectTimeout=15",
    "48:/tmp/moyi_history.tar.gz",
    local_tar,
]
print(f"-> 正在拉取 tar 包至本地: {local_tar} ...")
subprocess.run(scp_cmd, check=True)
print(f"   下载完成: {os.path.getsize(local_tar) / (1024**2):.2f} MB")

# 3. 解包至 docs/106moyi
print(f"-> 正在解包至 {LOCAL_TARGET} ...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(path=LOCAL_TARGET)

os.remove(local_tar)
print("-> 解压完成并已清理临时 tarball！")
print("=" * 80)
