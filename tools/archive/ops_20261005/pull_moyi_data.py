import os
import subprocess
import sys
import tarfile

TARGET_DIR = "docs/106moyi"
os.makedirs(TARGET_DIR, exist_ok=True)

print("=" * 80)
print("【正在打包并拉取 48 机的全部实验历史 JSON、CSV、日志与核心评估图】")
print("=" * 80)

# 1. 在 48 机上打包所有 json, csv, log, 评估报告, 及核心评估对比图 (排除大图与权重)
pack_code = """
import os, tarfile

base = '/home/ds/Workspace'
out_tar = '/tmp/moyi_jsons_fast.tar.gz'

dirs = [
    os.path.join(base, 'moyi/results'),
    os.path.join(base, 'moyi/logs'),
    os.path.join(base, 'DiT/experiments')
]

files = []
for d in dirs:
    if not os.path.exists(d): continue
    for root, _, fnames in os.walk(d):
        for f in fnames:
            ext = os.path.splitext(f)[1].lower()
            fp = os.path.join(root, f)
            rel = os.path.relpath(fp, base)
            # Skip weights and shards
            if ext in {'.pt', '.pth', '.ckpt', '.npz', '.bin', '.safetensors'}:
                continue
            # Keep all json, csv, txt, md, log
            if ext in {'.json', '.csv', '.txt', '.md', '.log'}:
                files.append((fp, rel))
            # Keep key posters and evaluation charts
            elif 'poster' in f.lower() or 'eval' in f.lower() or 'strict' in f.lower():
                if os.path.getsize(fp) < 2 * 1024 * 1024:
                    files.append((fp, rel))

print(f'Collected {len(files)} files for fast packing...')
with tarfile.open(out_tar, 'w:gz') as t:
    for fp, rel in files:
        t.add(fp, arcname=rel)
print(f'Archive created: {out_tar} ({os.path.getsize(out_tar)/(1024**2):.2f} MB)')
"""

cmd_pack_48 = [
    "ssh",
    "-o",
    "ConnectTimeout=10",
    "4090",
    (
        "ssh -o ConnectTimeout=5 ds@10.222.120.101"
        " '/home/ds/miniconda3/envs/pytorch/bin/python -c \""
        + pack_code.replace('"', '\\"').replace("\n", " ")
        + "\"'"
    ),
]

print("-> 1. 在 48 机上执行快速打包...")
res = subprocess.run(cmd_pack_48, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

# 2. 从 48 机传输到 4090
cmd_48_to_4090 = [
    "ssh",
    "-o",
    "ConnectTimeout=10",
    "4090",
    (
        "scp -o ConnectTimeout=5 ds@10.222.120.101:/tmp/moyi_jsons_fast.tar.gz"
        " /root/Workspace/xy/ && ls -lh /root/Workspace/xy/moyi_jsons_fast.tar.gz"
    ),
]
print("-> 2. 从 48 机同步至 4090 中转...")
res2 = subprocess.run(cmd_48_to_4090, capture_output=True, text=True)
print(res2.stdout)

# 3. 从 4090 下载至本地 docs/106moyi
local_tar = os.path.join(TARGET_DIR, "moyi_jsons_fast.tar.gz")
cmd_dl = [
    "scp",
    "-P",
    "36430",
    "-o",
    "ConnectTimeout=15",
    "root@10.176.54.17:/root/Workspace/xy/moyi_jsons_fast.tar.gz",
    local_tar,
]
print(f"-> 3. 从 4090 拉取至本地: {local_tar} ...")
subprocess.run(cmd_dl, check=True)

# 4. 解包
print(f"-> 4. 解包至 {TARGET_DIR} ...")
with tarfile.open(local_tar, "r:gz") as tar:
    tar.extractall(TARGET_DIR)
os.remove(local_tar)
print("-> 解包完成并清理临时压缩包！")
print("=" * 80)
