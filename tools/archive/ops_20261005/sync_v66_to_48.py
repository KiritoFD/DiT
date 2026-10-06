import os
import subprocess
import sys

print("=" * 80)
print("【从 4090 将 v66 条件分层路由代码完整同步至 48 机】")
print("=" * 80)

sync_cmd = """
cd /root/Workspace/xy/DiT
scp -o ConnectTimeout=5 src/model/injections.py src/model/dit.py ds@10.222.120.101:/home/ds/Workspace/DiT/src/model/
scp -o ConnectTimeout=5 src/train/cli.py src/train/train.py ds@10.222.120.101:/home/ds/Workspace/DiT/src/train/
scp -o ConnectTimeout=5 src/eval/model_io.py src/eval/cli.py ds@10.222.120.101:/home/ds/Workspace/DiT/src/eval/
scp -o ConnectTimeout=5 src/utils/inject_probe.py ds@10.222.120.101:/home/ds/Workspace/DiT/src/utils/
echo "Files copied to 48 successfully!"
"""

cmd_via_4090 = [
    "ssh",
    "-o",
    "ConnectTimeout=15",
    "4090",
    f"bash -c '{sync_cmd}'",
]

res = subprocess.run(cmd_via_4090, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)

# 验证 48 机导入
verify_py = "from src.model import DiT_2Cond_models; m = DiT_2Cond_models['DiT-2Cond-B/2'](cond_inject_at='2,4,5,6', cond_inject_scale=True); print('✓ 48 Verified DiT-2Cond-B/2 cond_router:', m.cond_router)"

cmd_verify = [
    "ssh",
    "-o",
    "ConnectTimeout=15",
    "4090",
    (
        "ssh -o ConnectTimeout=5 ds@10.222.120.101"
        " '/home/ds/miniconda3/envs/pytorch/bin/python -c \""
        + verify_py
        + "\"'"
    ),
]

res_v = subprocess.run(cmd_verify, capture_output=True, text=True)
print(res_v.stdout)
if res_v.stderr:
    print("VERIFY STDERR:", res_v.stderr)
print("=" * 80)
