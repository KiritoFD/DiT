import subprocess
import sys

cmd = """
mkdir -p /root/Workspace/xy/transfer_logs

# Run rsync or scp from 4090 to 48 in background
ssh -o ConnectTimeout=5 ds@10.222.120.101 'mkdir -p /home/ds/Workspace/RAW_DATASET'

nohup scp -r -o ConnectTimeout=10 /root/Workspace/xy/MODELSCOPE_UPLOAD/* ds@10.222.120.101:/home/ds/Workspace/RAW_DATASET/ > /root/Workspace/xy/transfer_logs/scp_48.log 2>&1 &
echo "SCP launched with PID: $!"
"""

res = subprocess.run(
    cmd, shell=True, capture_output=True, text=True, executable="/bin/bash"
)
print("=== SCP LAUNCH ON 4090 ===")
print(res.stdout)
if res.stderr:
    print("=== STDERR ===")
    print(res.stderr)
