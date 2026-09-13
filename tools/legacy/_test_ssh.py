import subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
REMOTE_PORT = "36430"
REMOTE_USER = "root"
REMOTE_HOST = "10.176.54.17"

cmd = ["ssh", "-o", "ConnectTimeout=15", "-p", REMOTE_PORT,
       f"{REMOTE_USER}@{REMOTE_HOST}",
       "find /root/Workspace/xy/DiT/5script/results -name log.txt 2>/dev/null | xargs ls -t 2>/dev/null | head -1"]
print("cmd:", " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
print("rc:", r.returncode)
print("stdout:", repr(r.stdout[:200]))
print("stderr:", repr(r.stderr[:200]))
