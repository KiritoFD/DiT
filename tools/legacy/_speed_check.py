import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
with open("/root/Workspace/xy/DiT/logs_s10.txt", errors="replace") as f:
    lines = f.readlines()
for line in lines[-40:]:
    m = re.search(r"step=(\d+).*Steps/Sec: ([\d.]+)", line)
    if m:
        print(f"step={m.group(1)} sps={m.group(2)}")
