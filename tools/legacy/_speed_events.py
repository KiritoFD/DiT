import re, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
with open("/root/Workspace/xy/DiT/logs_s10.txt", errors="replace") as f:
    lines = f.readlines()
# find when speed dropped and recovered
prev_sps = None
for line in lines:
    m = re.search(r"step=(\d+).*Steps/Sec: ([\d.]+)", line)
    if m:
        step = int(m.group(1))
        sps = float(m.group(2))
        if prev_sps is not None:
            if abs(sps - prev_sps) > 0.5:
                ts_m = re.search(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", line)
                ts = ts_m.group(1) if ts_m else "?"
                print(f"  step={step} sps={sps} (was {prev_sps}) at {ts}")
        prev_sps = sps
