# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmds = [
    ("CANNY_D3 find", "find /root/Workspace/xy/DiT/final_canny_d3 -name '*.png' | wc -l"),
    ("CANNY_D3 ls", "ls /root/Workspace/xy/DiT/final_canny_d3/ | head -5"),
    ("CANNY_D3 dir", "ls -d /root/Workspace/xy/DiT/final_canny_d3 2>/dev/null && echo EXISTS || echo MISSING"),
    ("CANNY orig", "find /root/Workspace/xy/DiT/final_canny -name '*.png' | wc -l"),
]
for name, cmd in cmds:
    r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
        capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
    print(f"=== {name} ===")
    print(r.stdout.strip() if r.returncode==0 else f"(rc={r.returncode})")
    print()