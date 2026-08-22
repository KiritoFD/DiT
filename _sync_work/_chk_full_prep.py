# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmds = [
    ("CANNY_D3 COUNT", "ls /root/Workspace/xy/DiT/final_canny_d3/*.png 2>/dev/null | wc -l"),
    ("GEN LOG", "tail -8 /tmp/_gen_canny_d3.log"),
    ("TMUX", "tmux ls 2>/dev/null"),
    ("GPU", "nvidia-smi --query-gpu=memory.used --format=csv,noheader"),
    ("TRAIN_FULL", "wc -l /root/Workspace/xy/DiT/5script/train_full.csv"),
]
for name, cmd in cmds:
    r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
        capture_output=True,text=True,timeout=60, encoding="utf-8", errors="replace")
    print(f"=== {name} ===")
    print(r.stdout.strip() if r.returncode==0 else f"(rc={r.returncode})")
    print()