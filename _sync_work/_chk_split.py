# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
# 分离命令以避免引号问题
cmds = [
    ("SKEL 1px", "tail -4 /root/Workspace/xy/DiT/run_skel_dec.log"),
    ("SKEL 1px HIST", "cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json"),
    ("SKEL 3px", "tail -8 /root/Workspace/xy/DiT/run_skel_d3.log"),
    ("SKEL 3px HIST", "cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json"),
    ("PIXEL", "tail -2 /root/Workspace/xy/DiT/run_px_s.log"),
]
for name, cmd in cmds:
    r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
        capture_output=True,text=True,timeout=60)
    print(f"=== {name} ===")
    print(r.stdout.strip() if r.returncode==0 else f"(rc={r.returncode})")
    print()