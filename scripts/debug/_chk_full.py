# -*- coding: utf-8 -*-
"""查全部状态: pixel GPU训练 + 3px skel decoder + 1px skel decoder."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"

cmds = [
    ("PIXEL GPU", "tail -3 /root/Workspace/xy/DiT/run_px_s.log"),
    ("PX EVAL", "cat /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/eval_auto_*.json 2>/dev/null | tail -2"),
    ("3px SKEL", "tail -4 /root/Workspace/xy/DiT/run_skel_d3.log"),
    ("3px HIST", "cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json 2>/dev/null"),
    ("1px SKEL", "tail -2 /root/Workspace/xy/DiT/run_skel_dec.log"),
    ("1px HIST", "cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null"),
    ("TMUX", "tmux ls 2>/dev/null"),
    ("GPU", "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"),
]
for name, cmd in cmds:
    r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
        capture_output=True,text=True,timeout=60)
    print(f"=== {name} ===")
    print(r.stdout.strip() if r.returncode==0 else f"(rc={r.returncode})")
    print()