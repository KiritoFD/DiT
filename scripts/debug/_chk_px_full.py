# -*- coding: utf-8 -*-
"""查 pixel diffusion 全部 eval 结果 + 当前训练状态."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"

cmds = [
    ("PX TRAIN", "tail -4 /root/Workspace/xy/DiT/run_px_s.log"),
    ("PX EVAL ALL", "for f in /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/eval_auto_*.json; do cat $f; echo; done 2>/dev/null"),
    ("PX CKPT", "ls /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/*.pt 2>/dev/null | sort"),
    ("GPU", "nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader"),
    ("3px", "tail -2 /root/Workspace/xy/DiT/run_skel_d3.log; cat /root/Workspace/xy/DiT/5script/results/skel_decoder_d3/history.json 2>/dev/null"),
]
for name, cmd in cmds:
    r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
        capture_output=True,text=True,timeout=60)
    print(f"=== {name} ===")
    print(r.stdout.strip() if r.returncode==0 else f"(rc={r.returncode})")
    print()