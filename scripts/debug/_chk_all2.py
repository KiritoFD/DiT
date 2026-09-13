# -*- coding: utf-8 -*-
"""查 skel decoder 1px 训练进度 + pixel diffusion 状态。"""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmd = ("echo '=== SKEL 1px ==='; tail -6 /root/Workspace/xy/DiT/run_skel_dec.log; "
       "echo '=== SKEL HIST ==='; cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null; "
       "echo '=== PIXEL ==='; tail -2 /root/Workspace/xy/DiT/run_px_s.log; "
       "echo '=== PX EVAL ==='; ls /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/eval_auto_*.json 2>/dev/null | tail -3")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
    capture_output=True,text=True,timeout=60)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")