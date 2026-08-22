# -*- coding: utf-8 -*-
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
cmd = ("echo '=== SKEL 1px TRAIN ==='; "
       "tail -4 /root/Workspace/xy/DiT/run_skel_dec.log; "
       "echo '=== SKEL 1px HISTORY ==='; "
       "cat /root/Workspace/xy/DiT/5script/results/skel_decoder/history.json 2>/dev/null; "
       "echo '=== GEN D3 ==='; "
       "tail -3 /tmp/_gen_skel_d3.log 2>/dev/null; "
       "ls /root/Workspace/xy/DiT/final_skeleton_d3/ 2>/dev/null | wc -l; "
       "echo '=== PIXEL TRAIN ==='; "
       "tail -2 /root/Workspace/xy/DiT/run_px_s.log; "
       "echo '=== PX EVAL ==='; "
       "cat /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/eval_auto_*.json 2>/dev/null | tail -4")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
    capture_output=True,text=True,timeout=90)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")