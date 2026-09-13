# -*- coding: utf-8 -*-
"""查 pixel diffusion 训练日志 + eval 进度。"""
import subprocess
HOST = "root@10.176.54.17"; PORT = "36430"
cmd = ("echo '=== TRAIN LOG (last 6) ==='; tail -6 /root/Workspace/xy/DiT/run_px_s.log; "
       "echo '=== EVAL LOG ==='; tail -6 /root/Workspace/xy/DiT/cpu_eval_px.log; "
       "echo '=== EVAL JSONS ==='; "
       "ls /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/eval_auto_*.json 2>/dev/null | tail -5; "
       "echo '=== CKPT COUNT ==='; "
       "ls /root/Workspace/xy/DiT/5script/results/px_s_scratch/*/checkpoints/*.pt 2>/dev/null | wc -l; "
       "echo '=== PROCS ==='; pgrep -af 'train_pixel|auto_eval_pixel' | head -4; "
       "echo '=== GPU ==='; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader")
r = subprocess.run(["ssh","-o","ConnectTimeout=25","-p",PORT,HOST,cmd],
                   capture_output=True, text=True, timeout=90)
print(r.stdout if r.returncode==0 else f"rc={r.returncode} {r.stderr[:200]}")