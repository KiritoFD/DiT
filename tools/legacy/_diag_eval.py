#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检查远程 s8 ckpt 目录里实际有哪些 eval_auto_*.json + cpu_eval_state.json 内容。
关键: eval_auto_*.json 是 auto_eval_cpu.py 写的, 但旧 s7 的 eval_auto 文件可能残留在
s7 的 ckpt 目录, 而 pull_monitor 读的是 s8 的 ckpt 目录 —— 问题在 eval_log_lines。"""
import subprocess, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEP="===_DSH_SEP_==="
CMD=(
    "echo '{SEP}S8_EVAL_AUTO'; "
    "ls -la /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/20260823-234546-s8-klf4-clean-dino/checkpoints/eval_auto_*.json 2>/dev/null; "
    "echo '{SEP}S8_STATE'; "
    "cat /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/20260823-234546-s8-klf4-clean-dino/checkpoints/cpu_eval_state.json 2>/dev/null; "
    "echo '{SEP}EVAL_LOG_TAIL'; "
    "tail -30 /root/Workspace/xy/DiT/auto_eval_cpu.log 2>/dev/null; "
    "echo '{SEP}EVAL_LOG_FULL'; "
    "cat /root/Workspace/xy/DiT/auto_eval_cpu.log 2>/dev/null; "
    "echo '{SEP}END'"
).replace("{SEP}", SEP)

r=subprocess.run(["ssh","-o","ConnectTimeout=15","-p","36430","root@10.176.54.17",CMD],
    capture_output=True,timeout=40)
out=r.stdout.decode("utf-8", errors="replace")
parts=out.split(SEP)
for i,p in enumerate(parts):
    p=p.strip()
    if p and not p.startswith("END"):
        print("=== section",i,"===")
        print(p[:1500])