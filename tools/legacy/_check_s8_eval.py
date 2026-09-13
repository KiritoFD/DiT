# -*- coding: utf-8 -*-
"""检查远程 eval_auto_*.json 文件，确认 s8 当前 eval 产出。
注意: pull_monitor 把 s7 旧 eval 文件也误读进了 train_data.json ——
需要确认 s8 是否真有 eval 产出（s8 step 才到 3160，不应该有 eval）。"""
import subprocess, re
SEP="===_DSH_SEP_==="
CMD=(
    f"echo '{SEP}S8_EVAL_DIR'; "
    f"ls /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/*/checkpoints/eval_auto_*.json 2>/dev/null; "
    f"echo '{SEP}S8_STATE'; "
    f"cat /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/*/checkpoints/cpu_eval_state.json 2>/dev/null; "
    f"echo '{SEP}ACTIVE'; "
    f"cat /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/_active_ckpt_dir.txt 2>/dev/null; "
    f"echo '{SEP}S8_LOG_TAIL'; "
    f"tail -3 /root/Workspace/xy/DiT/5script/results/s8_klf4_clean_dino/*/log.txt 2>/dev/null; "
    f"echo '{SEP}END'"
)
r=subprocess.run(["ssh","-o","ConnectTimeout=15","-p","36430","root@10.176.54.17",CMD],
    capture_output=True,text=True,timeout=40)
out=r.stdout
parts=out.split(SEP)
for i,p in enumerate(parts):
    p=p.strip()
    if p and not p.startswith("END"):
        print("=== section",i,"===")
        print(p[:500])