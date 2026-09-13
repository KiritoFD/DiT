# -*- coding: utf-8 -*-
"""拉取 eval 可视化到本地查看."""
import subprocess
HOST="root@10.176.54.17"; PORT="36430"
import os
os.makedirs(r"G:\GitHub\DiT\_sync_work\px_eval_55k", exist_ok=True)
for f in ["eval_grid.png", "pred_0.png", "pred_1.png", "gt_0.png", "gt_1.png"]:
    r = subprocess.run(["scp","-o","ConnectTimeout=25","-P",PORT,
        f"root@10.176.54.17:/tmp/px_eval_55k/{f}",
        rf"G:\GitHub\DiT\_sync_work\px_eval_55k\{f}"],
        capture_output=True, timeout=60)
    print(f"{f}: {r.returncode==0}")