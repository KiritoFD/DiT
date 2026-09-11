# -*- coding: utf-8 -*-
"""对历史 ckpt 用当前 strict 协议 (n=237, cfg0.7, Heun50) GPU 重评.

- 每个 run 取最新一个 ckpt
- 架构不兼容 (xattn/style/spatial/DINO 等已删模块) -> 记录 not_loadable
- 独立临时 results 目录避免幂等跳过
输出: /tmp/reeval_strict.txt
"""
import glob
import os
import re
import shutil
import subprocess
import sys

D = "/root/Workspace/xy/DiT"
R = os.path.join(D, "5script/results")
CSV = "5script/eval_fame3_strict_clean_v9.csv"
TMP = "/tmp/reeval_tmp"

runs = sorted([os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(R, "*")) if os.path.isdir(p)])
out = []
for run in runs:
    cks = sorted(glob.glob(f"{R}/{run}/*/checkpoints/[0-9]*.pt"),
                 key=lambda p: int(os.path.basename(p).split(".")[0]))
    if not cks:
        continue
    ck = cks[-1]
    step = int(os.path.basename(ck).split(".")[0])
    shutil.rmtree(TMP, ignore_errors=True)
    os.makedirs(TMP, exist_ok=True)
    cmd = [sys.executable, "-u", "tools/eval/eval_stdskel_batch.py",
           "--results-dir", TMP, "--ckpt-override", ck, "--device", "cuda",
           "--sets", f"strict:{CSV}:237", "--dit-batch", "64", "--vae-batch", "32"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=D)
    o = r.stdout + r.stderr
    m = re.search(r"strict: ssim=([0-9.]+)", o)
    if m:
        line = f"{run:56s} step{step:>7} strict={m.group(1)}"
        print(line, flush=True)
    else:
        err = ""
        if "unexpected=" in o:
            err = "unexpected=" + o.split("unexpected=")[1].split()[0]
        elif "OutOfMemory" in o:
            err = "OOM"
        else:
            err = [l for l in o.splitlines() if "Error" in l or "error" in l][-1:] or ["fail"]
            err = str(err)
        line = f"{run:56s} step{step:>7} NOT_LOADABLE ({err})"
        print(line, flush=True)
    out.append(line)
open("/tmp/reeval_strict.txt", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("REEVAL_DONE", len(out))
