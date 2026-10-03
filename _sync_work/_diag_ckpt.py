# -*- coding: utf-8 -*-
"""_diag_ckpt.py — 排查 ckpt 从 03:34 之后不再保存的原因."""
import glob
import os
import re
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LOG = "logs/v11_pretrain_Sp2_base_sym/train_r3.log"
print(f"=== {LOG} 里的保存/异常相关行 ===")
pat = re.compile(r"(sav|ckpt|Checkpoint|Error|Traceback|Exception|raise|fail|WARN)",
                 re.I)
hits = []
for i, line in enumerate(open(LOG, encoding="utf-8", errors="replace")):
    if pat.search(line):
        hits.append((i, line.rstrip()[:200]))
print(f"匹配 {len(hits)} 行, 显示前后各若干:")
for i, l in hits[:6]:
    print(f"  [{i}] {l}")
print("  ...")
for i, l in hits[-14:]:
    print(f"  [{i}] {l}")

print("\n=== 日志首部(启动信息) ===")
for line in list(open(LOG, encoding="utf-8", errors="replace"))[:40]:
    if line.strip():
        print("  " + line.rstrip()[:190])

print("\n=== 实验目录 ===")
for d in sorted(glob.glob("assets/results/v11_pretrain_Sp2_base_sym/*")):
    print(f"  {d}  mtime={os.path.getmtime(d)}")
    ck = os.path.join(d, "checkpoints")
    if os.path.isdir(ck):
        fs = sorted(glob.glob(os.path.join(ck, "[0-9]*.pt")))
        print(f"     ckpt: {[os.path.basename(f) for f in fs]}")

print("\n=== _active_ckpt_dir.txt ===")
p = "assets/results/v11_pretrain_Sp2_base_sym/_active_ckpt_dir.txt"
print("  " + (open(p, encoding="utf-8").read().strip() if os.path.exists(p) else "(缺失)"))
