# -*- coding: utf-8 -*-
"""code_md5.py — 打印关键文件的 md5（用于两端一致性核对）"""
import hashlib
import os
import sys

_R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(_R if os.path.isdir(os.path.join(_R, "src")) else os.getcwd())
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
FILES = ["src/model/dit.py", "src/train/train.py", "src/train/cli.py", "src/train/dit.py",
         "src/eval/inference.py", "src/eval/in_mem_eval.py", "src/utils/latent_dataset.py",
         "src/loss/flow_matching.py", "src/utils/std_soft_glyph.py", "tools/gen_eval_rows.py"]
for f in FILES:
    if os.path.exists(f):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()[:12]
        print(f"  {h}  {f}  ({os.path.getsize(f)} B)")
    else:
        print(f"  {'MISSING':<12}  {f}")
