# -*- coding: utf-8 -*-
import os
import sys
import traceback

sys.path.insert(0, "/root/Workspace/xy/DiT")
os.chdir("/root/Workspace/xy/DiT")
from src.eval.in_mem_eval import render_poster
try:
    p = render_poster("assets/results/v11_pretrain_Sp2_base", "seen")
    print("OK:", p)
except Exception:
    traceback.print_exc()
