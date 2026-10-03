# -*- coding: utf-8 -*-
"""_redraw_poster.py — 单独重画 poster (CPU, 秒级), 不重启训练."""
import os
import sys

ROOT = "/root/Workspace/xy/DiT"
os.chdir(ROOT)
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.eval.in_mem_eval import render_poster  # noqa: E402

rd = sys.argv[1] if len(sys.argv) > 1 else "assets/results/v11_pretrain_Sp2_base_wz"
for s in ("seen", "strict"):
    p = render_poster(rd, s)
    print(f"[redraw] {s} -> {p}", flush=True)
