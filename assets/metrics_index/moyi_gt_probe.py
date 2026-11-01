"""Read characters straight off the MoYi GT images with the project's own OCR.

If the GT renders are legible, the character sequence can be reconstructed without
any protocol CSV; writer labels remain unknown, so only OCR (not E-G) becomes usable.
"""
import io
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
PANEL = Path("assets/server_48_evals/moyi_12ch_eval200fix")

for name in ["gt0.png", "gt1.png", "gt2.png", "gt3.png"]:
    p = PANEL / name
    with Image.open(p) as im:
        a = im.convert("L")
        hist = Counter(a.getdata())
    total = sum(hist.values())
    dark = sum(v for k, v in hist.items() if k < 128)
    print(f"{name}: size={a.size} dark_frac={dark / total:.3f} "
          f"mean={sum(k * v for k, v in hist.items()) / total:.1f}")
