# -*- coding: utf-8 -*-
"""把 _MID_SKEL_SRC/_MID_SKEL_WARNED 的定义从 `if rank == 0:` 块内移到块之前。

插在 rank==0 块内会导致: ① if 后缺少缩进块 -> IndentationError;
② 即便修好缩进, rank!=0 时这两个名字根本不会被定义 -> 调用处 NameError。
"""
import io

P = "/root/Workspace/xy/DiT/src/train/train.py"
src = "\n".join(io.open(P, encoding="utf-8", newline="").read().split("\r\n"))
lines = src.split("\n")

i = next(k for k, l in enumerate(lines) if "_MID_SKEL_SRC = str(" in l)
j = next(k for k, l in enumerate(lines) if "_MID_SKEL_WARNED = False" in l)
c = i - 1 if "中程载体来源开关" in lines[i - 1] else i
block = lines[c:j + 1]
del lines[c:j + 1]

r = next(k for k, l in enumerate(lines) if l.strip() == "if rank == 0:")
lines[r:r] = block

out = "\r\n".join("\n".join(lines).split("\n"))
io.open(P, "w", encoding="utf-8", newline="").write(out)
print("moved before 'if rank == 0:'")
