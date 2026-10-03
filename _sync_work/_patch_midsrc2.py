# -*- coding: utf-8 -*-
"""修 _MID_SKEL_SRC 定义的缩进: 它插在 `if w_std_mid>0:` 块内, 需要 8 空格。"""
import io

P = "/root/Workspace/xy/DiT/src/train/train.py"
src = "\n".join(io.open(P, encoding="utf-8", newline="").read().split("\r\n"))

OLD = ("    _MID_SKEL_SRC = str(getattr(args, 'std_mid_skel_src', 'g') or 'g')\n"
       "    _MID_SKEL_WARNED = False")
NEW = ("        _MID_SKEL_SRC = str(getattr(args, 'std_mid_skel_src', 'g') or 'g')\n"
       "        _MID_SKEL_WARNED = False")
assert OLD in src, "OLD not found"
src = src.replace(OLD, NEW, 1)
io.open(P, "w", encoding="utf-8", newline="").write("\r\n".join(src.split("\n")))
print("fixed indent")
