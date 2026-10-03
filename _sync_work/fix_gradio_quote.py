"""修 gradio_stdskel.py 第 155 行附近的 f-string 引号冲突。"""
import io
import os
import re

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

lines = s.split("\n")
out = []
fixed = 0
for ln in lines:
    if 'print(f"[bank]' in ln and "不在库存" in ln:
        # 整条 print 改成不含嵌套双引号的写法
        indent = ln[: len(ln) - len(ln.lstrip())]
        out.append(indent + 'print("[bank]    consequence: all chars will be '
                            'reported as not-in-bank (g library not loaded)", '
                            'flush=True)')
        fixed += 1
        print("  替换:", ln.strip()[:70])
    else:
        out.append(ln)

s = "\n".join(out)
io.open(P, "w", encoding="utf-8").write(s)
print(f"  修了 {fixed} 行")
import ast

ast.parse(s)
print("  SYNTAX OK")
