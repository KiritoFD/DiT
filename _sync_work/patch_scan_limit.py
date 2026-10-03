"""ckpt 太多(1283)会让下拉不可用 —— 改成每个实验只保留最新的若干步。"""
import io
import os

os.chdir("/root/Workspace/xy/DiT")
P = "gradio_stdskel.py"
s = io.open(P, encoding="utf-8").read()

OLD = '''def _scan_ckpts():
    """返回 [(显示名, 路径)]。显示名 = 实验目录/步数。"""
    import glob as _g
    out = []
    for p in sorted(_g.glob("assets/results/*/*/checkpoints/*.pt")):
        parts = p.split(os.sep)
        exp = parts[2] if len(parts) > 2 else "?"
        step = os.path.basename(p).replace(".pt", "").lstrip("0") or "0"
        out.append((f"{exp} @{step}k", p))
    # 最新步数在前
    out.sort(key=lambda kv: os.path.getmtime(kv[1]), reverse=True)
    return out'''

NEW = '''def _scan_ckpts(keep_per_exp=3):
    """返回 [(显示名, 路径)]。

    ⚠ 直接扫会得到 1000+ 个 ckpt（每个实验每 5k 步存一个），下拉根本没法用。
    所以**每个实验只保留步数最大的 keep_per_exp 个**，
    并按 mtime 排序（最近训的实验在前）。
    """
    import glob as _g
    from collections import defaultdict as _dd
    by_exp = _dd(list)
    for p in sorted(_g.glob("assets/results/*/*/checkpoints/*.pt")):
        parts = p.split(os.sep)
        exp = parts[2] if len(parts) > 2 else "?"
        try:
            step = int(os.path.basename(p).replace(".pt", ""))
        except ValueError:
            step = -1
        by_exp[exp].append((step, p))
    out = []
    for exp, lst in by_exp.items():
        lst.sort(key=lambda t: t[0], reverse=True)
        for step, p in lst[:keep_per_exp]:
            out.append((f"{exp} @{step}", p))
    out.sort(key=lambda kv: os.path.getmtime(kv[1]), reverse=True)
    return out'''

if OLD in s:
    s = s.replace(OLD, NEW, 1)
    print("  ✓ 已改为每实验保留 3 个")
else:
    print("  ⚠ 没找到原 _scan_ckpts")

io.open(P, "w", encoding="utf-8").write(s)
import ast

ast.parse(s)
print("  SYNTAX OK")
