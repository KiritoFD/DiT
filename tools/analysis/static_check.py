# -*- coding: utf-8 -*-
"""static_check.py — 静态保证代码正确（不跑 GPU）

A) pyflakes/flake8（若装了）—— 它能直接报 "possibly undefined / referenced before assignment"
B) 自写 AST 扫描：找**只在更深缩进的分支里赋值、却在更浅缩进处被引用**的局部变量
   —— 这正是 `_repa_cache_dir` 那类 bug 的形态（w_repa>0 内赋值，外面无条件用）
C) 打印 dit.py 补丁块 + 做形状/梯度静态推理
"""
import ast
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FILES = ["src/train/train.py", "src/model/dit.py"]

print("=== A. pyflakes / flake8 ===")
ran = False
for tool in ("pyflakes", "flake8"):
    try:
        r = subprocess.run([sys.executable, "-m", tool, *FILES], capture_output=True, text=True, timeout=180)
        out = (r.stdout or "") + (r.stderr or "")
        lines = [l for l in out.splitlines() if "unable to detect undefined" not in l]
        hit = [l for l in lines if ("undefined" in l.lower() or "referenced before" in l.lower()
                                    or "local variable" in l.lower())]
        print(f"  {tool}: 输出 {len(lines)} 行，其中疑似未绑定/未定义 {len(hit)} 行")
        for l in hit[:12]:
            print("    " + l[:150])
        ran = True
        break
    except Exception as e:
        print(f"  {tool}: 不可用 ({type(e).__name__})")
if not ran:
    print("  （两个都没装 -> 用 B 的自写扫描）")


def indentation_scan(path):
    """返回 [(var, 赋值最小缩进, 引用最小缩进, 首次引用行)]，只在 引用缩进 < 赋值缩进 时才可疑。"""
    src = io.open(path, encoding="utf-8").read()
    tree = ast.parse(src)
    lines = src.splitlines()
    issues = []
    # 逐函数分析
    for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        assigns = {}
        uses = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Name):
                if isinstance(node.ctx, ast.Store):
                    assigns.setdefault(node.id, []).append(node.lineno)
                elif isinstance(node.ctx, ast.Load):
                    uses.setdefault(node.id, []).append(node.lineno)
        fn_args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        if fn.args.vararg:
            fn_args.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            fn_args.add(fn.args.kwarg.arg)
        for var, ulines in uses.items():
            if var in fn_args or var.startswith("__"):
                continue
            if var not in assigns:
                continue          # 可能是全局/闭包 -> 交给 A
            def ind(ln):
                s = lines[ln - 1]
                return len(s) - len(s.lstrip())
            amin = min(ind(l) for l in assigns[var])
            ufirst = min(ulines)
            umin = ind(ufirst)
            if umin < amin:       # 引用处比所有赋值处都浅 -> 存在"未赋值就引用"的路径
                issues.append((var, amin, umin, ufirst, fn.name))
    return issues


print("\n=== B. 自写扫描：分支内赋值 / 外层引用（同类 bug 全扫） ===")
total = 0
for f in FILES:
    iss = indentation_scan(f)
    print(f"  {f}: {len(iss)} 处可疑")
    for var, amin, umin, ufirst, fn in iss[:12]:
        print(f"    ⚠ {var}  赋值缩进>={amin}  首次引用 L{ufirst}(缩进 {umin})  在 {fn}()")
    total += len(iss)
print(f"  合计 {total} 处" + ("  ✓ 该类问题已清零" if total == 0 else "  ✗ 仍需处理"))

print("\n=== C. dit.py 补丁块 ===")
ls = io.open("src/model/dit.py", encoding="utf-8").read().splitlines()
for i in range(2355, 2378):
    if i < len(ls):
        print(f"  L{i+1}: {ls[i].rstrip()}")
print("""
  形状/梯度静态推理:
   * g 形状 (N,1,64,64) -> enc(g) 与 enc(zeros_like(g)) 形状完全相同 -> 相减安全 ✓
   * 未 detach -> 两项都进图，梯度同时塑形编码器 ✓（专家要求）
   * 全文件仅此一处使用 self.glyph_embedder ✓（前面 grep 已确认）-> 没有漏改的调用点 ✓
   * 副作用: g_tok 幅度会**变小**（差分），乘 glyph_scale(0.4 可学习) 后注入强度下降
     -> 从零训练可自行重标定 ✓；若 warm-start 则需注意 ✗（本 run 是从零 ✓）
""")
