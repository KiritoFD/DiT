#!/usr/bin/env python
"""度量"可抽取块"的依赖 —— 重构选点的**证据工具**（2026-10-13 ✓）。

★ 由来: 我在 PR3/PR4 抽取时, 需要反复回答"这个方法/这段代码依赖什么、边界在哪" ✗,
  而用临时 `python -c` 脚本度量**连续失败 4 次** ✗（跨行签名 / 片段缩进 / 嵌套 if 边界 ✓）。
  ⇒ 把该能力固化成工具 ✓：只做**只读分析**，不修改任何文件 ✓。

用法:
    python tools/measure_deps.py src/model/dit.py --method DiT_2Cond._fuse_condition
    python tools/measure_deps.py src/train/train.py --contains-line 1484
    python tools/measure_deps.py src/model/dit.py --list-methods DiT_2Cond

输出: 目标跨度、语句数、**enclosing 顶层语句**（防"锚点落在内层"✗）、
      以及 `self.X` 依赖分类（nn.Module vs 其它 ✓ —— 前者必须显式传入 ✓）。
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path


def _assign_call_attrs(cls: ast.ClassDef) -> set:
    """`__init__` 里赋值为**调用**的 self.X = nn.Module/子模块（需显式传入 ✓）。"""
    out = set()
    for n in cls.body:
        if isinstance(n, ast.FunctionDef) and n.name == "__init__":
            for x in ast.walk(n):
                if isinstance(x, ast.Assign) and isinstance(x.value, ast.Call):
                    for tg in x.targets:
                        if isinstance(tg, ast.Attribute) and getattr(tg.value, "id", "") == "self":
                            out.add(tg.attr)
    return out


def _self_attrs(node: ast.AST) -> set:
    return {x.attr for x in ast.walk(node)
            if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name)
            and x.value.id == "self"}


def _find_class(tree: ast.AST, name: str):
    return next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == name), None)


def _find_func(scope, name: str):
    return next((n for n in scope.body
                 if isinstance(n, ast.FunctionDef) and n.name == name), None)


def _enclosing_top_level(fn: ast.FunctionDef, lineno: int):
    """找出 `fn` 里**包含**该行的顶层语句 ✓ —— 防止锚点落进内层 if ✗。"""
    for st in fn.body:
        end = st.end_lineno or st.lineno
        if st.lineno <= lineno <= end:
            return st
    return None


def _report_func(src_lines, fn: ast.FunctionDef, cls: ast.ClassDef | None, title: str):
    body_lo = fn.body[0].lineno if fn.body else fn.lineno
    body_hi = fn.body[-1].end_lineno if fn.body else fn.end_lineno
    print(f"  {title}")
    print(f"    def 跨度      : L{fn.lineno}..L{fn.end_lineno}"
          f"（{fn.end_lineno - fn.lineno + 1} 行；签名是否跨行: "
          f"{'是 ✓' if fn.body and fn.body[0].lineno > fn.lineno + 1 else '否'}）")
    print(f"    语句体跨度    : L{body_lo}..L{body_hi}（{len(fn.body)} 条顶层语句 ✓）")
    print(f"    参数          : {[a.arg for a in fn.args.args]}")
    # ★ 语句体必须能独立 parse（否则不可整块搬 ✓）—— 用 if True: 包裹（不需 dedent ✓）
    seg = "\n".join(src_lines[body_lo - 1:body_hi])
    try:
        ast.parse("if True:\n" + seg)
        print("    ✓ 语句体可独立 parse（可整块搬移 ✓）")
    except SyntaxError as e:
        print(f"    ✗ 语句体**不可**独立 parse: {e}（说明边界跨层 ✗）")
    if cls is not None:
        mods = _assign_call_attrs(cls)
        used = _self_attrs(ast.Module(body=fn.body, type_ignores=[]))
        mm = sorted(a for a in used if a in mods)
        other = sorted(a for a in used if a not in mods)
        print(f"    nn.Module 依赖 ({len(mm)}): {mm}")
        print(f"    其它 self.X   ({len(other)}): {other}")
        # ★★ 抽取陷阱扫描（2026-10-13 实测踩到 ✗ ⇒ 固化到工具 ✓✓）
        #   ① `getattr(self, "X", DEFAULT)` —— 纯 `self.X` 替换**抓不到** ✗
        #      且 `from_model` 的默认值**必须与原 DEFAULT 一致** ✗（否则 None>0 会 TypeError ✓）
        g = [(x.lineno, ast.unparse(x)) for x in ast.walk(ast.Module(body=fn.body, type_ignores=[]))
             if isinstance(x, ast.Call) and getattr(x.func, "id", "") == "getattr" and x.args
             and isinstance(x.args[0], ast.Name) and x.args[0].id == "self"]
        if g:
            print(f"    ⚠ **getattr(self, ...) 模式 {len(g)} 处**（替换 self.X 抓不到 ✗；"
                  f"from_model 的默认值必须与原默认一致 ✓）:")
            for ln, txt in g:
                print(f"        L{ln}: {txt[:88]}")
        #   ② 惰性访问的属性（某分支才读 ✓）—— 若 from_model **急切**读取会 AttributeError ✗
        print("    ⚠ 惰性访问提示: 上列 self.X 中, 有些只在**某分支**才被读 ✓ ⇒")
        print("      from_model 必须用 getattr(m, X, None) 兜底 ✗（否则急切求值会崩 ✓）")
    # 每条 ≥8 行的顶层语句（看结构 ✓）
    big = [(i, type(st).__name__, st.lineno, (st.end_lineno or st.lineno) - st.lineno + 1)
           for i, st in enumerate(fn.body)
           if (st.end_lineno or st.lineno) - st.lineno + 1 >= 8]
    if big:
        print("    ≥8 行的顶层语句:")
        for i, k, ln, n in big[:12]:
            print(f"      {i:2d}) {k:10s} L{ln:<6d} {n:4d} 行 | {src_lines[ln - 1].strip()[:60]}")


def main() -> int:
    ap = argparse.ArgumentParser(description="度量可抽取块的依赖（只读 ✓）")
    ap.add_argument("path")
    ap.add_argument("--method", help="Class.method 或 function")
    ap.add_argument("--contains-line", type=int, help="报告包含该行的 enclosing 顶层语句")
    ap.add_argument("--list-methods", help="列出该类里 ≥20 行的方法")
    args = ap.parse_args()

    p = Path(args.path)
    src = p.read_text(encoding="utf-8")
    lines = src.split("\n")
    tree = ast.parse(src)
    print(f"文件: {p}（{len(lines)} 行 ✓）")

    if args.list_methods:
        cls = _find_class(tree, args.list_methods)
        if cls is None:
            print("  找不到该类 ✗")
            return 1
        print(f"  类 {args.list_methods}（L{cls.lineno}..L{cls.end_lineno}）≥20 行的方法:")
        for n in cls.body:
            if isinstance(n, ast.FunctionDef):
                ln = n.end_lineno - n.lineno + 1
                if ln >= 20:
                    print(f"    {n.name:26s} L{n.lineno:<6d} {ln:4d} 行")
        return 0

    if args.contains_line:
        for n in tree.body:
            if isinstance(n, ast.FunctionDef):
                st = _enclosing_top_level(n, args.contains_line)
                if st is not None:
                    print(f"  包含 L{args.contains_line} 的顶层语句: {type(st).__name__} "
                          f"L{st.lineno}..L{st.end_lineno}"
                          f"（{(st.end_lineno or st.lineno) - st.lineno + 1} 行, "
                          f"缩进 {len(lines[st.lineno - 1]) - len(lines[st.lineno - 1].lstrip())}）")
                    print(f"    首行: {lines[st.lineno - 1].strip()[:80]}")
        return 0

    if args.method:
        if "." in args.method:
            cname, mname = args.method.split(".", 1)
            cls = _find_class(tree, cname)
            fn = _find_func(cls, mname) if cls else None
        else:
            cls = None
            fn = next((n for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name == args.method), None)
        if fn is None:
            print("  找不到该方法/函数 ✗")
            return 1
        _report_func(lines, fn, cls, args.method)
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
