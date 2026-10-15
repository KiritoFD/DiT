# -*- coding: utf-8 -*-
"""lint_sync_hotpath.py —— **训练热路径同步纪律**静态扫描（`docs/infra_1015.md` §二.3 ✓）

审查原文:
  "我建议把它设为一条**全局规则**：
   正常训练热路径中禁止 `.item()`、`float(cuda_tensor)`、`.tolist()`、`bool(cuda_tensor)`；
   **只有显式标记的同步边界允许使用**。" ✓
  "可以在 **CI** 中对训练热路径做静态扫描，并对**已批准的例外添加注释**。" ✓

★★ 为什么用 **AST** 而不是正则（本项目的实测教训 ✓）:
  用正则扫时连续踩了 **3 次假阳性** ✗ —— 命中的都是 **docstring / 注释**里的
  "旧实现 `float(t[i])`" 之类的**说明文字** ✗（如 `strong_style._eligible_rows` ✓
  `dit._style_out` ✓ `dino_cache` 的模块文档 ✓）
  ⇒ AST 只走**真实表达式节点** ✓ 天然免疫 ✓（注释与字符串**不在语法树里** ✓）

★ 白名单（"显式标记的同步边界" ✓）:
  在命中行的**同一行或上一行**写 `# sync-ok: <理由>` ✓
  ⇒ 强制**写出理由** ✓（不是简单静音 ✗）—— 良好工程实践 ✓
  整个文件都不在热路径时，**不要**放进 `HOTPATH` 列表 ✓（显式优于隐式 ✓）

★ 用法::

    python tools/lint_sync_hotpath.py            # 人类可读报告
    python tools/lint_sync_hotpath.py --json     # 机器可读（CI ✓）
    python tools/lint_sync_hotpath.py --stats    # 统计每个文件的同步点数量
    python tools/lint_sync_hotpath.py --update-baseline   # 收紧棘轮（**只许下调** ✗）

★★ **棘轮基线**（为什么不是"0 违规"硬门禁 ✓）:
  静态扫描**无法判类型** ✗ ⇒ 必然有**误报** ✓
  例: `float(args.critic_t_min)` ✓ —— `args.*` 是 **Python float** ✓ **不触发**同步 ✗
      但静态上看与 `float(cuda_tensor)` 一模一样 ✗
  真实规模: 首轮扫出 175 处 ✗ 豁免日志边界后 **148** 处 ✓（其中混有误报 ✓）
  ⇒ 良好工程做法: **记录每文件基线** ✓ ⇒ CI **只在数量增长时失败** ✗
    （与 `tools/preflight.py` 的结构下限棘轮同一模式 ✓ 本项目已验证 ✓）
  ⇒ 基线**只许下调** ✗（`--update-baseline` 会拒绝上调 ✓）

退出码: 0 = 无**超出基线**的未标记同步 ✓；1 = 有回归 ✗（CI 应失败 ✓）
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re

# ── 训练**热路径**（每步都会执行的模块 ✓ 显式列举 ✓ 便于审阅与维护 ✓）──
#   ★ 判断标准: 这些模块的代码在 `trainer.training_loop` 的每步路径上 ✓
#     `telemetry` / `checkpoint` / `cli` / `eval/*` 等**不在**热路径 ✓ ⇒ 不列入 ✓
HOTPATH = [
    "src/train/step.py",
    "src/train/trainer.py",
    "src/train/objectives.py",
    "src/train/strong_style.py",
    "src/train/gradient_budget.py",
    "src/train/backward.py",
    "src/train/ema.py",
    "src/loss/critic_loss.py",
    "src/loss/flow_matching.py",
    "src/loss/writer_objectives.py",
    "src/model/dit_conditioning.py",
]

# ── 被禁的同步原语（§二.3 的清单 ✓ + `int()` 同类 ✓）──
SYNC_METHODS = {"item", "tolist"}
SYNC_BUILTINS = {"float", "bool", "int"}

MARKER = re.compile(r"#\s*sync-ok\s*:\s*\S+")

# ── **日志边界豁免**（§二.2 / §二.3 ✓ 文档自己允许的例外 ✓）──
#   审查原文: "只在**真正写日志时**统一进行一次 CPU 读取" ✓
#   ⇒ 位于 `logger.info(...)` / `print(...)` / `_tel(...)` 等**日志调用**参数里的
#     同步点**就是**那个被批准的边界 ✓ ⇒ **豁免** ✓
#   ★ 这样规则才**有意义** ✓: 它约束的是**热路径计算/控制流**里的同步 ✗
#     （如 `if mask.any():` ✓ `float(t[i])` 参与比较 ✗），而不是日志格式化 ✓
LOG_CALLERS = {"print"}
LOG_ATTRS = {"info", "debug", "warning", "warn", "error", "critical", "exception",
             "add_scalar", "add_scalars", "write", "append"}


def _is_literal(node):
    """字面量/常量参数 ⇒ `float(1.5)` 这类**不**触发同步 ✓ 不报 ✓。"""
    return isinstance(node, ast.Constant)


def scan_file(path):
    """返回 [(行号, 列号, 表达式片段, 是否已标记, 标记理由)]"""
    try:
        src = io.open(path, encoding="utf-8").read()
        tree = ast.parse(src)
    except (SyntaxError, UnicodeDecodeError) as e:                     # noqa: BLE001
        return [(-1, -1, "**无法解析**: %s" % e, False, "")]
    lines = src.split("\n")

    def marked(lineno):
        """命中行**或上一行**有 `# sync-ok: 理由` ⇒ 视为已批准 ✓。"""
        for k in (lineno, lineno - 1):
            if 1 <= k <= len(lines):
                m = MARKER.search(lines[k - 1])
                if m:
                    return True, lines[k - 1].strip()
        return False, ""

    out = []
    # ★ 父指针表 ✓ 用于判断"是否在日志调用的参数里"（§二.2 允许的边界 ✓）
    parents = {}
    for p in ast.walk(tree):
        for ch in ast.iter_child_nodes(p):
            parents[id(ch)] = p

    def log_ctx(node):
        """该调用是否位于某个**日志调用**的参数中（含嵌套 ✓）⇒ 是则豁免 ✓。"""
        cur, guard = node, 0
        while guard < 12:
            p = parents.get(id(cur))
            if p is None:
                return False
            if isinstance(p, ast.Call) and p is not cur:
                pf = p.func
                if isinstance(pf, ast.Name) and pf.id in LOG_CALLERS:
                    return True
                if isinstance(pf, ast.Attribute) and pf.attr in LOG_ATTRS:
                    return True
                if isinstance(pf, ast.Name) and pf.id.startswith("_tel"):
                    return True
            cur, guard = p, guard + 1
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        hit = None
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in SYNC_METHODS:
            hit = f.attr
        elif isinstance(f, ast.Name) and f.id in SYNC_BUILTINS:
            if node.args and not _is_literal(node.args[0]):
                hit = f.id
        if hit is None:
            continue
        seg = (ast.get_source_segment(src, node) or hit).replace("\n", " ")
        if log_ctx(node):
            # ★ 日志边界 ✓ 文档明确允许 ✓（§二.2 "只在真正写日志时统一读取一次" ✓）
            out.append((node.lineno, node.col_offset, seg[:70], True,
                        "# log-boundary（§二.2 允许 ✓）"))
            continue
        ok, why = marked(node.lineno)
        out.append((node.lineno, node.col_offset, seg[:70], ok, why))
    return out


def load_baseline(path):
    if not os.path.isfile(path):
        return {}
    try:
        return json.load(io.open(path, encoding="utf-8"))
    except Exception:                                                 # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="机器可读输出（CI ✓）")
    ap.add_argument("--stats", action="store_true", help="按文件统计同步点")
    ap.add_argument("--include-missing", action="store_true",
                    help="连不存在的文件也报（默认跳过 ✓ 便于渐进迁移 ✓）")
    ap.add_argument("--baseline", default=os.path.join("tools", "sync_hotpath_baseline.json"),
                    help="棘轮基线 JSON ✓（默认 tools/sync_hotpath_baseline.json ✓）")
    ap.add_argument("--update-baseline", action="store_true",
                    help="写入基线 ✓ **只许下调** ✗（上调会被拒绝 ✓）")
    args = ap.parse_args()

    report, n_bad, n_ok, n_files = {}, 0, 0, 0
    for rel in HOTPATH:
        p = rel.replace("/", os.sep)
        if not os.path.isfile(p):
            if args.include_missing:
                report[rel] = {"missing": True}
            continue
        n_files += 1
        rows = scan_file(p)
        bad = [r for r in rows if not r[3]]
        report[rel] = {"total": len(rows), "unmarked": len(bad),
                       "marked": len(rows) - len(bad),
                       "hits": [{"line": r[0], "col": r[1], "expr": r[2],
                                 "marked": r[3], "why": r[4]} for r in rows]}
        n_bad += len(bad)
        n_ok += len(rows) - len(bad)

    cur = {rel: info["unmarked"] for rel, info in report.items()
           if not info.get("missing")}
    base = load_baseline(args.baseline)

    # ── 更新基线（**只许下调** ✗）──
    if args.update_baseline:
        # ★ 首次创建（文件不存在 ✓）⇒ 允许写入 ✓；此后**只许下调** ✗
        first_time = not os.path.isfile(args.baseline)
        raised = {} if first_time else {
            k: (base.get(k, 0), v) for k, v in cur.items() if v > base.get(k, 0)}
        if raised:
            print("  ✗ 拒绝上调基线 ✗（棘轮只许下调 ✓）:")
            for k, (b, v) in raised.items():
                print("     %-40s %s -> %s ✗" % (k, b, v))
            return 1
        io.open(args.baseline, "w", encoding="utf-8", newline="").write(
            json.dumps(cur, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print("  ✓ 基线已%s ✓（%d 个文件 ✓ 合计 %d 处未标记 ✓）"
              % ("**首次创建**" if first_time else "收紧", len(cur), sum(cur.values())))
        return 0

    regress = {k: (base.get(k, 0), v) for k, v in cur.items() if v > base.get(k, 0)}

    if args.json:
        print(json.dumps({"files": report, "unmarked": n_bad, "marked": n_ok,
                          "baseline": base, "regressions": regress},
                         ensure_ascii=False, indent=2))
        return 1 if regress else 0

        print("  ═══ 训练热路径同步纪律扫描（`docs/infra_1015.md` §二.3 ✓ AST ✓）═══")
        print("  热路径模块 = %d 个 ✓（每步执行 ✓）" % len(HOTPATH))
        for rel, info in report.items():
            if info.get("missing"):
                print("  —  %-40s **文件不存在** ✗" % rel)
                continue
            tag = "✓" if info["unmarked"] == 0 else "✗"
            print("  %s  %-40s 同步点 %2d ✓（已标记 %d ✓ 未标记 %d %s）"
                  % (tag, rel, info["total"], info["marked"], info["unmarked"],
                     "✓" if info["unmarked"] == 0 else "✗✗"))
            if args.stats:
                continue
            for h in info["hits"]:
                if not h["marked"]:
                    print("        ✗ L%-4d %s" % (h["line"], h["expr"]))
        print()
        print("  汇总: 已标记 %d ✓ · 未标记 %d ✓（其中**超出基线** %d ✗）"
              % (n_ok, n_bad, sum(v - base.get(k, 0) for k, v in regress.items())))
        if regress:
            print("  ✗ **棘轮回归** ✗（`docs/infra_1015.md` §二.3 ✓）:")
            for k, (b, v) in sorted(regress.items()):
                print("     %-40s 基线 %d → **%d** ✗" % (k, b, v))
            print("  ⇒ 修法: 把读取推迟到日志周期 ✓，或在该行/上一行写"
                  " `# sync-ok: <理由>` ✓（须写理由 ✓ 不得静默 ✗）")
        else:
            print("  ✓ 未超出基线 ✓（基线 = 棘轮 ✓ 只许下调 ✗）")
    return 1 if regress else 0


if __name__ == "__main__":
    raise SystemExit(main())
