# -*- coding: utf-8 -*-
"""preflight.py —— 唯一的启动前闸门（把今天所有踩过的坑固化成自动检查）

设计原则（教训驱动）:
  ① **工具自检**: 每个检查必须先证明"它真的在跑"，否则报 UNKNOWN（我曾把"模块不存在"报成 0 处 ✓✗）
  ② **只增不减是违规**: 常数蔓延/坏味道用棘轮基线，只允许下降（`--update-baseline` 只写更小值）
  ③ **一处改动覆盖一类隐患**: 配置显式性检查覆盖全部 46 处 H3

用法:
  python tools/preflight.py --config <cfg.json>            # 单配置全闸门
  python tools/preflight.py --config <cfg> --strict        # 有高危则 exit 1（用于启动脚本）
  python tools/preflight.py --update-baseline              # 收紧基线
"""
import argparse
import ast
import glob
import io
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = sorted(glob.glob("src/**/*.py", recursive=True))
TOOLS = sorted(glob.glob("tools/**/*.py", recursive=True))
BASE = "tools/analysis/preflight_baseline.json"
RATCHET = {                       # 棘轮指标: 名称 -> 当前实测值（只允许下降）
    "hardcoded_0.18215": None,
    "hardcoded_TIME_SCALE": None,
    "h3_default_mismatch": None,
}
results = []


def rec(level, name, msg):
    results.append((level, name, msg))


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                          encoding="utf-8", errors="replace")


# ───────────────────────── ① 工具自检 ─────────────────────────
def gate_toolcheck():
    r = sh([sys.executable, "-m", "pyflakes", "--version"])
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    ok = r.returncode == 0 and out and "No module named" not in out
    rec("OK" if ok else "UNKNOWN", "pyflakes",
        f"版本={out[:40]}" if ok else f"未安装/不可用({out[:60]}) —— H1 类检查降级为启发式")
    return ok


# ───────────────────────── ①b 语法门（compile() 级）─────────────────────────
def gate_syntax():
    """每个 .py 必须可编译。教训: 曾有全局替换把标识符污染成 'data/skel/std_skel',
    5 个文件语法损坏而旧闸门没拦住（pyflakes 只筛了 undefined 行）。"""
    src_bad, tools_bad = [], []
    for f in SRC:
        try:
            compile(io.open(f, encoding="utf-8-sig", errors="replace").read(), f, "exec")
        except SyntaxError as e:
            src_bad.append(f"{f}:{e.lineno}: {e.msg}")
    for f in TOOLS:
        try:
            compile(io.open(f, encoding="utf-8-sig", errors="replace").read(), f, "exec")
        except SyntaxError as e:
            tools_bad.append(f"{f}:{e.lineno}: {e.msg}")
    rec("FAIL" if src_bad else "OK", "语法 src/", f"{len(src_bad)} 个文件不可编译（要求 0）")
    for b in src_bad[:6]:
        rec("FAIL", "  └", b[:140])
    rec("INFO", "语法 tools/", f"{len(tools_bad)} 个（棘轮）")
    for b in tools_bad[:3]:
        rec("INFO", "  └", b[:140])
    return len(src_bad)


# ───────────────────────── ② pyflakes / H1 ─────────────────────────
def gate_ruff():
    """★ 2026-10-13 ruff 门槛: **src/ 必须 0**（已全清 ✓）; tools/ 历史欠债走棘轮。
    规则集 = F401 未用 import / F541 无用 f-string / F811 重复定义（均安全级）。"""
    def _run(target):
        try:
            r = sh([sys.executable, "-m", "ruff", "check", target])
        except Exception as e:                                    # noqa: BLE001
            return None, f"不可用 ({type(e).__name__})"
        out = (r.stdout or "") + (r.stderr or "")
        if "No module named" in out:
            return None, "未安装（pip install ruff）"
        m = re.search(r"Found (\d+) error", out)
        return (int(m.group(1)) if m else 0), ""
    n_src, err = _run("src")
    if n_src is None:
        rec("UNKNOWN", "ruff", err); return None
    rec("FAIL" if n_src else "OK", "ruff src/", f"{n_src} 条（要求 0，已清理部分不许回退 ✗）")
    n_tools, _ = _run("tools")
    if n_tools is not None:
        rec("INFO", "ruff tools/", f"{n_tools} 条历史欠债（棘轮，只许下降）")
    return n_src


def gate_pyflakes():
    # 注意: Windows 命令行长度上限 —— 传目录而不是逐个文件（149 文件必炸 WinError 206）
    r = sh([sys.executable, "-m", "pyflakes", "src", "tools"])
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    if "No module named" in out:
        rec("UNKNOWN", "pyflakes 扫描", "跳过（未安装）")
        return None
    lines = [l for l in out.splitlines() if l.strip()]
    bad = [l for l in lines if "undefined name" in l or "referenced before assignment" in l]
    # ★ 策略: **src/** 必须零缺漏（那是训练/评测真跑的路）; tools/ 历史欠债只计数走棘轮
    src_bad = [l for l in bad
               if (l.replace("/", os.sep).startswith("src") or l.startswith("src/"))
               and "legacy" not in l.replace("/", os.sep)]
    tools_bad = [l for l in bad if l not in src_bad]
    rec("FAIL" if src_bad else "OK", "pyflakes src/", f"未定义/先引用 {len(src_bad)} 条（要求 0）")
    for l in src_bad[:10]:
        rec("FAIL", "  └", l[:140])
    rec("INFO", "pyflakes tools/", f"{len(tools_bad)} 条历史欠债（棘轮，只许下降）")
    for l in tools_bad[:5]:
        rec("INFO", "  └", l[:140])
    return len(src_bad)


def gate_h1():
    """启发式（pyflakes 不可用时兜底）：分支内赋值 / 外层引用。噪声大 -> 只作计数棘轮。"""
    n = 0
    for f in SRC:
        try:
            tree = ast.parse(io.open(f, encoding="utf-8-sig", errors="replace").read())
        except SyntaxError:
            continue
        lines = io.open(f, encoding="utf-8-sig", errors="replace").read().splitlines()
        for fn in [x for x in ast.walk(tree) if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            A, U = {}, {}
            for x in ast.walk(fn):
                if isinstance(x, ast.Name):
                    (A if isinstance(x.ctx, ast.Store) else U).setdefault(x.id, []).append(x.lineno)
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            for v, ul in U.items():
                if v in args or v not in A:
                    continue
                ind = lambda ln: len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
                if ind(min(ul)) < min(ind(l) for l in A[v]):
                    n += 1
    rec("INFO", "H1 启发式计数", f"{n}（噪声大，仅作棘轮；真信号看 pyflakes）")
    return n


# ───────────────────────── ③ 配置显式性（H3） ─────────────────────────
def _consensus():
    cfg = {}
    for p in glob.glob("src/train/configs/**/*.json", recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        for b in ([d] if isinstance(d, dict) else [x for x in d if isinstance(x, dict)]):
            for k, v in b.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cfg.setdefault(k, set()).add(round(float(v), 6))
    return cfg


def _defaults():
    out = []
    for f in SRC:
        s = io.open(f, encoding="utf-8-sig", errors="replace").read()
        for m in re.finditer(r"getattr\(\s*args\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*,\s*([^)]+)\)", s):
            try:
                dv = ast.literal_eval(m.group(2).strip())
            except Exception:
                continue
            if isinstance(dv, (int, float)) and not isinstance(dv, bool):
                out.append((f, s[:m.start()].count("\n") + 1, m.group(1), float(dv)))
    return out


def gate_config(cfg_path):
    if not cfg_path:
        rec("INFO", "配置显式性", "未指定 --config，跳过")
        return 0
    try:
        d = json.load(open(cfg_path, encoding="utf-8"))
    except Exception as e:
        rec("FAIL", "配置可读", f"{cfg_path}: {e}")
        return 1
    if not isinstance(d, dict):
        rec("INFO", "配置显式性", "非 dict 配置，跳过")
        return 0
    have = {k for k, v in d.items() if isinstance(v, (int, float, bool, str))}
    cons, defs, bad, seen = _consensus(), _defaults(), [], set()
    for f, ln, k, dv in defs:
        if k in have or k in seen:
            continue
        vals = cons.get(k)
        if vals and len(vals) == 1:
            (cv,) = tuple(vals)
            if abs(cv - dv) > 1e-12:
                seen.add(k)
                bad.append(f"{k}: 未显式给出, 代码默认={dv}, 其它配置共识={cv}  ({f}:{ln})")
    rec("FAIL" if bad else "OK", "配置显式性(H3)", f"{len(bad)} 项高危")
    for b in bad[:10]:
        rec("FAIL", "  └", b[:150])
    return len(bad)


# ───────────────────────── ④ 常数棘轮 ─────────────────────────
def gate_ratchet():
    txts = {f: io.open(f, encoding="utf-8-sig", errors="replace").read() for f in SRC + TOOLS}
    # ★ 只统计**代码里的 token**，排除注释/字符串/文档里的"提及" ✓
    #   （否则"单一来源"文件自己声明的 SD_SF=0.18215、以及 linter 的 pattern
    #     都会被算成违规 ✗ —— 那不是蔓延，是定义本身）
    import tokenize as _tk

    def _count_tok(text, kind, lit):
        n = 0
        try:
            for tok in _tk.generate_tokens(io.StringIO(text).readline):
                if tok.type == kind and tok.string.replace("_", "") == lit:
                    n += 1
        except Exception:                                    # noqa: BLE001
            return text.count(lit)
        return n

    def _count_number(text, lit):
        return _count_tok(text, _tk.NUMBER, lit)

    def _count_name(text, lit):
        # TIME_SCALE 既可能是标识符也可能作为字符串键出现 —— 两种都算 ✓
        return _count_tok(text, _tk.NAME, lit) + _count_tok(text, _tk.STRING, lit)

    counts = {
        "hardcoded_0.18215": sum(_count_number(s, "0.18215") for s in txts.values()),
        "hardcoded_TIME_SCALE": sum(_count_name(s, "TIME_SCALE") for s in txts.values()),
    }
    base = {}
    if os.path.exists(BASE):
        base = json.load(open(BASE, encoding="utf-8"))
    for k, v in counts.items():
        b = base.get(k)
        if b is None:
            rec("INFO", f"棘轮 {k}", f"首次记录 = {v}")
        elif v > b:
            rec("FAIL", f"棘轮 {k}", f"**增加**了: {b} -> {v}（只许下降）")
        else:
            rec("OK", f"棘轮 {k}", f"{b} -> {v}" + ("（已收紧 ✓）" if v < b else ""))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    a = ap.parse_args()

    gate_toolcheck()
    gate_syntax()
    gate_ruff()
    gate_pyflakes()
    h1 = gate_h1()
    h3 = gate_config(a.config)
    counts = gate_ratchet()
    counts["h3_default_mismatch"] = h3
    counts["h1_heuristic"] = h1

    if a.update_baseline:
        old = json.load(open(BASE, encoding="utf-8")) if os.path.exists(BASE) else {}
        merged = {k: (v if (k not in old or v < old[k]) else old[k]) for k, v in counts.items()}
        json.dump(merged, open(BASE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        rec("INFO", "基线", f"已更新 -> {BASE}（只收紧不放松）")

    icon = {"OK": "✓", "FAIL": "✗", "UNKNOWN": "?", "INFO": "·"}
    print("=" * 78)
    for lv, name, msg in results:
        print(f" {icon.get(lv,'?')} [{lv:7}] {name:22} {msg}")
    nf = sum(1 for lv, _, _ in results if lv == "FAIL")
    nu = sum(1 for lv, _, _ in results if lv == "UNKNOWN")
    print("=" * 78)
    print(f"  FAIL={nf}  UNKNOWN={nu}   （UNKNOWN = 该检查没真正跑起来，不得当作通过 ✗）")
    if a.strict and nf:
        sys.exit(1)


if __name__ == "__main__":
    main()
