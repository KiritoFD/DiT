# -*- coding: utf-8 -*-
"""audit_repo.py — 针对"本项目已真实发生过的静默 bug"做系统性审查

规则集（每条都对应一次真实事故）:
  H1 分支内赋值 / 外层引用        -> _repa_cache_dir 未绑定回归 (2026-10-12)
  H2 硬编码的"必须随变量走"的常数  -> sf=0.18215 抄 SD, 未随 VAE 重拟合 (E5)
  H3 getattr(args, k, default) 的默认值 与某些 config 的实际值不一致 -> 静默用错默认 (E5 同类)
  H4 同一约定常数在多文件重复      -> 0.18215 散落 11 个编码脚本 (E5 扩散)
  H5 模块级单槽缓存                -> load_eval_vae 只按 device 缓存 (E3); _get_vae 就地打补丁 (E9)
  H6 就地改写共享对象              -> 给缓存共享的 VAE 打 decode 补丁 (E9)

用法: python tools/analysis/audit_repo.py [--json out.json]
"""
import ast
import glob
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SRC = sorted(glob.glob("src/**/*.py", recursive=True))
findings = defaultdict(list)


def rd(p):
    return io.open(p, encoding="utf-8", errors="replace").read()


# ── H1: 分支内赋值 / 外层引用（要求**所有分支都赋值**才算安全 -> 这里只报"存在更浅引用"的候选） ──
def h1():
    for f in SRC:
        try:
            tree = ast.parse(rd(f))
        except SyntaxError:
            continue
        lines = rd(f).splitlines()
        for fn in [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
            A, U = {}, {}
            for n in ast.walk(fn):
                if isinstance(n, ast.Name):
                    (A if isinstance(n.ctx, ast.Store) else U).setdefault(n.id, []).append(n.lineno)
            args = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
            for v, ul in U.items():
                if v in args or v not in A:
                    continue
                ind = lambda ln: len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
                if ind(min(ul)) < min(ind(l) for l in A[v]):
                    findings["H1_分支赋值外层引用"].append(f"{f}:{min(ul)} {v} in {fn.name}()")


# ── H2/H4: 必须随变量走的常数，且在多文件重复 ──
CONST_PAT = {
    "0.18215(VAE sf)": r"0\.18215",
    "0.3611/0.1159(FLUX sf/shift)": r"0\.(3611|1159)",
    "TIME_SCALE": r"TIME_SCALE",
    "sf_字面赋值": r"scaling_factor\s*=\s*0\.\d+",
}


def h2h4():
    hits = defaultdict(list)
    for f in SRC + sorted(glob.glob("tools/**/*.py", recursive=True)):
        s = rd(f)
        for name, pat in CONST_PAT.items():
            for m in re.finditer(pat, s):
                ln = s[:m.start()].count("\n") + 1
                hits[name].append(f"{f}:{ln}")
    for name, lst in hits.items():
        if len(lst) >= 2 or name.startswith("0.18215"):
            findings["H2H4_硬编码常数"].append(f"{name}: {len(lst)} 处 -> {lst[:6]}")


# ── H3: getattr(args, k, default) 的默认值与 config 实际值不一致 ──
def h3():
    cfg = defaultdict(set)
    for p in glob.glob("src/train/configs/**/*.json", recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        blobs = []
        if isinstance(d, dict):
            blobs.append(d)
        elif isinstance(d, list):
            blobs += [x for x in d if isinstance(x, dict)]
        for blob in blobs:
            for k, v in blob.items():
                if isinstance(v, (int, float, bool, str)):
                    cfg[k].add(v)
    for f in SRC:
        s = rd(f)
        for m in re.finditer(r"getattr\(\s*args\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*,\s*([^)]+)\)", s):
            k, dflt = m.group(1), m.group(2).strip()
            vals = cfg.get(k)
            if not vals:
                continue
            ln = s[:m.start()].count("\n") + 1
            try:
                dv = ast.literal_eval(dflt)
            except Exception:
                continue
            if not isinstance(dv, (int, float)) or isinstance(dv, bool):
                continue
            nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums and all(abs(v - dv) > 1e-12 for v in nums):
                findings["H3_getattr默认值与config不一致"].append(
                    f"{f}:{ln} key={k} default={dv} config值={sorted(nums)[:4]}")


# ── H5/H6: 模块级缓存 / 就地改写 ──
def h5h6():
    for f in SRC:
        s = rd(f)
        for m in re.finditer(r"^_[A-Z_]+\s*=\s*None\s*$", s, re.M):
            ln = s[:m.start()].count("\n") + 1
            findings["H5_模块级单槽缓存"].append(f"{f}:{ln} {s.splitlines()[ln-1].strip()}")
        for m in re.finditer(r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\.(decode|encode)\s*=\s*", s, re.M):
            ln = s[:m.start()].count("\n") + 1
            findings["H6_就地改写对象方法"].append(f"{f}:{ln} {s.splitlines()[ln-1].strip()[:90]}")


h1(); h2h4(); h3(); h5h6()

# ── pyflakes 全仓 ──
try:
    r = subprocess.run([sys.executable, "-m", "pyflakes", *SRC], capture_output=True, text=True, timeout=300)
    pf = [l for l in ((r.stdout or "") + (r.stderr or "")).splitlines() if l.strip()]
    findings["PYFLAKES"] = pf[:40]
except Exception as e:
    findings["PYFLAKES"] = [f"(不可用: {type(e).__name__})"]

print(f"  扫描 {len(SRC)} 个源文件\n")
for k in sorted(findings):
    v = findings[k]
    print(f"=== {k}  ({len(v)}) ===")
    for x in v[:14]:
        print("   " + x[:150])
    if len(v) > 14:
        print(f"   ... 另有 {len(v)-14} 条")
    print()
if "--json" in sys.argv:
    out = sys.argv[sys.argv.index("--json") + 1]
    json.dump({k: v for k, v in findings.items()}, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  -> {out}")
