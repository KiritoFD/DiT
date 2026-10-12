# -*- coding: utf-8 -*-
"""validate_config.py — 启动闸门：把"配置漏写 -> 静默用错默认"变成**启动前硬失败**

原理（唯一一条规则，覆盖 46 处 H3 隐患）:
  对代码里每个 `getattr(args, '<k>', <default>)`：
    · 若 <k> **在 config 里显式给出**                -> 安全 ✓
    · 若 <k> 不在 config 里，但代码 default 与**其它配置的共识值一致** -> 安全 ✓（只是省写）
    · 若 <k> 不在 config 里，且 default != 共识值    -> **高危** ✗（静默用错默认）

用法:
  python tools/analysis/validate_config.py --config src/train/configs/v83A_stdsoft_zerofix.json
  python tools/analysis/validate_config.py --config <cfg> --strict      # 有高危则 exit 1
  python tools/analysis/validate_config.py --all                        # 扫全部配置
"""
import argparse
import ast
import glob
import io
import json
import os
import re
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SRC = sorted(glob.glob("src/**/*.py", recursive=True))


def consensus():
    """键 -> 其它配置里出现的数值集合"""
    cfg = defaultdict(set)
    for p in glob.glob("src/train/configs/**/*.json", recursive=True):
        try:
            d = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        blobs = [d] if isinstance(d, dict) else [x for x in d if isinstance(x, dict)]
        for b in blobs:
            for k, v in b.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    cfg[k].add(round(float(v), 6))
    return cfg


def defaults():
    """代码里 (file, line, key, default) 列表"""
    out = []
    for f in SRC:
        s = io.open(f, encoding="utf-8", errors="replace").read()
        for m in re.finditer(r"getattr\(\s*args\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*,\s*([^)]+)\)", s):
            k, d = m.group(1), m.group(2).strip()
            try:
                dv = ast.literal_eval(d)
            except Exception:
                continue
            if isinstance(dv, (int, float)) and not isinstance(dv, bool):
                out.append((f, s[:m.start()].count("\n") + 1, k, float(dv)))
    return out


def check(cfg_path, cons, defs):
    try:
        d = json.load(open(cfg_path, encoding="utf-8"))
    except Exception as e:
        return [("PARSE", f"{cfg_path}: {e}")]
    if not isinstance(d, dict):
        return [("SKIP", f"{cfg_path}: 非 dict")]
    have = {k for k, v in d.items() if isinstance(v, (int, float, bool, str))}
    bad = []
    seen = set()
    for f, ln, k, dv in defs:
        if k in have or k in seen:
            continue
        vals = cons.get(k)
        if not vals:
            continue
        if len(vals) == 1:                      # 配置共识值唯一
            (cv,) = tuple(vals)
            if abs(cv - dv) > 1e-12:            # 而代码默认与之不同 -> 高危
                seen.add(k)
                bad.append(("HIGH", f"{k}: config 未给, 代码默认={dv}, 其它配置共识={cv}  ({f}:{ln})"))
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--strict", action="store_true")
    a = ap.parse_args()
    cons, defs = consensus(), defaults()
    cfgs = ([a.config] if a.config else
            sorted(glob.glob("src/train/configs/*.json")) if a.all else [])
    if not cfgs:
        print("  用 --config <json> 或 --all"); return
    total = 0
    for c in cfgs:
        bad = check(c, cons, defs)
        high = [x for x in bad if x[0] == "HIGH"]
        total += len(high)
        if high or a.config:
            print(f"\n=== {os.path.basename(c)} : 高危 {len(high)} ===")
            for _, msg in high:
                print("   ✗ " + msg)
        if a.config and not high:
            print(f"\n  ✓ {os.path.basename(c)}: 无「配置漏写->静默默认」风险")
    print(f"\n  合计高危 {total}")
    if a.strict and total:
        sys.exit(1)


if __name__ == "__main__":
    main()
