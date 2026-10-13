# -*- coding: utf-8 -*-
"""CLIP-S / DINO-S 完整趋势：6 个 step + 3 条基线（本地 GPU ✓ 标准 ViT-L/14 ✓）。"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "style_eval_local", "stylesim_trend")
os.makedirs(OUT, exist_ok=True)
STEPS = ["step0020000", "step0050000", "step0080000", "step0100000", "step0115000", "step0120000"]
GENS = []
for s in STEPS:
    GENS += ["--gen", "%s=assets/style_eval_src/%s" % (s, s)]
ENC = [("dinov2_vitl14", "DINO-S"), ("clip_vitl14", "CLIP-S")]
MAXREF = os.environ.get("MAXREF", "4000")


def run(enc, tag):
    out = os.path.join(OUT, tag)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "style_eval.py"),
           "--enc", enc, "--pool", "cls",
           "--csv", "exp-std/csv/train.csv", "--root", ".", "--root", "data",
           "--gen-map", "exp-std/csv/eval200_fixed.csv", *GENS,
           "--out", out, "--max-ref", MAXREF, "--baselines", "1", "--bsd", "0",
           "--batch", "24"]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                       timeout=10800)
    p = os.path.join(out, "style_report.json")
    if r.returncode != 0 or not os.path.isfile(p):
        print("  ✗ %s exit=%d %s" % (tag, r.returncode,
                                     (r.stderr or "").strip().split("\n")[-1][:100]))
        return None
    print(r.stdout.strip().split("\n")[-6:][0] if r.stdout else "")
    return json.load(open(p, encoding="utf-8"))


for enc, name in ENC:
    print("=== %s (%s) ===" % (name, enc), flush=True)
    d = run(enc, name)
    if not d:
        continue
    bl = d.get("baselines", {})
    def s(x):
        return (x or {}).get("style_sim", {}).get("s_all", float("nan"))
    ceil = s(bl.get("real_real_loo"))
    realx = s(bl.get("real_real"))
    wrong = s(bl.get("real_other_writer"))
    ttf = s(bl.get("ttf_only"))
    print("  上限 real_real_loo = %.4f | 跨作品 real = %.4f | 错书家 = %.4f | TTF-only = %.4f"
          % (ceil, realx, wrong, ttf))
    print("  %-12s %9s %9s %9s %9s %9s" % ("step", name, "归一化", "mAP", "XCTAcc", "HWD-CN"))
    rows = []
    for st in STEPS:
        g = d["sets"].get(st, {})
        sv = s(g)
        # 归一化到 [0,1]: (生成 − TTF) / (真上限 − TTF) ✓（跨编码器可比 ✓）
        norm = ((sv - ttf) / (ceil - ttf)) if (ceil == ceil and ttf == ttf and ceil != ttf) else float("nan")
        rows.append((st, sv, norm, g.get("retrieval", {}).get("map", float("nan")),
                     g.get("xctacc", {}).get("xctacc", float("nan")),
                     g.get("hwd_cn", {}).get("macro", float("nan"))))
        print("  %-12s %9.4f %9.3f %9.4f %9.4f %9.3f" % (st, sv, norm, rows[-1][3], rows[-1][4],
                                                         rows[-1][5]))
    json.dump({"encoder": enc, "name": name, "ceiling": ceil, "real_crosswork": realx,
               "wrong_writer": wrong, "ttf_only": ttf, "steps": rows},
              open(os.path.join(OUT, "%s_summary.json" % name), "w", encoding="utf-8"),
              ensure_ascii=False, default=float)
print("\n[done] %s" % OUT)
