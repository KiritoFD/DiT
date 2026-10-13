# -*- coding: utf-8 -*-
"""moyi(48) vs 我们(callig-DiT) 的双轴对比：DINO-S / CLIP-S + 检索 mAP + XCTAcc + HWD-CN + BSD。

全部用**同一评测集**(eval200_fixed, n=187 ✓) 与同一协议(上限=留一法 ✓, TTF-only 下限 ✓)。
"""
import json
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "style_eval_local", "moyi_vs_ours")
os.makedirs(OUT, exist_ok=True)

SETS = [
    ("moyi_top10_rf", "assets/style_eval_src/moyi_top10_rf"),
    ("moyi_top10_rf_4ch", "assets/style_eval_src/moyi_top10_rf_4ch"),
    ("v79@75k(我们,风格最佳)", "assets/results/v79_callivae_physics_sp_c2ot/"
                              "20261011-133610-v79-callivae-physics-sp-c2ot/eval_samples_ctrl/"
                              "step0075000/eval200fix"),
    ("v83A@105k(我们)", "assets/results/v83A_stdsoft_zerofix/"
                        "20261012-213952-v83A-stdsoft-zerofix/eval_samples_ctrl/"
                        "step0105000/eval200fix"),
    ("v84A_std@120k(我们,最新)", "assets/style_eval_src/step0120000"),
]
ARGS = []
for n, d in SETS:
    if not os.path.isdir(os.path.join(ROOT, d)):
        print("  ⚠ 缺 %s (%s) ✗" % (n, d))
        continue
    ARGS += ["--gen", "%s=%s" % (n, d)]

ENC = [("dinov2_vitl14", "DINO-S"), ("clip_vitl14", "CLIP-S")]
results = {}
for enc, name in ENC:
    out = os.path.join(OUT, name)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "style_eval.py"),
           "--enc", enc, "--pool", "cls",
           "--csv", "exp-std/csv/train.csv", "--root", ".", "--root", "data",
           "--gen-map", "exp-std/csv/eval200_fixed.csv", *ARGS,
           "--out", out, "--max-ref", os.environ.get("MAXREF", "4000"),
           "--baselines", "1", "--bsd", "1", "--bsd-ref-max", "1200", "--batch", "24"]
    print("\n=== %s ===" % name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                       timeout=14400)
    p = os.path.join(out, "style_report.json")
    if r.returncode != 0 or not os.path.isfile(p):
        print("  ✗ exit=%d %s" % (r.returncode, (r.stderr or "").strip().split("\n")[-1][:110]))
        results[name] = None
        continue
    results[name] = json.load(open(p, encoding="utf-8"))

print("\n\n========== moyi(48) vs 我们 ==========")
for enc, name in ENC:
    d = results.get(name)
    if not d:
        continue
    bl = d.get("baselines", {})
    def sv(x):
        return (x or {}).get("style_sim", {}).get("s_all", float("nan"))
    ceil, ttf, wrong = sv(bl.get("real_real_loo")), sv(bl.get("ttf_only")), \
        sv(bl.get("real_other_writer"))
    print("\n%s: 上限 %.4f | 错书家 %.4f | TTF-only %.4f" % (name, ceil, wrong, ttf))
    print("  %-26s %8s %8s %7s %7s %8s %7s" % ("集合", name, "归一化", "mAP", "XCT", "HWD-CN", "BSD宽"))
    for n, _ in SETS:
        g = (d.get("sets") or {}).get(n)
        if not g:
            continue
        s = sv(g)
        norm = (s - ttf) / (ceil - ttf) if (ceil == ceil and ttf == ttf and ceil != ttf) else float("nan")
        b = (g.get("bsd") or {}).get("width", float("nan"))
        print("  %-26s %8.4f %8s %7.4f %7.4f %8.3f %7s"
              % (n, s, ("%.3f" % norm) if norm == norm else "--",
                 g.get("retrieval", {}).get("map", float("nan")),
                 g.get("xctacc", {}).get("xctacc", float("nan")),
                 g.get("hwd_cn", {}).get("macro", float("nan")),
                 ("%.3f" % b) if b == b else "--"))
json.dump({k: (v or {}) for k, v in results.items()},
          open(os.path.join(OUT, "moyi_vs_ours.json"), "w", encoding="utf-8"),
          ensure_ascii=False, default=float)
print("\n[done] %s" % OUT)
