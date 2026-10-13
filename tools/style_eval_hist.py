# -*- coding: utf-8 -*-
"""历史 ckpt 全量风格评测：8 个历史 run + v84A 的 6 个 step × DINO-S / CLIP-S。

同一评测集 ✓（全部 eval200fix: eval200_fixed.csv:187 ✓）⇒ 可直接横比 ✓。
输出: 每个编码器一份报告 + 合并汇总表（含上限/TTF/错书家三条参照 ✓）。
"""
import glob
import json
import os
import re
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "assets", "style_eval_local", "hist_compare")
os.makedirs(OUT, exist_ok=True)

HIST = ["v72_callivae_kl1e6_s12500_frozen_sp_c2ot", "v74_callivae_mode_sp_c2ot",
        "v76_callivae_topband_sp_c2ot", "v77_callivae_topband_norepa_sp_c2ot",
        "v78_callivae_topband_scale1_sp_c2ot", "v79_callivae_physics_sp_c2ot",
        "v80_callivae_physics_noOT", "v83A_stdsoft_zerofix"]
V84_STEPS = ["step0020000", "step0050000", "step0080000", "step0100000",
             "step0115000", "step0120000"]


def find_hist_dirs():
    out = []
    for run in HIST:
        cands = glob.glob(os.path.join(ROOT, "assets", "results", run, "**", "step*",
                                       "eval200fix"), recursive=True)
        if not cands:
            cands = glob.glob(os.path.join(ROOT, "assets", "results", run, "**", "eval200fix"),
                              recursive=True)
        if not cands:
            print("  ⚠ %s: 找不到 eval200fix ✗" % run)
            continue
        def num(p):
            m = re.search(r"step(\d+)", p)
            return int(m.group(1)) if m else -1
        d = max(cands, key=num)
        step = num(d)
        out.append(("v83A@%dk" % (step // 1000) if run.startswith("v83A") else
                    "%s@%dk" % (run.split("_")[0], step // 1000), d))
    return out


GENS = find_hist_dirs()
for s in V84_STEPS:
    d = os.path.join(ROOT, "assets", "style_eval_src", s)
    if os.path.isdir(d):
        GENS.append(("v84A_std@" + ("%dk" % (int(s[4:]) // 1000)), d))
print("=== 参与评测的集合 %d 个 ===" % len(GENS))
for n, d in GENS:
    print("  %-22s %s" % (n, os.path.relpath(d, ROOT)))

ENC = [("dinov2_vitl14", "DINO-S"), ("clip_vitl14", "CLIP-S")]
ARGS = []
for n, d in GENS:
    ARGS += ["--gen", "%s=%s" % (n, os.path.relpath(d, ROOT))]
MAXREF = os.environ.get("MAXREF", "4000")
results = {}
for enc, name in ENC:
    out = os.path.join(OUT, name)
    cmd = [sys.executable, os.path.join(ROOT, "tools", "style_eval.py"),
           "--enc", enc, "--pool", "cls",
           "--csv", "exp-std/csv/train.csv", "--root", ".", "--root", "data",
           "--gen-map", "exp-std/csv/eval200_fixed.csv", *ARGS,
           "--out", out, "--max-ref", MAXREF, "--baselines", "1", "--bsd", "1",
           "--bsd-ref-max", "1200", "--batch", "24"]
    print("\n=== 跑 %s ===" % name, flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                       timeout=14400)
    p = os.path.join(out, "style_report.json")
    if r.returncode != 0 or not os.path.isfile(p):
        print("  ✗ %s exit=%d %s" % (name, r.returncode,
                                     (r.stderr or "").strip().split("\n")[-1][:110]))
        results[name] = None
        continue
    results[name] = json.load(open(p, encoding="utf-8"))

print("\n\n========== 汇总（历史 ckpt 全量）==========")
for enc, name in ENC:
    d = results.get(name)
    if not d:
        continue
    bl = d.get("baselines", {})
    def sv(x):
        return (x or {}).get("style_sim", {}).get("s_all", float("nan"))
    ceil, wrong, ttf = sv(bl.get("real_real_loo")), sv(bl.get("real_other_writer")), \
        sv(bl.get("ttf_only"))
    print("\n=== %s ===  上限(留一)=%.4f | 错书家=%.4f | TTF-only=%.4f" % (name, ceil, wrong, ttf))
    print("  %-22s %8s %8s %7s %7s %8s %7s" % ("集合", name, "归一化", "mAP", "XCT", "HWD-CN", "BSD宽"))
    rows = []
    for n, _ in GENS:
        g = (d.get("sets") or {}).get(n) or {}
        s = sv(g)
        norm = ((s - ttf) / (ceil - ttf)) if (ceil == ceil and ttf == ttf and ceil != ttf) else float("nan")
        b = (g.get("bsd") or {}).get("width", float("nan"))
        rows.append((n, s, norm, g.get("retrieval", {}).get("map", float("nan")),
                     g.get("xctacc", {}).get("xctacc", float("nan")),
                     g.get("hwd_cn", {}).get("macro", float("nan")), b))
    for n, s, norm, mp, xc, hw, bw in sorted(rows, key=lambda x: -(x[1] if x[1] == x[1] else -9)):
        print("  %-22s %8.4f %8s %7.4f %7.4f %8.3f %7s"
              % (n, s, ("%.3f" % norm) if norm == norm else "--", mp, xc, hw,
                 ("%.3f" % bw) if bw == bw else "--"))
    json.dump({"name": name, "encoder": enc, "ceiling": ceil, "wrong_writer": wrong,
               "ttf_only": ttf, "rows": rows},
              open(os.path.join(OUT, "%s_summary.json" % name), "w", encoding="utf-8"),
              ensure_ascii=False, default=float)
print("\n[done] %s" % OUT)
