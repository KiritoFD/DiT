# -*- coding: utf-8 -*-
"""负样本数量扫描（用户建议 ✓）: n_font_neg ∈ {1,2,4,8} ⇒ 看哪个最好 ✓。

为什么要扫 ✓
    "负样本越多越好"是直觉，但**未经验证** ✗；而且负样本越多 ⇒ 每步编码的前向越多 ⇒
    越慢 ✓。所以要在**判别质量 vs 成本**之间找点 ✓。
做法 ✓
    · 每个设定用**短跑**（--steps 1000 ✓）—— 对比学习收敛快，短跑足够比较 ✓
    · 统一其它超参（img/batch/tau/lr/seed ✓）⇒ 单变量 ✓
    · 每个设定跑完自动出**门槛指标**（G1 2AFC / G2 mAP / 字体维度 ✓）✓
    · 顺序执行（避免同卡争抢 ⇒ 也避免 DDP 端口那类坑 ✓）
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "style_eval_local", "disc_sweep")
os.makedirs(OUT, exist_ok=True)

SETTINGS = [1, 2, 4, 8]
STEPS = int(os.environ.get("SWEEP_STEPS", "1000"))
IMG = os.environ.get("SWEEP_IMG", "128")
BATCH = os.environ.get("SWEEP_BATCH", "48")

rows = []
for n in SETTINGS:
    tag = "neg%d" % n
    ck = os.path.join(ROOT, "assets", "style_disc_%s.pt" % tag)
    log = os.path.join(OUT, "%s.log" % tag)
    print("\n===== 设定 n_font_neg=%d（steps=%d img=%s batch=%s）=====" % (n, STEPS, IMG, BATCH),
          flush=True)
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, os.path.join(ROOT, "tools",
                                                         "train_style_discriminator.py"),
                            "--steps", str(STEPS), "--batch", BATCH, "--img", IMG,
                            "--font-per-char", "4", "--n-font-neg", str(n),
                            "--out", ck, "--n-pairs", "1200", "--bootstrap", "1200"],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, text=True, timeout=14400)
    el = time.time() - t0
    txt = open(log, encoding="utf-8").read()
    rec = {"n_font_neg": n, "steps": STEPS, "img": IMG, "batch": BATCH,
           "elapsed_s": round(el, 1), "exit": r.returncode,
           "sec_per_step": round(el / max(1, STEPS), 3)}
    try:
        ckd = __import__("torch").load(ck, map_location="cpu", weights_only=False)
        rec["gates_real"] = ckd.get("gates_real")
        rec["gates_font"] = ckd.get("gates_font")
    except Exception as e:                                              # noqa: BLE001
        rec["load_err"] = str(e)[:120]
    rows.append(rec)
    g = (rec.get("gates_real") or {})
    g1 = (g.get("G1") or {})
    g2 = (g.get("G2") or {})
    print("  ⇒ 用时 %.0f 秒（%.2f 秒/步 ✓）| G1=%.4f (n=%s) | G2 mAP=%.4f | 字体 G1=%.4f"
          % (el, rec["sec_per_step"], g1.get("acc", float("nan")), g1.get("n"),
             g2.get("map", float("nan")),
             ((rec.get("gates_font") or {}).get("G1") or {}).get("acc", float("nan"))),
          flush=True)

json.dump(rows, open(os.path.join(OUT, "sweep.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=2, default=float)
print("\n===== 扫描汇总（负样本数 vs 判别质量 vs 成本 ✓）=====")
print("  %-10s %10s %10s %10s %10s" % ("n_font_neg", "秒/步", "G1", "G2 mAP", "字体G1"))
for r in rows:
    g = (r.get("gates_real") or {})
    print("  %-10d %10.3f %10.4f %10.4f %10.4f"
          % (r["n_font_neg"], r["sec_per_step"], (g.get("G1") or {}).get("acc", float("nan")),
             (g.get("G2") or {}).get("map", float("nan")),
             ((r.get("gates_font") or {}).get("G1") or {}).get("acc", float("nan"))))
print("[done] %s" % os.path.join(OUT, "sweep.json"))
