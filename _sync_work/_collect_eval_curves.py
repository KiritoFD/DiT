# -*- coding: utf-8 -*-
"""收集各实验的 eval 曲线 (step -> ssim/lpips/mse), 用于 dashboard 对比。

输出: _sync_work/dashboard_data.json
格式: { "s21": {"rows": [{step, ssim, lpips, mse, skel_iou}, ...], "name": "..."}, ... }
"""
import os, sys, glob, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
os.chdir("/root/Workspace/xy/DiT")

# 要对比的实验 (基模预训练)
EXPS = [
    ("s21", "assets/results/s21_fame_flow_v2/*/checkpoints/eval_auto_*.json",
     "s21 真迹DINO(ln_only)"),
    ("s25", "assets/results/s25_ids_pretrain/*/checkpoints/eval_auto_*.json",
     "s25 IDS部件码本"),
    ("s28", "assets/results/s28_std_dino_pretrain/*/checkpoints/eval_auto_*.json",
     "s28 标准字形DINO(PCA+OT)"),
]

out = {}
for key, pattern, name in EXPS:
    files = sorted(glob.glob(pattern))
    rows = []
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        step = d.get("step", 0)
        ssim = d.get("ssim", 0.0)
        lpips = d.get("lpips", None)
        mse = d.get("mse", 0.0)
        skel = d.get("skel_iou", 0.0)
        rows.append({"step": step, "ssim": ssim, "lpips": lpips,
                     "mse": mse, "skel_iou": skel})
    # 去重: 同 step 保留最新 (多个 run 时)
    seen = {}
    for r in rows:
        seen[r["step"]] = r
    rows = [seen[s] for s in sorted(seen)]
    out[key] = {"name": name, "rows": rows}
    print(f"{key}: {len(rows)} eval points "
          f"(step {rows[0]['step'] if rows else '-'}..{rows[-1]['step'] if rows else '-'})")
    if rows:
        best = max(rows, key=lambda r: r["ssim"])
        print(f"   best ssim={best['ssim']:.4f} @step {best['step']}")
        print(f"   latest ssim={rows[-1]['ssim']:.4f} @step {rows[-1]['step']}")

with open("_sync_work/dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(f"\nsaved -> _sync_work/dashboard_data.json")
