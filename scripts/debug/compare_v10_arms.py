#!/usr/bin/env python
"""compare_v10_arms.py — v10a / v10b / v10a-dino 三臂 eval_auto 曲线同点对比"""
import json, glob

ARMS = {
    "v10a":      "/root/Workspace/xy/DiT/5script/results/v10a_skel_cond_pretrain/*/checkpoints/eval_auto_*.json",
    "v10b":      "/root/Workspace/xy/DiT/5script/results/v10b_skel_only_pretrain/*/checkpoints/eval_auto_*.json",
    "v10a-dino": "/root/Workspace/xy/DiT/5script/results/v10adino_skel_cond_pretrain/*/checkpoints/eval_auto_*.json",
}

def load_curve(pat):
    out = {}
    for f in glob.glob(pat):
        try:
            d = json.load(open(f))
            st = int(d.get("step") or f.split("eval_auto_")[-1].split(".")[0])
            s = d.get("ssim") or d.get("ssim_mean")
            if s:
                out[st] = float(s)
        except Exception:
            pass
    return out

curves = {k: load_curve(v) for k, v in ARMS.items()}

# v10a-dino 的采点 (稀疏到每 12500)
dino_steps = sorted(k for k in curves["v10a-dino"] if k % 12500 == 0)
if 80000 not in dino_steps:
    dino_steps = sorted(k for k in curves["v10a-dino"] if k % 12500 == 0 or k == 80000)
    dino_steps = sorted(set(dino_steps + [k for k in curves["v10a-dino"] if k in (67500, 70000, 72500, 75000, 77500, 80000)]))

print(f"{'step':>8} | {'v10a':>8} | {'v10b':>8} | {'v10a-dino':>10} | delta(dino-v10a)")
for st in sorted(dino_steps):
    row = [curves["v10a"].get(st), curves["v10b"].get(st), curves["v10a-dino"].get(st)]
    def f(x):
        return f"{x:.4f}" if x is not None else "  -  "
    d = row[2] - row[0] if row[0] and row[2] else None
    print(f"{st:>8} | {f(row[0]):>8} | {f(row[1]):>8} | {f(row[2]):>10} | "
          f"{('+' if d and d>0 else '') + f'{d:.4f}' if d is not None else '  -  '}  ({curves['v10a-dino'].get(st, '')})")

# 三臂峰值汇总
print("\n=== 峰值 ===")
for k, c in curves.items():
    if c:
        best = max(c.items(), key=lambda kv: kv[1])
        print(f"{k:<10} best {best[1]:.4f} @ {best[0]}  (点数 {len(c)})")