#!/usr/bin/env python
"""plot_v10_arms.py — 三臂 SSIM 曲线图 (v10a / v10b / v10a-dino)"""
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
colors = {"v10a": "#1f77b4", "v10b": "#ff7f0e", "v10a-dino": "#2ca02c"}
marks = {"v10a": "o", "v10b": "s", "v10a-dino": "^"}

fig, ax = plt.subplots(figsize=(10, 6))
for k, c in curves.items():
    if not c:
        continue
    xs = sorted(c)
    ys = [c[x] for x in xs]
    ax.plot(xs, ys, color=colors[k], marker=marks[k], markersize=3,
            linewidth=1.5, label=f"{k} (best {max(c.values()):.4f}@{max(c, key=c.get)})")
ax.set_xlabel("step")
ax.set_ylabel("SSIM (g=GT skel, n=100, cfg 0.7)")
ax.set_title("v10 series: SSIM curves (char factor ablation)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out = "/root/Workspace/xy/DiT/5script/results/skel_follow_gpu/v10_arms_ssim.png"
fig.savefig(out, dpi=130)
print("saved:", out)