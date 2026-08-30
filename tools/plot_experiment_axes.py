# -*- coding: utf-8 -*-
"""plot_experiment_axes.py — 实验设计变量 × 性能可视化 (本地 matplotlib)."""
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "docs", "system", "imgs")
os.makedirs(OUT, exist_ok=True)

# ── 图 1: 统一口径下 5 个基模 (eval_strict_midclean, n=501, cfg1.7) ─────────
names = ["s18\nS/2旧架构\ntop6 1.1万张", "s17\nS/2旧架构\n3top30",
         "s19\nS/2旧架构\nmidclean+增广11.9万", "s15\nWS/2大模型\n3top30",
         "s20\nS/2新架构\nmid-common 2.4万真实"]
ssim = [0.4336, 0.5204, 0.5160, 0.5245, 0.5294]
mse = [0.8541, 0.8166, 0.8554, 0.7888, 0.8155]
colors = ["#999999", "#4C72B0", "#8C8CFF", "#55A868", "#C44E52"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bars = axes[0].bar(names, ssim, color=colors)
axes[0].axhline(0.563, ls="--", c="crimson", lw=1)
axes[0].text(4.4, 0.566, "同写本真迹互比 0.563\n(逐实例 SSIM 天花板)", fontsize=8,
             color="crimson", ha="right")
axes[0].set_ylim(0.40, 0.60)
axes[0].set_title("SSIM (统一口径 eval_strict_midclean, n=501)")
axes[0].tick_params(axis="x", labelsize=8)
for b, v in zip(bars, ssim):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                 ha="center", fontsize=9)
bars = axes[1].bar(names, mse, color=colors)
axes[1].set_ylim(0.6, 0.95)
axes[1].set_title("MSE (×4 口径, 越低越好)")
axes[1].tick_params(axis="x", labelsize=8)
for b, v in zip(bars, mse):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.4f}",
                 ha="center", fontsize=9)
fig.suptitle("图1 · 数据 / 架构 / 规模三轴对预训练质量的影响 (同 eval 同噪声同 cfg)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_unified_models.png"), dpi=130)
plt.close(fig)

# ── 图 2: s20 训练曲线 (SSIM + LPIPS 双轴) ──────────────────────────────────
curve = """1000 1.07183 0.4363 0.5123
2000 1.04899 0.4563 0.4733
3000 1.04767 0.4609 0.4607
4000 1.02677 0.4694 0.4498
5000 1.01970 0.4720 0.4454
7500 0.99169 0.4816 0.4352
10000 0.98069 0.4852 0.4297
12500 0.97428 0.4870 0.4259
15000 0.95714 0.4903 0.4228
17500 0.94152 0.4952 0.4181
20000 0.94598 0.4932 0.4169
22500 0.93003 0.4982 0.4128
25000 0.91661 0.5008 0.4104
27500 0.90644 0.5044 0.4073
30000 0.89744 0.5071 0.4046
32500 0.89626 0.5072 0.4030
35000 0.88753 0.5092 0.4005
37500 0.88434 0.5101 0.3995
40000 0.87859 0.5121 0.3973
42500 0.87957 0.5113 0.3970
45000 0.86517 0.5155 0.3938
47500 0.85852 0.5180 0.3922
50000 0.85917 0.5177 0.3919
52500 0.85513 0.5183 0.3912
55000 0.85218 0.5185 0.3905
57500 0.84748 0.5203 0.3893
60000 0.84247 0.5216 0.3884
62500 0.84027 0.5227 0.3876
65000 0.83795 0.5231 0.3875
67500 0.83451 0.5244 0.3863
70000 0.83106 0.5259 0.3851
72500 0.83048 0.5258 0.3854
75000 0.83040 0.5258 0.3850
77500 0.82800 0.5264 0.3847
80000 0.82636 0.5263 0.3845
82500 0.82524 0.5270 0.3842
85000 0.82506 0.5264 0.3844
87500 0.82215 0.5274 0.3837
90000 0.81989 0.5283 0.3832
92500 0.81801 0.5287 0.3829
95000 0.81781 0.5290 0.3827
97500 0.81739 0.5292 0.3825
100000 0.81731 0.5292 0.3823
102500 0.81546 0.5294 0.3822"""
steps, mses, ssims, lpipss = [], [], [], []
for line in curve.strip().splitlines():
    a, b, c, d = line.split()
    steps.append(int(a)); mses.append(float(b)); ssims.append(float(c)); lpipss.append(float(d))

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(steps, ssims, "-o", ms=3, c="#C44E52", label="SSIM (左轴)")
ax1.set_xlabel("训练步数"); ax1.set_ylabel("SSIM", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax2 = ax1.twinx()
ax2.plot(steps, lpipss, "-s", ms=3, c="#4C72B0", label="LPIPS (右轴)")
ax2.set_ylabel("LPIPS", color="#4C72B0")
ax2.tick_params(axis="y", labelcolor="#4C72B0")
ax1.axvline(90000, ls=":", c="gray")
ax1.annotate("进入平台期\n(最后 1 万步 +0.001)", xy=(90000, 0.5283), xytext=(60000, 0.45),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
ax1.annotate("early-stop @102.5k", xy=(102500, 0.5294), xytext=(82000, 0.44),
             fontsize=9, arrowprops=dict(arrowstyle="->", color="gray"))
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
plt.title("图2 · s20 (新架构 S/2) 训练收敛曲线 — eval_strict_midclean n=501, cfg1.7")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_s20_curve.png"), dpi=130)
plt.close(fig)

# ── 图 3: 条件信息量 = 性能 (实例级条件是唯一杠杆) ───────────────────────────
labels = ["VAE 重建上限", "同写本真迹互比\n(逐实例天花板)", "s20 基模\n(书家+字 ID)",
          "基模 + ControlNet\n(GT 骨架 latent)"]
vals = [0.962, 0.563, 0.5294, 0.7201]
cols = ["#BBBBBB", "crimson", "#999999", "#C44E52"]
fig, ax = plt.subplots(figsize=(9.5, 5))
bars = ax.bar(labels, vals, color=cols)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_ylabel("SSIM vs GT")
ax.set_title("图3 · 条件携带的信息量决定 SSIM — 逐实例指标只有实例级条件能推上去\n"
             "(ctrl 数值来自 s19 基 ControlNet step10000, n=100, cfg1.7, 供量级参考)", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_condition_value.png"), dpi=130)
plt.close(fig)

# ── 图 4: 三条设计轴的增量效应 (ΔSSIM, 统一口径) ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
panels = [
    ("数据轴 (S/2 旧架构)", ["top6\n1.1万张\n(s18)", "3top30\n(s17)", "mid-clean\n+增广 11.9万\n(s19)"],
     [0.4336, 0.5204, 0.5160], "换到干净大数据 +0.087;\n增广变体无额外收益 (-0.004)"),
    ("架构轴 (S/2, mid-common)", ["旧架构\nLN+gelu\n(s19 同数据)", "新架构\nrms+swiglu+qknorm+RoPE\n(s20)"],
     [0.5160, 0.5294], "新架构 +0.013,\n且以 1/2.4 训练步数达成"),
    ("规模轴 (旧架构)", ["S/2 33M\n(s17)", "WS/2 ~70M\n(s15)"],
     [0.5204, 0.5245], "2 倍参数只换 +0.004 —\n容量不是当前瓶颈"),
]
for ax, (title, labs, vals, note) in zip(axes, panels):
    bars = ax.bar(labs, vals, color=["#4C72B0", "#C44E52", "#8C8CFF"][:len(labs)])
    ax.set_ylim(min(vals) - 0.03, max(vals) + 0.02)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=8, color="#333333",
            va="bottom")
fig.suptitle("图4 · 单变量增量效应 (ΔSSIM, 全部在统一口径 eval_strict_midclean 下测量)", y=1.0)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_design_axes.png"), dpi=130)
plt.close(fig)

print("saved 4 figs ->", OUT)
