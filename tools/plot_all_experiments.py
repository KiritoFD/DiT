# -*- coding: utf-8 -*-
"""plot_all_experiments.py — 全部 77 run 的设计变量 × 性能综合分析 (分 cohort 可比)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                   "docs", "system", "imgs")
os.makedirs(OUT, exist_ok=True)

# (series, run尾, cohort, model, 数据, best_step, best_ssim, 备注)
RUNS = [
    ("s2_fromscratch_2factor", "kailishu", "S/2", 45000, 0.5748),
    ("s2_fromscratch_glyphmid", "kailishu", "S/2", 5000, 0.2954),
    ("s5_2factor_top30", "top30", "S/2", 45000, 0.5089),
    ("s5_2factor_top30", "top30", "S/2", 65000, 0.5186),
    ("s5_2factor_top30", "top30", "S/2", 95000, 0.5186),
    ("s5_2factor_struct_post50k", "top30", "S/2", 60000, 0.5079),
    ("s5_2factor_B_canny05_pixelsk", "top30", "B/2", 320000, 0.4784),
    ("s5_2factor_B_latentstruct", "top30", "B/2", 125000, 0.5059),
    ("s5_2factor_B_latentstruct_pixelsk_opt", "top30", "B/2", 110000, 0.3895),
    ("s5_2factor_B_pixelfp32", "top30", "B/2", 45000, 0.2175),
    ("s7_klf4_top30", "top30", "S/4", 70000, 0.5073),
    ("s8_klf4_clean_dino", "top30", "S/4", 100000, 0.5239),
    ("s10_b4_grey_clear", "top30", "B/4", 55000, 0.4656),
    ("s6_top6_diffonly", "top6", "S/2", 190000, 0.7310),
    ("s6_top6_diffonly", "top6", "S/2", 30000, 0.4883),
    ("s6_top6_struct_fp32", "top6", "S/2", 85000, 0.3301),
    ("s6_top6_struct_fp32_full", "top6", "S/2", 10000, 0.1410),
    ("s7_ramp_b8all", "top6", "S/2", 205000, 0.5768),
    ("s8_structv2_b8all", "top6", "S/2", 205000, 0.5387),
    ("s9_skelonly", "top6", "S/2", 210000, 0.4925),
    ("s11_top6_p4", "top6", "S/4", 140000, 0.6399),
    ("s11_top6_p4", "top6", "S/4", 120000, 0.6085),
    ("v3a", "strata", "S/2", 30000, 0.3663),
    ("v3a_xl", "strata", "XL/2", 4000, 0.0314),
    ("v3a_xl_highdim", "strata", "XL/2", 4000, 0.1947),
    ("v3a_xl_highdim", "strata", "XL/2", 4000, 0.1681),
    ("v3a_xl_skelhead", "strata", "XL/2", 10000, 0.3915),
    ("v3b_xl_glyphcond", "kailishu", "XL/2", 25000, 0.4829),
    ("compositional", "strata", "3Cond-S/2", 20000, 0.4576),
    ("s12_3top30_dino", "eval500_3top30", "S/2", 60000, 0.4886),
    ("s15_ws_flow", "eval500_3top30", "WS/2", 195000, 0.5390),
    ("s17_s_flow", "eval500_3top30", "S/2", 165000, 0.5325),
    ("s18_s_flow_small", "strict_top6", "S/2", 39000, 0.5454),
    ("s19_midclean_s_flow", "strict_top6", "S/2", 47500, 0.5218),
    ("s20_midcommon_s_flow_v2", "strict_midclean", "S/2新架构", 100000, 0.5292),
]
COHORT_COLORS = {
    "kailishu": "#8C8CFF", "top30": "#4C72B0", "top6": "#55A868",
    "strata": "#CB7B7B", "eval500_3top30": "#DD8452",
    "strict_top6": "#937860", "strict_midclean": "#C44E52",
}
COHORT_NAMES = {
    "kailishu": "kailishu 口径", "top30": "eval100_top30", "top6": "eval100_top6",
    "strata": "eval_strata 组合泛化", "eval500_3top30": "eval500_3top30 (latent)",
    "strict_top6": "eval_strict_top6 (flow)", "strict_midclean": "eval_strict_midclean (flow)",
}

# ── 图 5: 全景散点 ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12.5, 6.5))
seen_c = set()
for series, coh, model, step, ssim in RUNS:
    label = COHORT_NAMES[coh] if coh not in seen_c else None
    seen_c.add(coh)
    ax.scatter(step, ssim, s=70, c=COHORT_COLORS[coh], label=label,
               edgecolors="white", linewidths=0.8, zorder=3)
ann = [
    (190000, 0.7310, "s6 diffonly 0.731 (旧口径最高)", (60000, 0.80)),
    (140000, 0.6399, "s11 patch4 0.640", (140000, 0.72)),
    (100000, 0.5292, "s20 0.529 新架构 (可信口径)", (6000, 0.575)),
    (195000, 0.5390, "s15 WS/2", (230000, 0.50)),
    (39000, 0.5454, "s18", (8000, 0.60)),
    (4000, 0.1947, "XL 系多在早期发散", (12000, 0.10)),
    (45000, 0.2175, "B/2 pixelfp32 发散", (70000, 0.16)),
]
for x, y, txt, (tx, ty) in ann:
    ax.annotate(txt, xy=(x, y), xytext=(tx, ty), fontsize=8, color="#333333",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.6))
ax.set_xscale("log")
ax.set_xlabel("best checkpoint 所在步数 (log)")
ax.set_ylabel("best SSIM (各自口径)")
ax.set_title("图5 · 全部 run 全景: 颜色 = eval 口径 (同色才可比), x = 收敛位置\n"
             "旧口径数字普遍偏高 (eval 样本少/无 held-out); 越靠下的口径越严格", fontsize=10)
ax.legend(fontsize=8, loc="lower right", ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_landscape.png"), dpi=130)
plt.close(fig)

# ── 图 6: 每个 cohort 的历史最好成绩 ─────────────────────────────────────────
cohorts = ["kailishu", "top30", "top6", "strata", "eval500_3top30", "strict_top6", "strict_midclean"]
best_in_c, best_run = {}, {}
for series, coh, model, step, ssim in RUNS:
    if ssim > best_in_c.get(coh, 0):
        best_in_c[coh] = ssim
        best_run[coh] = f"{series} ({model})"
order = [c for c in cohorts if c in best_in_c]
vals = [best_in_c[c] for c in order]
labs = [f"{COHORT_NAMES[c]}\n{best_run[c]}" for c in order]
fig, ax = plt.subplots(figsize=(12.5, 5))
bars = ax.bar(labs, vals, color=[COHORT_COLORS[c] for c in order])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.set_ylim(0, 0.85)
ax.set_ylabel("该口径下历史最好 SSIM")
ax.tick_params(axis="x", labelsize=8)
ax.set_title("图6 · 各 eval 口径下的历史最好成绩 — 这是 7 张不同难度的考卷, 不是一条可比曲线\n"
             "早期口径宽松(样本少/无 held-out)数字虚高; 后期口径逐步收紧", fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig6_cohort_best.png"), dpi=130)
plt.close(fig)

# ── 图 7: pixel 时代的设计变量效应 (cohort 内可比) ───────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))
panels = [
    ("结构损失 (top6 口径, S/2)",
     [("纯扩散\ns6_diffonly", 0.7310), ("+结构损失\ns6_struct", 0.3301),
      ("struct_full\n(发散)", 0.1410)],
     "当时结论: 结构损失在该设置下\n严重伤害生成质量 →\n转向 diff-only + ControlNet 注入"),
    ("模型宽度 (top30 口径)",
     [("S/2 33M\ns5_2factor", 0.5186), ("B/2 2x宽\ns5_B 系列", 0.5059)],
     "加宽无增益; B 系更容易发散\n(多个 run best 在早期)"),
    ("数据清洗 (top30 系)",
     [("原始 top30\ns5 (S/2)", 0.5186), ("KLF4 清洗\ns7 (S/4)", 0.5073),
      ("清洗+DINO\ns8 (S/4)", 0.5239)],
     "清洗本身持平; DINO 字符嵌入\n微增 +0.017 (与 patch4 混杂)"),
    ("patch 大小 (top6 口径)",
     [("S/2 patch2\ns7_ramp", 0.5768), ("S/4 patch4\ns11", 0.6399)],
     "patch4 (更细 token 网格)\npixel 时代 +0.063;\nlatent 时代未再扫此轴"),
]
for ax, (title, labs, note) in zip(axes, panels):
    bars = ax.bar([l for l, _ in labs], [v for _, v in labs],
                  color=["#4C72B0", "#C44E52", "#8C8CFF", "#55A868"][:len(labs)])
    ax.set_ylim(min(v for _, v in labs) - 0.06, max(v for _, v in labs) + 0.05)
    for b, (l, v) in zip(bars, labs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8)
    ax.text(0.0, -0.30, note, transform=ax.transAxes, fontsize=8, va="top")
fig.suptitle("图7 · pixel 时代 (s2–s11) 的设计变量效应 — 各 panel 内同口径可比",
             y=1.02, fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig7_pixel_era_axes.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

print("saved fig5/6/7 ->", OUT)
