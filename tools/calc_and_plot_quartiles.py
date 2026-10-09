import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 设置绘图字体与风格
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 载入各模型数据
e200_file = "assets/eval200fix_models_per_sample.csv"
moyi_file = "assets/moyi_12ch_e200_per_sample.csv"

df_all = pd.read_csv(e200_file)
df_moyi = pd.read_csv(moyi_file)
df_moyi["model_name"] = "moyi_12ch"
df_moyi["step"] = 50000

# 提取 v68 并排序
df_v68 = df_all[df_all["model_name"] == "v68"].copy()
df_v68["ssim"] = df_v68["ssim"].astype(float)
df_v68["idx"] = df_v68["idx"].astype(int)
df_v68_ranked = df_v68.sort_values(by="ssim", ascending=False).reset_index(drop=True)

# 确定四分位区间
# Top 25%: 前 47 (0..46)
# Mid 50%: 中间 93 (47..139)
# Worst 25%: 尾部 47 (140..186)
idx_top25 = set(df_v68_ranked.iloc[0:47]["idx"].tolist())
idx_mid50 = set(df_v68_ranked.iloc[47:140]["idx"].tolist())
idx_worst25 = set(df_v68_ranked.iloc[140:187]["idx"].tolist())

# 模型清单
models_to_analyze = [
    ("v68 (C2OT 旗舰)", "v68", "#10b981"),
    ("v66 (黄金路由)", "v66", "#38bdf8"),
    ("v54 (字表基线)", "v54", "#a78bfa"),
    ("moyi_12ch (48先锋)", "moyi_12ch", "#f59e0b"),
    ("v70 (开集集大成)", "v70", "#ec4899"),
]

stats_rows = []
model_tier_data = {}

for label, mkey, color in models_to_analyze:
    if mkey == "moyi_12ch":
        m_df = df_moyi.copy()
    else:
        m_df = df_all[df_all["model_name"] == mkey].copy()
    
    m_df["ssim"] = m_df["ssim"].astype(float)
    m_df["mse"] = m_df["mse"].astype(float)
    m_df["idx"] = m_df["idx"].astype(int)
    
    sub_top = m_df[m_df["idx"].isin(idx_top25)]
    sub_mid = m_df[m_df["idx"].isin(idx_mid50)]
    sub_worst = m_df[m_df["idx"].isin(idx_worst25)]
    
    model_tier_data[label] = {
        "all": m_df,
        "top25": sub_top,
        "mid50": sub_mid,
        "worst25": sub_worst,
        "color": color
    }
    
    stats_rows.append({
        "Model": label,
        "Overall_SSIM": m_df["ssim"].mean(),
        "Top25_SSIM": sub_top["ssim"].mean(),
        "Mid50_SSIM": sub_mid["ssim"].mean(),
        "Worst25_SSIM": sub_worst["ssim"].mean(),
        "SSIM_Drop": sub_top["ssim"].mean() - sub_worst["ssim"].mean(),
        "Overall_MSE": m_df["mse"].mean(),
        "Top25_MSE": sub_top["mse"].mean(),
        "Mid50_MSE": sub_mid["mse"].mean(),
        "Worst25_MSE": sub_worst["mse"].mean(),
    })

df_stats = pd.DataFrame(stats_rows)
out_csv = "assets/eval200_quartile_comparison_table.csv"
df_stats.to_csv(out_csv, index=False, encoding="utf-8-sig")
print("=== 四分位数全景统计表 ===")
print(df_stats[["Model", "Top25_SSIM", "Mid50_SSIM", "Worst25_SSIM", "SSIM_Drop", "Overall_SSIM"]])

# ========================= 绘制现代多维可视化图表 =========================
fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=300)
fig.patch.set_facecolor('#0f172a')  # 深色 Slate 背景

for ax in axes:
    ax.set_facecolor('#1e293b')
    ax.tick_params(colors='#94a3b8', labelsize=11)
    ax.spines['bottom'].set_color('#334155')
    ax.spines['top'].set_color('#334155')
    ax.spines['left'].set_color('#334155')
    ax.spines['right'].set_color('#334155')
    ax.grid(True, linestyle='--', alpha=0.25, color='#94a3b8')

# 子图 1: 分组柱状图 (SSIM Across Quartiles)
ax1 = axes[0]
tier_names = ['Top 25% (高分层)', 'Mid 50% (中位层)', 'Worst 25% (攻坚层)', 'Overall (全集)']
x = np.arange(len(tier_names))
width = 0.16

for i, (label, mkey, color) in enumerate(models_to_analyze):
    row = df_stats[df_stats["Model"] == label].iloc[0]
    vals = [row["Top25_SSIM"], row["Mid50_SSIM"], row["Worst25_SSIM"], row["Overall_SSIM"]]
    rects = ax1.bar(x + (i - 2) * width, vals, width, label=label, color=color, alpha=0.9, edgecolor='#0f172a', linewidth=0.8)
    # 顶部标注数值
    for rect in rects:
        h = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2., h + 0.008, f'{h:.3f}', ha='center', va='bottom', fontsize=8, color='#f1f5f9', rotation=45)

ax1.set_title('各模型在 eval200 四分位区间上的 SSIM 表现对比', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(tier_names, fontsize=11, fontweight='bold', color='#f1f5f9')
ax1.set_ylabel('Structural Similarity (SSIM ↑)', fontsize=12, color='#cbd5e1')
ax1.set_ylim(0.35, 0.88)
ax1.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10, labelcolor='#f1f5f9', loc='upper right')

# 子图 2: 均方误差 (MSE Across Quartiles - 越低越好)
ax2 = axes[1]
for i, (label, mkey, color) in enumerate(models_to_analyze):
    row = df_stats[df_stats["Model"] == label].iloc[0]
    vals = [row["Top25_MSE"], row["Mid50_MSE"], row["Worst25_MSE"], row["Overall_MSE"]]
    rects = ax2.bar(x + (i - 2) * width, vals, width, label=label, color=color, alpha=0.9, edgecolor='#0f172a', linewidth=0.8)
    for rect in rects:
        h = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2., h + 0.02, f'{h:.2f}', ha='center', va='bottom', fontsize=8, color='#f1f5f9', rotation=45)

ax2.set_title('各模型在 eval200 四分位区间上的均方误差 MSE 对比 (越低越好)', fontsize=14, fontweight='bold', color='#f8fafc', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(tier_names, fontsize=11, fontweight='bold', color='#f1f5f9')
ax2.set_ylabel('Mean Squared Error (4x MSE ↓)', fontsize=12, color='#cbd5e1')
ax2.set_ylim(0.0, 1.65)
ax2.legend(facecolor='#0f172a', edgecolor='#334155', fontsize=10, labelcolor='#f1f5f9', loc='upper left')

plt.tight_layout()
chart_path = "docs/04_experiments/imgs/eval200_quartiles_distribution_chart.png"
plt.savefig(chart_path, dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()

print(f"🎉 四分位区间统计对比可视化图表已生成: {chart_path}")
