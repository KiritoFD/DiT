# 旧实验归档与 ckpt 清理快照（2026-08-29）

> 执行：`tools/cleanup_old_experiments.py`（dry-run 审查后 --apply）。
> 政策：每个 run 只保留 **best-2 ckpt**（按各自 eval 主指标选取）+ 被活动训练
> 引用/在跑系列的 ckpt。指标随时可由 eval_samples PNG 重算，不依赖被删 ckpt。

## 清理结果

- 扫描范围：`5script/results/` + `results/`，共 77 个 run
- 保护（未动）：11 个 run —— `ctrl_skel`（含在跑）、`s20_ctrl_skel_flow_v2`（用户在跑）、
  2 小时内有写入的目录；以及 s20 被引用的 `0102500.pt`
- 清理：66 个 run，**释放 ~882 GB**（`5script/results` 885G→184G，`results` 221G→40G）
- 产物：
  - 汇总 CSV：`5script/old_experiments_summary_20260829.csv`（每 run：状态/说明/
    n_ckpts/保留/删除/释放/best_step/best_metrics）
  - 每 run 快照 JSON：`5script/results/_cleanup_20260829/<series>__<run>.json`
    （配置摘要 + best2 指标 + eval 历史尾部 + 保留清单）

## 主要系列一览（best 指标为各自口径，跨代不可直接比）

| 系列 | 说明 | 释放 | best step | best 指标 |
|---|---|---|---|---|
| **s20_midcommon_s_flow_v2** | **新架构 rms/swiglu/qk_norm/RoPE + mid-common，当前基模** | 24.9G | 102500 | **ssim 0.5294 / mse 0.8155**（eval_strict_midclean, cfg1.7） |
| s19_midclean_s_flow | mid-clean 增广 + flow（旧架构末代） | 14.1G | 50000 | ssim 0.5222（旧口径） |
| s18_s_flow_small | S/2 flow top6 小数据 | 12.9G | 43000 | ssim 0.5476（旧口径，eval 全 in-domain） |
| s17_s_flow | S/2 flow 3top30 | 20.6G | 165000 | ssim 0.5325（eval500_3top30） |
| s15_ws_flow | WS/2 宽体 flow | 63.1G | 195000 | ssim 0.5390（eval500_3top30） |
| s12/s13/s14 | 3top30 DINO ddpm 系 | ~37G | — | 短跑/放弃 |
| s5/s6/s7/s8/s9/s10/s11 | pixel 时代二因子/结构损失系列 | ~350G | — | 旧 pixel 口径（如 s6 diffonly 0.731、s11 p4 0.640，**与 latent 时代不可比**） |
| v3a/v3b/v3c XL 系 | 早期 XL 条件融合/字形条件探索 | ~155G | — | 多为短跑（best step ≤25k） |
| exp_xl_head r8/32/64 | XL skel-head 秩消融（pixel 重建口径） | ~164G | 33000 | ssim ~0.90（**重建口径**，非生成） |
| exp_s_scratch / dit_s_pretrain | pixel 时代 S 从零/预训练 | ~6G | — | 同上 |

> ⚠ 指标口径演变：pixel 时代（s2–s11、v3、exp_*）与 latent+flow 时代
> （s12 之后）的 eval 集和指标定义不同；s18→s19→s20 的 eval 口径也各不相同。
> 真正同口径的对比只有 `5script/eval_unified_20260829.csv`（s15/s17/s18/s19
> 在 eval_strict_midclean 上的统一评测）与 s20 的 eval_auto 曲线。

## 现存关键资产

| 资产 | 位置 |
|---|---|
| s20 基模 best ckpt | `5script/results/s20_midcommon_s_flow_v2/.../checkpoints/0102500.pt`（被 ctrl 训练引用，永久保护） |
| s20 eval 曲线 | 同目录 `eval_auto_*.json` + `/tmp/s20_eval_daemon.log` |
| 统一对比 CSV | `5script/eval_unified_20260829.csv`（s15/s17/s18/s19） |
| ctrl 训练（在跑） | `5script/results/s20_ctrl_skel_flow_v2/`（注意 injections ckpt bug，见诊断报告） |
| 旁路监控 CSV | `5script/eval_s20_ctrl_monitor.csv` |

## 综合分析图（docs/system/imgs/）

- `fig5_landscape.png` — 全部 run 全景散点（颜色=eval 口径，x=收敛步数 log 轴）
- `fig6_cohort_best.png` — 7 个 eval 口径各自的历史最好成绩（口径难度演进）
- `fig7_pixel_era_axes.png` — pixel 时代设计变量效应（结构损失/宽度/清洗/patch）
- `fig1_unified_models.png` — 统一口径下 s15/s17/s18/s19/s20 对比
- `fig2_s20_curve.png` — s20 收敛曲线
- `fig3_condition_value.png` — 条件信息量 vs SSIM
- `fig4_design_axes.png` — 数据/架构/规模三轴增量效应

跨口径结论：旧口径数字不可比（图6 七张考卷）；同口径内可信的设计结论见
图4（数据 +0.087、新架构 +0.013、加宽 +0.004）与图7（结构损失有害、
加宽无益、DINO 嵌入微增、patch4 在 pixel 时代有效）。
