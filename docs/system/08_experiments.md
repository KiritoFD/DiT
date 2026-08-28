# 实验史与当前状态

> 详细时间线见 `docs/experiments/experiment_log.md`（历史）；本文件记录**与当前系统直接相关的关键决策与运行状态**（2026-08-28）。

## 1. 关键里程碑

| 阶段 | 说明 |
|---|---|
| V1→S5 | 早期 DiT 条件生成探索（3cond→2cond 因子化条件），像素结构损失（canny/skel） |
| S6 | **ddpm s6 @195k**：diff-only vs struct 对比报告（`docs/s6_report/REPORT.md`）+ eval500 修复（character_id 编码越界 146 样本剔除） |
| S12/S13 | 过拟合根因修复：减参 + 去 proj + **DINO 直通** + eval500 重建（S13 开始 DINO 字形嵌入） |
| S18 | **flow s18 @43k**：首个 Flow Matching 预训练（Euler 50 步），cfg=1.7；基线 MSE 0.7246 / SSIM 0.5476 / skelIoU 0.0395 |
| 2026-08-28 重构 | 核心代码分层 `src/{model,loss,train,eval,utils}` + 根 shim；eval/inference 统一核心；**统一时间步采样**修复 ControlNet bug 类 |
| S19（当前） | **mid-clean 数据集 + 4-way dropout 配比（which_glyph=0.75）+ flow 预训练** |

## 2. 当前运行状态（2026-08-28 14:37 重启后）

- **s19 mid-clean 预训练 RUNNING**：
  - 配置：`s19_midclean_s_flow.json` → `5script/results/s19_midclean_s_flow/`
  - 模型 DiT-2Cond-S/2 33M，batch 240，~3.36 steps/s，显存 ~20.88G/24G
  - loss 2.42 → 1.31 @ step 200（正常收敛）
  - `diffusion_type=flow`、`eval_cfg=1.7`、dropout 0.05/0.25/**0.75** 全部在 resolved_config.json 确认
  - **不要重启**；首个 ckpt 在 step 2500（含 in-process GPU eval + CPU daemon 指标）
- **ControlNet flow 重训（待办）**：等 s19 聚类后，以 s19 ckpt 为 `main_ckpt` 从干净起点重训（统一 `sample_t`、gpu_eval_cfg=1.7、in-process ctrl eval + step_tag daemon）。旧坏训练目录已删除。
- **eval_ctrl_metrics_daemon**：目前未启动（无 ctrl 在跑）；ctrl 训练启动时需同时拉起。

## 3. 关键决策记录

| 决策 | 依据 |
|---|---|
| flow 取代 ddpm | 训练更稳、步数更少（50 Euler），s18 在更少步数超过 s6 定式 |
| 默认 eval_cfg=1.7 | 用户指定：flow 最佳 cfg≈1.7，4.0 过强 |
| mid-clean 增广到 6 份 | 每 (script,char,calli) 组合恰 6 样本 → 类别频次可预测、长尾抹平 |
| `cond_drop_which_glyph_prob=0.75` | 67 书家样本充足 vs 5461 字符稀疏（中位 18）→ 专门预算倾斜给字符内容分 s_G（用户质疑原 1/8 配比后修正） |
| 统一 `sample_t` | 用户发现 flow/randint bug（t∈{0..49}→t*1000 OOD）后根治：调用方永不分支 |
| 冻结主模型 + zero-init ctrl | ControlNet 标准 warm-start：训练起点=主模型行为不变 |
| DINO 384 LN-only + 冻结字符表 | 去掉冗余投影，条件变成纯语义向量，S13 起验证有效 |

## 4. 后续计划

1. s19 预训练收敛（监控 loss / eval_auto 指标曲线）。
2. ControlNet flow 重训（warm-start s19 ckpt，`ctrl_skel_s19_flow.json` 新配置，固定 `sample_t` + cfg 1.7）。
3. 与 ddpm s6 195k / flow s18 43k 基线同口径对比（MSE/SSIM/skelIoU/LPIPS，eval_strict_top6 n=271）。
4. 文档随代码演进持续更新（本目录为权威版本）。