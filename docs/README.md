# DiT-MCCD 文档索引

> 统一入口。**2026-10-05 做过一次归档整理**；**2026-10-09 做了双机资产整理 + 主表/主图重抽**（见下表）。历史文档全部移入 `docs/archive/`，本目录只保留**当前权威集**。找不到的旧文档见 §归档。

## ⭐ 权威集（读这些就够）

| 文档 | 内容 |
|---|---|
| **[infra-1015-measurements.md](infra-1015-measurements.md)** | **Infra 实测剖析报告**：单卡 4090 阶段级吞吐剖析、v88 170k 步收敛分析、**§10 评测快链路 2.44× 提速真机终测**（178s→73s） |
| [infra_1015.md](infra_1015.md) | 基础设施瓶颈剖析与工程重构建议规范（批次、通信、内存、编译、代码瘦身） |
| **[experiments/2026-10-16-v89-style-encoder-complete.md](experiments/2026-10-16-v89-style-encoder-complete.md)** | **V89 独立风格编码器完整报告**：20k 训练、留出验证、表征过拟合诊断与决策 |
| [experiments/2026-10-15-posttraining-and-v88-review.md](experiments/2026-10-15-posttraining-and-v88-review.md) | 后训练（A100/B25/B50/C/D）收尾总结与 v88 基模架构审查 |
| [refactor-report-2026-10-16.md](refactor-report-2026-10-16.md) | 架构重构与评测正确性修复报告（PR0~PR4 + 六项协议不变量落地 + OT 全链路连根拔除） |
| [v85e-final-report-2026-10-14.md](v85e-final-report-2026-10-14.md) | **v85e 最终评测报告**：风格判别器（Critic）引导与独立 Style 指标体系验证 |
| [v85-diagnostics-2026-10-14.md](v85-diagnostics-2026-10-14.md) | v85 诊断分析报告：模型容量、数据先验与生成稳定性诊断 |
| [v86A-writer-table-diagnostics-2026-10-14.md](v86A-writer-table-diagnostics-2026-10-14.md) | v86A 书家表微调诊断与梯度行为分析 |
| [c2ot_technical_specification.md](c2ot_technical_specification.md) | **C2OT 技术规范与证伪归档**：48 臂消融四指标全负实证复盘、SDE 独立配对理论冲突与 OT 连根切除记录 |
| [sde_analysis_brief.md](sde_analysis_brief.md) | **SDE 采样器与恒等式分析**：纯 Heun 与 Langevin 漂移纠偏 SDE、独立高斯配对解析 |
| [system/00_README.md](system/00_README.md) | 总览、管线图、核心事实速查、阅读顺序 |
| [system/01_code_layout.md](system/01_code_layout.md) | `src/{model,loss,train,eval,utils}` 分层、根 shim 兼容层、tools/ 边界 |
| [system/02_diffusion.md](system/02_diffusion.md) | 统一时间步设计（flow/ddpm 语义表）、历史 bug 复盘、`sample_t` 铁律 |
| [system/03_model.md](system/03_model.md) | DiT_2Cond 结构、4-way 条件 dropout、CFG、DINO 字符表 |
| [system/04_controlnet.md](system/04_controlnet.md) | ControlNet 骨架分支、zero-init warm-start、两种训练模式 |
| [system/05_dataset.md](system/05_dataset.md) | mid-clean 增广流水线（Phase A/B/C）、latent shards、数据事实 |
| [system/06_training.md](system/06_training.md) | train.py / train_controlnet.py、优化器/EMA、**配置字段全解**、resolved_config |
| [system/07_eval.md](system/07_eval.md) | 评测体系：**§8 现代评测快链路（batch 64 / GPU LPIPS / 2.44x 提速）**、指标约定 |
| [system/08_experiments.md](system/08_experiments.md) | 实验史与关键决策（**当前状态以 76 为准**） |
| [system/09_ops.md](system/09_ops.md) | 远程部署、SSH 运维、GPU 纪律、踩坑清单 |
| **[system/76_experiments_20261005.md](system/76_experiments_20261005.md)** | **全量实验整理与结论**（三目录 499 run → CSV → 同协议对照结论） |
| **[system/10_assets_cleanup_20261009.md](system/10_assets_cleanup_20261009.md)** | **双机资产整理记录**（48 删冗余 ckpt 回收 330G / 4090 run 家族重组 / 归档包与遗留项） |
| **[108/README.md](108/README.md)** | **双机资产总账与同步对账**（2026-10-09 晚）：远程 30 项死件隔离（43 GB）、**训练机代码落后本地 3 天（14 处真实分叉）**、616 点指标台账、本地 26 GB git 垃圾 |

## 📊 实验数据

| 文件 | 内容 |
|---|---|
| [../assets/ledger/points_all.csv](../assets/ledger/points_all.csv) | **逐评测点台账**（616 行 × 33 列，含在跑的 v72 与 5 个消融臂，2026-10-09 重抽） |
| [../assets/ledger/runs_all.csv](../assets/ledger/runs_all.csv) | 每 run 一行摘要 + 40 列配置快照（36 run） |
| [../assets/ledger/aligned_curve.csv](../assets/ledger/aligned_curve.csv) | step 对齐矩阵（12 条主线，用于画图/对读） |
| [108/03_metrics_ledger.md](108/03_metrics_ledger.md) | 台账字段说明 + 对齐曲线结论 + 消融台账 + 口径警告 |
| [experiments/mainline_20261009.csv](experiments/mainline_20261009.csv) | **主面 17 行大表数据**（`eval200_fixed` N=187 统一协议：v54~v71 + 容量阶梯 + moyi 复现，2026-10-09 重抽） |
| [experiments/all_runs_20261005.csv](experiments/all_runs_20261005.csv) | 每 run 一行：条件构成 + 曲线关键点 + best（499 run × 63 列，含 `eval_protocol` 列） |
| [experiments/all_runs_curves_20261005.csv](experiments/all_runs_curves_20261005.csv) | 长表 `(run_id, step, ssim, mse)`，870 点 |
| [experiments/README.md](experiments/README.md) | CSV 字段说明与查询示例 |

## 📈 可视化（2026-10-09 新增）

| 图 | 内容 |
|---|---|
| [system/imgs/fig_history_timeline_20261009.png](system/imgs/fig_history_timeline_20261009.png) | **实验历程时间线**：上=老协议时代 (v1→v21，seen/strict) · 下=187 统一协议时代 (v54→v71) |
| [system/imgs/fig_mainline_eval200_20261009.png](system/imgs/fig_mainline_eval200_20261009.png) | **主面对照大图**：eval200_fixed 187 协议下 v54/v56/v60/v61/v66/v68/v70 + 阶梯 + moyi 全曲线 |
| [system/imgs/fig_capacity_ladder_20261009.png](system/imgs/fig_capacity_ladder_20261009.png) | 容量阶梯：S/2 → B/2 → +增广 → +路由 逐级证明 |
| [system/imgs/fig_v68_family_20261009.png](system/imgs/fig_v68_family_20261009.png) | v68 家族与近线 run 曲线 (base/pix/dino_raw/randtab/v54/v66/v69/v70) |

> ⚠ **跨协议 SSIM 不可比**（四套评测集量级差 0.3）。详见 `system/76` §1；主表已统一为 187 协议。

## 🔧 当前已知问题

- [BUG_dynamo_recompile_oom_2026-10-04.md](BUG_dynamo_recompile_oom_2026-10-04.md) —— 训练期 dynamo 重编译打穿 `accumulated_cache_size_limit` → 回退 eager → OOM 的复盘

## 🗂 归档

| 位置 | 内容 |
|---|---|
| `docs/archive/20261005_slim/system/` | 旧 `system/` 历史文档（10~75 号） |
| `docs/archive/20261005_slim/04_experiments/` | 旧实验编年史（含 144KB `00_exhaustive_experiment_chronicle.md`）、leaderboard |
| `docs/archive/20261005_slim/01_architecture/`、`02_dataset/`、`03_training/` | 旧主题文档 |
| `docs/archive/20261005_slim/*.md` | 旧 HANDOVER / HISTORY_REVIEW / STATUS |
| `docs/archive/phase_922/`、`phase_919/`、`legacy_reports/`、`txt_reports/` | 更早的历史归档 |

## 其他

- **[`../state.md`](../state.md)**：**实验状态与对照关系（单一事实来源）** —— 谈任何数字之前先读它；登记在跑/在排实验的「变量 / 基准 / 口径 / 能否读结论」。
- VAE 转换/编码/验证流程：`tools/vae/DATA_PIPELINE.md`（设计论证 `tools/vae/README.md`）
- `docs/experiments/2026-*.md`：近期单次实验报告（保留原处）。其中：[2026-10-08-calli-vae-dino-repa-pretraining.md](experiments/2026-10-08-calli-vae-dino-repa-pretraining.md)（v71 前置）、[2026-10-09-calli-vae-decode-convention-rootcause.md](experiments/2026-10-09-calli-vae-decode-convention-rootcause.md)（VAE 解码范式根因）
- [107/README.md](107/README.md)：数据集规格 / 模型基准 / 证据分级与勘误 / 实验计划（规划集）
- [04_experiments/](04_experiments/)：全量实验超级编年史 + 分级基准海报（README §2 引用的专论集）
- `105/`、`106moyi/`：历史专题（105 = 阶段主题报告；106moyi = 墨意复现对比图文，700+ png 归档）
