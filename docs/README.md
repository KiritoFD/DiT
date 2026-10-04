# DiT-MCCD 文档索引

> 统一入口。**2026-10-05 做过一次归档整理**：历史文档全部移入 `docs/archive/`，
> 本目录只保留**当前权威集**（下表）。找不到的旧文档见 §归档。

## ⭐ 权威集（读这些就够）

| 文档 | 内容 |
|---|---|
| [system/00_README.md](system/00_README.md) | 总览、管线图、核心事实速查、阅读顺序 |
| [system/01_code_layout.md](system/01_code_layout.md) | `src/{model,loss,train,eval,utils}` 分层、根 shim 兼容层、tools/ 边界 |
| [system/02_diffusion.md](system/02_diffusion.md) | 统一时间步设计（flow/ddpm 语义表）、历史 bug 复盘、`sample_t` 铁律 |
| [system/03_model.md](system/03_model.md) | DiT_2Cond 结构、4-way 条件 dropout、CFG、DINO 字符表 |
| [system/04_controlnet.md](system/04_controlnet.md) | ControlNet 骨架分支、zero-init warm-start、两种训练模式 |
| [system/05_dataset.md](system/05_dataset.md) | mid-clean 增广流水线（Phase A/B/C）、latent shards、数据事实 |
| [system/06_training.md](system/06_training.md) | train.py / train_controlnet.py、优化器/EMA、**配置字段全解**、resolved_config |
| [system/07_eval.md](system/07_eval.md) | 评测体系：inference 唯一核心 + 薄壳 + CPU daemon、指标约定 |
| [system/08_experiments.md](system/08_experiments.md) | 实验史与关键决策（**当前状态以 76 为准**） |
| [system/09_ops.md](system/09_ops.md) | 远程部署、SSH 运维、GPU 纪律、踩坑清单 |
| **[system/76_experiments_20261005.md](system/76_experiments_20261005.md)** | **全量实验整理与结论**（三目录 499 run → CSV → 同协议对照结论） |

## 📊 实验数据（本次新增）

| 文件 | 内容 |
|---|---|
| [experiments/all_runs_20261005.csv](experiments/all_runs_20261005.csv) | 每 run 一行：条件构成 + 曲线关键点 + best（499 run × 63 列，含 `eval_protocol` 列） |
| [experiments/all_runs_curves_20261005.csv](experiments/all_runs_curves_20261005.csv) | 长表 `(run_id, step, ssim, mse)`，870 点 |
| [experiments/README.md](experiments/README.md) | CSV 字段说明与查询示例 |

> ⚠ **跨协议 SSIM 不可比**（四套评测集量级差 0.3）。详见 `system/76` §1。

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

- VAE 转换/编码/验证流程：`tools/vae/DATA_PIPELINE.md`（设计论证 `tools/vae/README.md`）
- `docs/experiments/2026-*.md`：近期单次实验报告（保留原处）
