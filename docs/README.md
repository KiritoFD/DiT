# DiT-MCCD 文档索引

> 本目录是项目文档的统一入口。按主题组织，每个子目录覆盖一个领域。

## ⭐ 当前实现文档（authoritative，2026-08-28 重构后）

**`docs/system/`** 是当前代码的唯一权威说明，与代码一一对应，读它即可：

| 文档 | 内容 |
|---|---|
| [system/00_README.md](system/00_README.md) | 总览、管线图、核心事实速查、阅读顺序 |
| [system/01_code_layout.md](system/01_code_layout.md) | `src/{model,loss,train,eval,utils}` 分层、根 shim 兼容层、tools/ 边界 |
| [system/02_diffusion.md](system/02_diffusion.md) | 统一时间步设计（flow/ddpm 语义表）、**历史 bug 复盘**、`sample_t` 铁律 |
| [system/03_model.md](system/03_model.md) | DiT_2Cond 结构、**4-way 条件 dropout**、CFG、DINO 字符表 |
| [system/04_controlnet.md](system/04_controlnet.md) | ControlNet 骨架分支、zero-init warm-start、两种训练模式 |
| [system/05_dataset.md](system/05_dataset.md) | mid-clean 增广流水线（Phase A/B/C）、latent shards、数据事实 |
| [system/06_training.md](system/06_training.md) | train.py / train_controlnet.py、优化器/EMA、**配置字段全解**、resolved_config |
| [system/07_eval.md](system/07_eval.md) | 评测体系：inference 唯一核心 + 薄壳 + CPU daemon、指标约定、基线对比 |
| [system/08_experiments.md](system/08_experiments.md) | 实验史、关键决策、当前运行状态（s19 mid-clean 预训练中） |
| [system/09_ops.md](system/09_ops.md) | 远程部署、SSH 运维、GPU 纪律、踩坑清单 |

## 历史资料（重构前，仅存档参考）

> 以下文档描述 8 月中旬之前的代码与设计，**部分内容已过时**；与 `docs/system/` 冲突时以 system/ 为准。

```
docs/
├── model/                              # 模型架构（历史）
│   ├── architecture.md                 # DiT 模型设计 (2Cond/3Cond, S/XL, f4/f8)
│   └── CONTROLNET.md                   # ControlNet 骨架条件分支（旧版）
├── training/
│   └── training.md                     # 训练管线（旧版）
├── data/
│   └── dataset.md                      # 数据集与 VAE（旧版）
├── eval/
│   └── evaluation.md                   # 评测流程（旧版）
├── experiments/
│   └── experiment_log.md               # V1→S7 实验时间线
├── design/
│   └── 2026-08-15-sparse-compositional-calligraphy-dit.md
├── HANDOVER_2026-08-15.md              # 交接文档（旧）
├── s6_report/                          # S6 实验报告 (diff-only vs struct)
│   └── REPORT.md / REPORT.pdf
└── legacy/                             # 更早的文档
    ├── DATASET.md, TRAINING.md, INFERENCE.md, ...
```

## 快速导航（历史）

| 你想了解… | 去哪里看 |
|-------------|----------|
| （新）整套系统的权威说明 | [system/00_README.md](system/00_README.md) |
| （旧）模型架构、条件融合、VAE 集成 | [model/architecture.md](model/architecture.md) |
| （旧）ControlNet 骨架引导 | [model/CONTROLNET.md](model/CONTROLNET.md) |
| （旧）bf16/EMA/显存调优 | [training/training.md](training/training.md) |
| （旧）数据集、latent 编码、VAE 对比 | [data/dataset.md](data/dataset.md) |
| （旧）评测流程、auto_eval、早停 | [eval/evaluation.md](eval/evaluation.md) |
| S6 结构损失对比报告 | [s6_report/REPORT.md](s6_report/REPORT.md) |

## VAE 工具文档

VAE 转换、编码、验证的完整流程文档在 `tools/vae/DATA_PIPELINE.md`，设计论证在 `tools/vae/README.md`。