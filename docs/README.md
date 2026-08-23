# DiT-MCCD 文档索引

> 本目录是项目文档的统一入口。按主题组织，每个子目录覆盖一个领域。

## 目录结构

```
docs/
├── README.md                           # 本文件（文档索引）
├── model/                              # 模型架构
│   ├── architecture.md                 # DiT 模型设计 (2Cond/3Cond, S/XL, f4/f8)
│   └── CONTROLNET.md                   # ControlNet 骨架条件分支
├── training/                           # 训练管线
│   └── training.md                     # 训练循环、优化器、EMA、bf16、显存、远程启动
├── data/                               # 数据与 VAE
│   └── dataset.md                      # MCCD 数据集、latent shard 编码、VAE 工具
├── eval/                               # 评测
│   └── evaluation.md                   # auto_eval_cpu、eval_auto、指标、早停
├── experiments/                        # 实验记录
│   └── experiment_log.md               # V1→S7 实验时间线与关键决策
├── design/                             # 设计文档
│   └── 2026-08-15-sparse-compositional-calligraphy-dit.md
├── HANDOVER_2026-08-15.md              # 交接文档
├── s6_report/                          # S6 实验报告 (diff-only vs struct)
│   ├── REPORT.md
│   └── REPORT.pdf
└── legacy/                             # 历史文档
    ├── DATASET.md, TRAINING.md, INFERENCE.md, ...
```

## 快速导航

| 你想了解... | 去哪里看 |
|-------------|----------|
| 模型架构、条件融合、VAE 集成 | [model/architecture.md](model/architecture.md) |
| ControlNet 骨架引导 | [model/CONTROLNET.md](model/CONTROLNET.md) |
| 怎么训练、bf16/EMA/显存调优 | [training/training.md](training/training.md) |
| 数据集、latent 编码、VAE 对比 | [data/dataset.md](data/dataset.md) |
| 评测流程、auto_eval、早停机制 | [eval/evaluation.md](eval/evaluation.md) |
| 为什么做这些决策、实验历史 | [experiments/experiment_log.md](experiments/experiment_log.md) |
| 稀疏组合条件的设计论证 | [design/2026-08-15-sparse-compositional-...](design/2026-08-15-sparse-compositional-calligraphy-dit.md) |
| S6 结构损失对比报告 | [s6_report/REPORT.md](s6_report/REPORT.md) |

## VAE 工具文档

VAE 转换、编码、验证的完整流程文档在 `tools/vae/DATA_PIPELINE.md`，设计论证在 `tools/vae/README.md`。
