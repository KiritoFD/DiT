# 系统实现文档 — 总览

> 本目录是**当前代码的权威说明**（2026-08-28 重构后）。`docs/legacy/` 与旧文档为历史资料，仅存档参考。
> 文档与代码一一对应：`docs/system/01_code_layout.md` 列出完整文件树，每个模块的文档都标注了对应源文件。

## 1. 这是什么

面向中文书法的 **DiT（Diffusion Transformer）生成系统**，带 **ControlNet 骨架结构控制**：

- **无条件/条件生成**：以「书家（calligrapher）+ 字符字形（glyph）」为条件，在 VAE latent 空间训练 DiT，生成 256×256 书法字图。
- **双扩散框架**：同一套代码同时支持经典 **DDPM**（epsilon 预测 + DDIM 采样）与 **Flow Matching**（直线插值 velocity 预测 + Euler ODE 采样），通过配置一键切换，时间步采样统一（见 `02_diffusion.md` —— 这是本系统最重要的正确性设计）。
- **结构控制**：ControlNet 分支注入骨架图（skel）条件，使生成结果遵循给定字形结构。

## 2. 系统管线总览

```
                        ┌─────────────────────────────────────────────┐
  数据（远程）           │              模型 / 训练（远程 GPU）           │
 ┌──────────────┐       │ ┌─────────────────────────────────────────┐ │
 │ MCCD 原始 329k │       │ │ src/model/dit.py    DiT_2Cond (S/2)      │ │
 │  GB2312 清洗  │       │ │ src/model/controlnet.py  ControlNetDiT    │ │
 │  23.6k 原样本 │       │ └─────────────────────────────────────────┘ │
 │  增广→118.8k  │       │ ┌─────────────────────────────────────────┐ │
 │  VAE→latent  │       │ │ src/loss/             FlowMatching /     │ │
 │  shards      │       │ │                       GaussianDiffusion  │ │
 └──────────────┘       │ │                       (统一 sample_t)     │ │
        │               │ └─────────────────────────────────────────┘ │
        ▼               │ ┌─────────────────────────────────────────┐ │
 5script/*.csv          │ │ src/train/train.py        主模型预训练    │ │
  (条件/评测表)          │ │ src/train/train_controlnet.py  ControlNet│ │
                        │ └─────────────────────────────────────────┘ │
                        └──────────────────┬──────────────────────────┘
                                           ▼
                             ckpt → in-process GPU 采样
                                   (bf16 DiT + fp32 VAE)
                                           ▼
                             eval_samples*/stepXXXXXXX/{base,ctrl}/*.png
                                           ▼
                              CPU 指标 daemon → eval_auto*.json
                                    (MSE/SSIM/skelIoU/LPIPS)
```

- **GPU 只有一张 24G 卡**（远程 4090 机器），训练期间约占 20.9G，**不允许并行 GPU 任务**（编码/评测须等训练停）。
- 开发机（Windows）与远程（Linux）通过 SSH 同步**代码文件**；数据（csv/图片/latent/ckpt）只存在远程，见 `09_ops.md`。

## 3. 阅读顺序

| 你想了解… | 文档 |
|---|---|
| 每个文件在哪、为什么这样分层 | [01_code_layout.md](01_code_layout.md) |
| 统一的扩散时间步设计（flow/ddpm，含历史 bug 复盘） | [02_diffusion.md](02_diffusion.md) |
| 模型结构、4-way 条件 dropout、CFG | [03_model.md](03_model.md) |
| ControlNet 骨架分支、warm-start | [04_controlnet.md](04_controlnet.md) |
| mid-clean 数据集与 latent 流水线 | [05_dataset.md](05_dataset.md) |
| 两个训练入口 + 配置字段全解 | [06_training.md](06_training.md) |
| 评测体系（一个核心 + 薄壳 + daemon） | [07_eval.md](07_eval.md) |
| 实验史、基线与当前状态 | [08_experiments.md](08_experiments.md) |
| 远程部署、重启、监控、踩坑 | [09_ops.md](09_ops.md) |
| 2026-08-28 问题诊断（v2 改造的动因） | [10_diagnosis_20260828.md](10_diagnosis_20260828.md) |
| 字形泛化分析 | [11_glyph_generalization.md](11_glyph_generalization.md) |
| DINO 条件实测（有效秩/检索/书体泄漏） | [12_dino_diagnosis_20260829.md](12_dino_diagnosis_20260829.md) |
| **v2 架构现代化（当前主干）** | **[13_arch_v2_modernization.md](13_arch_v2_modernization.md)** |
| **字符条件探测 + s22 诊断** | **[14_glyph_condition_probe.md](14_glyph_condition_probe.md)** |
| **代码/模型/文档问题清单** | **[15_codebase_review_20260830.md](15_codebase_review_20260830.md)** |
| **模型架构全景（参数/条件维度实测）** | **[16_model_architecture.md](16_model_architecture.md)** |
| **实验结果全集（640 评测点 / 47 实验）** | **[17_experiments_registry.md](17_experiments_registry.md)** |
| **数据资产清单** | **[18_data_assets.md](18_data_assets.md)** |
| **核心发现与方向（先读这篇）** | **[19_core_findings.md](19_core_findings.md)** |

> **新读者建议**：先读 [19_core_findings.md](19_core_findings.md)（一页看懂现状），
> 再按需查 [16 架构](16_model_architecture.md) / [17 实验](17_experiments_registry.md) / [18 数据](18_data_assets.md)。
> 这三篇的数据来自 `5script/*.csv`，由 `_scan_*.py` 自动生成，可随时重跑刷新。

### 3.1 结构化数据（CSV）

| CSV | 行数 | 内容 |
|---|---:|---|
| `5script/eval_points.csv` | 640 | 每个评测点（实验 × step × arm × 指标） |
| `5script/experiments_enriched.csv` | 47 | 实验级指标 + 数据集难度维度 |
| `5script/difficulty_summary.csv` | 7 | **数据难度 vs 指标**（SSIM 不可跨数据集比较） |
| `5script/skel_ablation.csv` | 20 | **骨架消融 1px / 3px / std-skel** |
| `5script/data_assets.csv` | 78 | 数据集 / latent / 骨架 / 字形库清点 |
| `5script/model_components.csv` | 10 | 模型逐模块参数量 |
| `5script/condition_dims.csv` | 4 | 条件信号维度对比 |

## 4. 核心事实速查（2026-08-30，fame 主干）

| 项 | 值 |
|---|---|
| 主模型 | DiT-2Cond-S/2，**46.52M** 参数，输入 (4,32,32) latent（**f4 VAE**，scaling 0.18215） |
| 骨干 | **RMSNorm + SwiGLU + 2D-RoPE + QK-Norm**（见 `13_arch_v2_modernization.md`，可回退） |
| 参数分布 | Transformer blocks **68.57%**；`y_char_embedder` **29.0%（默认冻结）** |
| 条件 | 书家 128 维 + 字 384 维（DINO，**有效秩仅 34**）；可选空间条件（骨架/字形 latent 4,096 维） |
| 扩散 | 默认 **flow**（直线插值），模型输入 t×1000，**Heun 25 步**（= 50 NFE） |
| t 采样 | **logit-normal**（SD3），`shift=1.0` 不 shift |
| 训练速度 | ~4.3 steps/s，batch **192**，显存 ~16.8G/24G |
| 优化器 | AdamW，cosine + warmup，EMA 0.9999 |
| 数据 | **fame：51,322 样本 / 44 书家 / 4,765 字 / 7 书体** |
| 评测 | `eval_fame_strict.csv`（500 行，字与书家均 100% 见过，测组合泛化），cfg=**0.7** |
| 早停 | **ssim + lpips** 双指标，patience 5，min_delta 0.002 / 0.003 |
| 当前最佳 | 预训练 **s21 SSIM 0.4664**@27.5k；ControlNet(3px) **0.7889**@50k |
| 当前运行 | **fame-ctrl-skel-1px**（1px 细骨架 ControlNet） |

> ⚠️ **SSIM 不可跨数据集比较**：11 书家上可达 0.73，44 书家上只有 0.47。
> 报告指标必须同时给出评测集书家数，见 `difficulty_summary.csv`。
>
> ⚠️ **ControlNet 的 0.80 是「复刻」指标**（骨架从目标图抽取），不是泛化指标。
> 真正的泛化基准是 base 的 0.50。详见 [19_core_findings.md](19_core_findings.md)。

> 关键环境约束：远程 4090 是 **torch 1.13.1**，无 `F.scaled_dot_product_attention`，
> 也无 xformers/flash-attn（sm_89 需 flash-attn 2.x，而 2.x 要 torch≥2.0）。
> 因此 attention 必然是 **eager**，显存占用高于 SDPA。详见 `13_arch_v2_modernization.md`。