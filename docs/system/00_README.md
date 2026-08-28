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

## 4. 核心事实速查（2026-08-29，v2 主干）

| 项 | 值 |
|---|---|
| 主模型 | DiT-2Cond-S/2，**46.2M** 参数，输入 (4,32,32) latent（f8 VAE） |
| 骨干 | **RMSNorm + SwiGLU + 2D-RoPE + QK-Norm**（见 `13_arch_v2_modernization.md`，可回退） |
| 扩散 | 默认 **flow**（直线插值），模型输入 t×1000，**Heun 25 步**（= 50 NFE） |
| t 采样 | **logit-normal**（SD3），`shift=1.0` 不 shift |
| 训练速度 | ~4.1 steps/s，batch **128**，显存 ~15.8G/24G |
| 优化器 | AdamW **1.5e-4**（sqrt 缩放自 2e-4@240），cosine + 3000 warmup，EMA 0.9999 |
| 数据 | mid_common：23,597 行真实样本（**无增广**，见 `08_experiments.md`） |
| 评测 | eval_strict_midclean.csv（501 行，zero-shot=0、组合未覆盖），cfg=1.7 |
| 早停 | **ssim + lpips** 双指标，patience 5，min_delta 0.002 / 0.003 |
| 当前运行 | **s20 mid_common 预训练**（`run_s20_midcommon.sh`） |

> 关键环境约束：远程 4090 是 **torch 1.13.1**，无 `F.scaled_dot_product_attention`，
> 也无 xformers/flash-attn（sm_89 需 flash-attn 2.x，而 2.x 要 torch≥2.0）。
> 因此 attention 必然是 **eager**，显存占用高于 SDPA。详见 `13_arch_v2_modernization.md`。