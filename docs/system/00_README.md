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

## 4. 核心事实速查（2026-08-28）

| 项 | 值 |
|---|---|
| 主模型 | DiT-2Cond-S/2，~33M 参数，输入 (4,32,32) latent（f8 VAE） |
| 扩散 | 默认 **flow**（直线插值），模型输入 t×1000，Euler 50 步 |
| 训练速度 | ~3.36 steps/s，batch 240，显存 ~20.88G/24G |
| 优化器 | AdamW 2e-4，cosine + 3000 warmup，EMA 0.9999（warmup） |
| 数据 | mid-clean：118,776 行 / 5461 字符 / 67 书家 / 25 latent shards |
| 评测 | eval_strict_top6.csv（271 行），flow 默认 cfg=1.7，50 步 |
| 基线 | ddpm s6@195k：MSE 0.7872 / SSIM 0.5276 / skelIoU 0.0376；flow s18@43k：0.7246 / 0.5476 / 0.0395 |
| 当前运行 | s19 mid-clean 预训练（flow，cond_drop_which_glyph_prob=0.75） |