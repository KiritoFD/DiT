# DiT 书法生成（MCCD）

用 **DiT (Diffusion Transformer)** 做「书家 × 字形」条件书法字生成（256×256）。

```
输入: (书家 calligrapher_id, 字形 glyph_id = script × character)
输出: 256×256 书法字图像
```

基于官方 DiT（[facebookresearch/DiT](https://github.com/facebookresearch/DiT)）改造成 2Cond 架构，支持：
- **factorized_add** 条件融合（书家 128d + glyph 256d → 投影相加，支持组合泛化）
- **kl-f4 VAE**（f4 下采样, 3ch latent, 55.3M params, 重建底噪 MSE=0.0019）
- **ControlNet** skel 结构条件（3px skeleton 注入，zero-init warm-start / from-scratch）
- 从零预训练 / warm-start 两种训练模式
- bf16 autocast + EMA + 自动 CPU eval + early stop

> 远程训练服务器：`root@10.176.54.17:36430`，项目目录 `/root/Workspace/xy/DiT`。
> 本仓库为本地工作副本，与远程代码保持同步（`train.py`/`models.py`/`eval_auto.py` 等）。

---

## 📖 文档

完整文档在 [`docs/`](docs/README.md)，按主题组织：

| 文档 | 内容 |
|------|------|
| [docs/model/architecture.md](docs/model/architecture.md) | DiT 模型架构 (2Cond/3Cond, S/XL, f4/f8 VAE) |
| [docs/training/training.md](docs/training/training.md) | 训练管线 (bf16, EMA, LR, 显存, 远程启动) |
| [docs/data/dataset.md](docs/data/dataset.md) | MCCD 数据集, latent 编码, VAE 工具 |
| [docs/eval/evaluation.md](docs/eval/evaluation.md) | auto_eval_cpu, 指标, 早停机制 |
| [docs/experiments/experiment_log.md](docs/experiments/experiment_log.md) | V1→S7 实验时间线与关键决策 |
| [docs/model/CONTROLNET.md](docs/model/CONTROLNET.md) | ControlNet 骨架条件分支 |
| [docs/s6_report/REPORT.md](docs/s6_report/REPORT.md) | S6 diff-only vs struct 对比报告 |

---

## 目录结构

```
src/                          # 核心源码（训练/模型/数据/loss/评测）
├── train.py                  # 主训练脚本 (DiT-2Cond/3Cond, latent/pixel)
├── models.py                 # DiT 模型定义 (2Cond/3Cond, S/XL, S/4)
├── losses.py                 # EdgeGradientLoss, SkeletonLoss, REPALoss
├── latent_dataset.py         # MCCDLatentDataset (auto-detect latent shape)
├── eval_auto.py              # 自动评测 (latent_channels/spatial/scaling 动态)
├── dataset.py                # MCCDDataset (pixel mode)
├── lora.py                   # LoRA 注入
└── ...

# 根目录镜像（与 src/ 保持同步，远程用）
train.py  models.py  losses.py  latent_dataset.py  eval_auto.py  ...
auto_eval_cpu.py  sample.py

configs/                      # 实验配置 (exp_*.json, resume_*.json)
├── s7_klf4_top30_diffonly.json  # 当前主跑: kl-f4, DiT-S/4, batch=224, 600k steps

tools/
├── vae/                      # VAE 转换/编码/验证/对比工具
├── controlnet/               # ControlNet 分支
├── dashboards/               # 静态 HTML dashboards
└── pull_monitor.py           # 远程日志拉取

diffusion/                    # 扩散过程 (DDPM/DDIM)
labels/                       # ID 映射 (calligrapher/character/script)
5script/                      # 数据 CSV + 实验配置
docs/                         # 项目文档（见上表）
```

---

## 模型架构

| 模型 | hidden | depth | heads | patch | params | VAE | tokens |
|------|--------|-------|-------|-------|--------|-----|--------|
| DiT-2Cond-S/2 | 384 | 12 | 6 | 2 | 33.0M | f8 (4ch, 32²) | 256 |
| **DiT-2Cond-S/4** | **384** | **12** | **6** | **4** | **41.8M** | **f4 (3ch, 64²)** | **256** |
| DiT-2Cond-B/2 | 768 | 12 | 12 | 2 | 132M | f8 | 256 |
| DiT-2Cond-XL/2 | 1152 | 28 | 16 | 2 | 673M | f8 | 256 |

**当前方案 S/4 + kl-f4**: 同样 256 tokens，但 latent 信息量 3× (12288 vs 4096)，VAE 底噪减半 (0.0019 vs 0.0037)。

---

## 快速开始

### 训练（远程 tmux）

```bash
# 1. 同步代码
scp -P 36430 train.py models.py eval_auto.py latent_dataset.py \
    root@10.176.54.17:/root/Workspace/xy/DiT/

# 2. 拉起训练 (GPU)
ssh -p 36430 root@10.176.54.17
cd /root/Workspace/xy/DiT
tmux new-session -d -s s7klf4 \
  '/opt/conda/bin/python train.py --config s7_klf4_top30_diffonly.json > run_s7_klf4.log 2>&1'

# 3. 拉起 CPU eval（不阻塞 GPU）
tmux new-session -d -s evalcpu \
  '/opt/conda/bin/python auto_eval_cpu.py \
    --results-dir 5script/results/s7_klf4_top30 \
    --workers 8 --worker-threads 8 \
    --seen5-csv 5script/seen5_top30.csv > auto_eval_cpu.log 2>&1'
```

### 本地 Dashboard

```bash
python tools/pull_monitor.py --loop --interval 30
# 打开 tools/dashboards/index.html
```

---

## 当前训练状态 (s7-klf4-top30-diffonly)

| 参数 | 值 |
|------|-----|
| 模型 | DiT-2Cond-S/4 (41.8M params, from scratch) |
| VAE | kl-f4 (f4, 3ch, 55.3M params) |
| 数据 | top30 calligraphers, 128,842 images |
| batch | 224 |
| max_steps | 600,000 (early stop: patience=6, min=60k) |
| 精度 | bf16 autocast |
| EMA | decay=0.9999, warmup |
| 速度 | 3.51 steps/s |
| 显存 | 19.74G / 24G (RTX 4090) |

---

## 关键结论

- **kl-f4 VAE 优于 sd-vae**: 重建底噪 MSE 0.0019 vs 0.0037 (减半), SSIM 0.988 vs 0.966, 参数更少
- **diff-only 优于 struct losses**: S6 报告证明 canny+skel 结构损失把 x0 推离 VAE 流形
- **factorized_add 优于 legacy**: 支持未见 (callig, glyph) 组合的组合泛化
- **ControlNet skel 有效**: warm-start MSE 0.443→0.249 (-44%), 但过拟合快 (25k 最佳)
- **条件覆盖极稀疏**: 理论 1.74 亿组合，训练仅覆盖 17.7 万 (0.01%) — OOD 泛化是核心课题

---

*License: 项目改造自官方 DiT（MIT），MCCD 数据集版权归原发布方。*
