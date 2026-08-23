# DiT 书法生成（MCCD）

用 **DiT (Diffusion Transformer)** 做「书家 × 字体 × 汉字」条件书法字生成（256×256）。

```
输入: (书家 calligrapher_id, 字体 script_id, 汉字 character_id)
输出: 256×256 书法字图像
```

基于官方 DiT（[facebookresearch/DiT](https://github.com/facebookresearch/DiT)）改造成 2Cond 架构，支持：
- **factorized_add** 条件融合（书家 128d + glyph 256d → 投影相加）
- **ControlNet** skel 结构条件（3px skeleton 注入，zero-init warm-start / from-scratch）
- 从零预训练 / warm-start 两种训练模式
- LoRA 容量可调 / 结构 loss（Canny+Skeleton）可选
- 自动 CPU/GPU eval + 静态 HTML dashboard

> 远程训练服务器：`root@10.176.54.17:36430`，项目目录 `/root/Workspace/xy/DiT`。  
> 本仓库为本地工作副本，与远程代码保持同步（`train.py`/`models.py`/`eval_auto.py` 等）。

---

## 目录结构

```
src/                          # 核心源码（训练/模型/数据/loss/评测）
├── train.py                  # 主训练脚本 (DiT-2Cond/3Cond, latent/pixel)
├── models.py                 # DiT 模型定义 (2Cond/3Cond, S/XL)
├── losses.py                 # EdgeGradientLoss, SkeletonLoss, REPALoss, LatentStructLoss
├── latent_dataset.py         # MCCDLatentDataset (latent shards + skel/canny)
├── latent_structure.py       # StructDecoder (latent→skel/canny)
├── samplers.py               # DDIM 采样器
├── eval_auto.py              # 自动评测 (单步重建 + 自由采样 MSE/SSIM)
├── dataset.py                # MCCDDataset (pixel mode)
├── download.py               # 模型下载
└── lora.py                   # LoRA 注入

# 根目录镜像（与 src/ 保持同步）
train.py  models.py  losses.py  latent_dataset.py  ...

# 顶层工具脚本
auto_eval_cpu.py             # 独立 CPU 评测进程（轮询 ckpt）
auto_eval_ctrl.py            # ControlNet 评测（base vs ctrl, GPU/CPU）
eval_auto.py                 # 评测核心（单步重建 + 自由采样）
eval_metrics.py              # CalligraphyEvaluator (OCR/书家/字体准确率)
eval_models.py               # MultiTaskCalligraphyEvalNet (DINOv2 backbone)
sample.py                    # 推理采样脚本
gradio_app.py                # Gradio 前端（旧版）
flask_app.py                 # Flask API（旧版）

tools/                        # 实验工具与脚本
├── controlnet/               # ControlNet 分支（详见 docs/CONTROLNET.md）
│   ├── controlnet_dit.py     # ControlNetDiT + ControlConditionEncoder
│   ├── train_controlnet.py   # 训练（warm-start / from-scratch）
│   ├── sample_controlnet.py  # 推理
│   ├── test_controlnet.py    # 单元测试
│   ├── gradio_controlnet.py  # Gradio 前端（195k 推理 + GT 对照）
│   ├── ctrl_skel.json        # warm-start 配置 (top6)
│   └── ctrl_skel_top30.json  # from-scratch 配置 (top30, batch=112)
├── pull_ctrl_monitor.py      # ControlNet 训练日志拉取
├── build_ctrl_dashboard.py   # ControlNet 静态 HTML dashboard 生成
├── build_dashboards.py       # 通用静态 dashboard 生成
├── pull_monitor.py           # 通用训练日志拉取
├── pull_log.py               # 旧版日志拉取
├── pull_all.py               # 批量拉取实验数据
├── sync.py                   # 定时同步 + build dashboard
├── auto_eval_pixel.py        # Pixel-DiT 评测
├── train_dashboard.html      # Dashboard HTML 模板
└── ...                       # 分析/诊断/预处理脚本

diffusion/                    # 扩散过程（DDPM/DDIM）
labels/                       # ID 映射
5script/                      # 实验配置与数据
docs/                         # 文档
├── CONTROLNET.md             # ControlNet 设计文档
├── HANDOVER_2026-08-15.md    # 交接文档
└── s6_report/                # s6 实验报告
```

## 快速开始

### 训练（远程）

```bash
# 1. 同步代码到远程
rsync -avz --exclude='.git' --exclude='__pycache__' \
  -e "ssh -p 36430" ./ root@10.176.54.17:/root/Workspace/xy/DiT/

# 2. 拉起训练（tmux detached）
ssh -p 36430 root@10.176.54.17
cd /root/Workspace/xy/DiT
tmux new-session -d -s train \
  '/opt/conda/bin/python train.py --config exp_s6_top6_diffonly.json > log.txt 2>&1'

# 3. 拉起 CPU eval（后台）
nohup /opt/conda/bin/python auto_eval_cpu.py \
  --results-dir 5script/results/s6_top6_diffonly --interval 30 > cpu_eval.log 2>&1 &
```

### ControlNet 训练

```bash
# warm-start (冻结主模型)
tmux new-session -d -s ctrlSkel \
  '/opt/conda/bin/python tools/controlnet/train_controlnet.py \
   --config tools/controlnet/ctrl_skel.json > run_ctrl_skel.log 2>&1'

# from-scratch (主模型+ctrl 一起训练)
tmux new-session -d -s ctrlTop30 \
  '/opt/conda/bin/python tools/controlnet/train_controlnet.py \
   --config tools/controlnet/ctrl_skel_top30.json > run_ctrl_top30.log 2>&1'
```

### 前端推理

```bash
# Gradio 前端（195k 主模型 + GT 对照）
python tools/controlnet/gradio_controlnet.py
# 访问 http://127.0.0.1:7861/
```

### Dashboard

```bash
# 本地拉取远程日志 + 生成静态 HTML（每 30s 循环）
python tools/build_ctrl_dashboard.py --loop --interval 30
# 打开 tools/dashboards/ctrl_skel.html
```

## 模型架构

| 模型 | hidden | depth | heads | patch | 参数量 | 条件 |
|------|--------|-------|-------|-------|--------|------|
| DiT-2Cond-S/2 | 384 | 12 | 6 | 2 | 33.0M | callig + glyph |
| DiT-2Cond-XL/2 | 1152 | 28 | 16 | 2 | 673M | callig + glyph |

- **glyph_id** = f(character, script)，同字不同体有不同 ID
- **factorized_add**: callig_embed(128) + char_embed(256) → 投影相加
- **ControlNet**: 冻结主模型 + trainable ctrl_encoder(33.8M), zero-init 注入

## 评测指标

| 指标 | 说明 |
|------|------|
| MSE | 生成图 vs GT 的像素 MSE（256×256） |
| SSIM | 结构相似性（11×11 高斯窗） |
| OCR Acc | 字符识别准确率（DINOv2 分类器） |
| Callig Acc | 书家风格准确率 |
| Script Acc | 字体准确率 |

## 关键结论

- **ControlNet skel 条件有效**：warm-start 模式 MSE 0.443→0.249（-44%），SSIM 0.725→0.808（+8.2%）
- **warm-start 最佳点**：step 25000（top6 数据集），之后过拟合
- **from-scratch top30**：进行中（128k 样本，batch=112，20.45G 显存）
- **条件覆盖**：理论组合 1.74 亿，训练实际仅覆盖 17.7 万（0.01%）—— OOD 泛化是核心课题

## 相关文档

- [ControlNet 设计文档](docs/CONTROLNET.md)
- [交接文档](docs/HANDOVER_2026-08-15.md)
- [s6 实验报告](docs/s6_report/REPORT.md)

---

*License: 项目改造自官方 DiT（MIT），MCCD 数据集版权归原发布方。*
