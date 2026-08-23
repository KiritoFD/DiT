# ControlNet for Latent Calligraphy DiT

> 2026-08-23  
> 本文档覆盖 ControlNet 分支的设计、训练、评测与前端推理。

---

## 1. 概述

在已训练的 `DiT-2Cond-S/2` 书法生成模型基础上，加入 **3px skeleton 结构条件**（ControlNet 风格），
让模型在生成时受骨架图引导，提升字形结构准确度。

- 主模型：`DiT-2Cond-S/2`（hidden=384, depth=12, heads=6, patch=2, latent 4×32×32）
- 条件：3px skeleton（1 通道 256×256 二值图），area downsample 到 32×32
- ControlNet 分支：12 层 DiTBlockSimple（与主模型同 hidden/depth/heads），逐层 zero-init 注入

---

## 2. 文件结构

```
tools/controlnet/
├── controlnet_dit.py        # ControlNetDiT + ControlConditionEncoder + DiTBlockSimple
├── train_controlnet.py      # 训练脚本 (warm-start / from-scratch 两种模式)
├── sample_controlnet.py     # 推理脚本 (skel-mode gt/std)
├── test_controlnet.py       # 6 个单元测试 (全部通过)
├── gradio_controlnet.py    # Gradio 前端 (书家+字体+字输入, GT 对照, 历史累积)
├── eval_controlnet_cpu.py   # CPU 单步重建 eval (独立脚本, 已废弃, 改用 auto_eval_ctrl)
├── ctrl_skel.json           # warm-start 训练配置 (top6, LR=3e-4)
├── ctrl_skel_top30.json     # from-scratch 训练配置 (top30, batch=112)
├── calligraphers.json       # 11 位书家 (top6) 名称+ID
├── char_meta.json           # 3154 字 → {script: glyph_id} 映射 (top6)
└── supported_chars.txt      # 3154 支持字列表 (tab分隔 char\tcount)

auto_eval_ctrl.py            # ControlNet 专用 CPU/GPU eval (轮询 ckpt, base vs ctrl)
tools/pull_ctrl_monitor.py   # 训练日志拉取 → train_data.json (供 dashboard)
```

---

## 3. 模型设计

### ControlConditionEncoder
```
skel(1,256,256) → area_downsample(32×32) → PatchEmbed(patch=2) → (N,256,D)
→ 12 × DiTBlockSimple → 逐层 zero_init_linear projection → list[12] of (N,256,D)
```

### ControlNetDiT
- 包装冻结主模型，forward 时复现 DiT_2Cond 的 embedding + condition fusion
- 主模型 blocks 逐层执行，每层后注入 `x += ctrl_feats[i]`
- `out_projs` 零初始化 → 训练开始时注入=0，主模型行为不变（完美 warm-start）

### CFG
- skel 始终提供（对两半），CFG 只作用于 callig/char
- `forward_with_cfg`: 批量翻倍，cond/cond 各跑一遍，CFG 混合 eps

### 参数量
- 主模型（冻结）：33.0M
- ctrl_encoder（可训练）：33.8M
- from-scratch 模式：75.5M 可训练（主模型 33M + ctrl 33.8M + 条件头）

---

## 4. 训练

### 两种模式

| 模式 | train_ctrl_only | 主模型 | pretrained | 用途 |
|------|----------------|--------|-----------|------|
| warm-start | True | 冻结 195k ckpt | — | 快速加 skel 条件 |
| from-scratch | False | 从零训练 | 可选 body | 主模型+ctrl 同步学 |

### warm-start (top6, 已完成)
- 数据：`train_top6.csv`（10,866 样本，11 书家，3154 字）
- batch=96, LR=3e-4, warmup=500, 100k 步
- 显存 14.08G（主模型冻结，只有 ctrl 建图）
- **结果**：skel 条件有效，MSE 0.443→0.249（-44%），SSIM 0.725→0.808（+8.2%）
- **过拟合**：25k 步后效果退化（25k=0.249 → 100k=0.273）

### from-scratch (top30, 进行中)
- 数据：`train_top30.csv`（128,842 样本，108 书家，6952 字）
- batch=112, LR=1e-4, warmup=2000, max 200k 步
- 显存 20.45G（主模型+ctrl 都建图）
- 从零训练，无 pretrained body（DiT-XL/U-DiT-S 架构不匹配）

### 关键 infra
- 主模型冻结时 forward 不建训练图 → 只有 ctrl_encoder 建图
- 每步 `del loss, loss_dict` → 无 graph 残留
- 不加载 VAE（latent mode）→ 省 ~500MB
- EMA 只覆盖可训练参数
- 每 5000 步存 ckpt + `.done` 标记（供 auto_eval 轮询）

---

## 5. 评测

### auto_eval_ctrl.py
- 轮询 `_active_ckpt_dir.txt`，发现新 ckpt（带 `.done`）自动评测
- 每个 ckpt 跑两组：**base（无 skel）** vs **ctrl（有 GT skel）**
- 自由采样 DDIM 50 步 → VAE decode → MSE/SSIM vs GT
- 结果写 `eval_auto_{step}.json`
- 支持 `--device cuda`（GPU 加速）和 `--device cpu`

### warm-start eval 结果

| Step | MSE_ctrl | SSIM_ctrl | ΔMSE | ΔSSIM |
|------|----------|-----------|------|-------|
| 5000 | 0.2659 | 0.7974 | -0.177 | +0.072 |
| 15000 | 0.2553 | 0.8041 | -0.188 | +0.079 |
| 25000 | **0.2487** | **0.8079** | **-0.194** | **+0.082** |
| 100000 | 0.2734 | 0.7644 | -0.183 | +0.043 |

Base（无 skel）：MSE=0.4432, SSIM=0.7255（所有 ckpt 相同，主模型冻结）

---

## 6. 前端

### gradio_controlnet.py
- 195k 主模型推理（无 ControlNet）
- 书家下拉（11 位）、字体下拉（楷/隶）、字输入（3154 字）
- DDIM 50 步 CFG 采样，VAE decode
- 4 张 GT 对照图（完整展示，标书家+字体）
- 历史生成 gallery（累积）

### 条件映射
- `y_callig`：书家 ID（1011 类，factorized_add）
- `y_char`：glyph_id = f(character, script)（35130 类，同字不同体有不同 ID）
- GT 查找：`train_top6.csv` → `image_path` → `final_manifest.json` → MCCD 本地路径

---

## 7. 远程运行

```
# 训练 (tmux detached)
tmux new-session -d -s ctrlTop30 \
  '/opt/conda/bin/python tools/controlnet/train_controlnet.py \
   --config tools/controlnet/ctrl_skel_top30.json > run_ctrl_top30.log 2>&1'

# CPU/GPU eval (后台)
nohup /opt/conda/bin/python auto_eval_ctrl.py \
  --results-dir 5script/results/ctrl_skel --interval 30 > cpu_eval_ctrl.log 2>&1 &

# GPU 一次性 eval 所有 ckpt
/opt/conda/bin/python auto_eval_ctrl.py \
  --ckpt-dir <ckpt_dir> --device cuda --once --batch 16
```
