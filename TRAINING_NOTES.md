# 训练笔记：NaN 根因排查与分阶段训练方案

记录本次 MCCD 书法数据集 DiT-2Cond 训练的排查过程与结论。时间：2026-08-12。

## 背景

本项目把标准 DiT-XL 改造成 `DiT_2Cond`（书法家 + 字符双条件），用 LoRA 微调，
叠加结构条件（Canny 边缘 + Skeleton 骨架）与 REPA 表示对齐。原始训练反复出现
`diff=nan` 后 `scaler.update()` 崩溃的问题。

## 阶段训练方案

为避免结构 loss 干扰 diffusion 主信号收敛，采用分阶段训练：

- **阶段 1（当前）**：纯 diffusion 驱动，关闭全部结构 loss。
  ```json
  "w_canny": 0.0,
  "w_skel": 0.0,
  "w_repa": 0.0,
  "use_canny": false,
  "use_skel": false
  ```
- **阶段 2（后续）**：diff 收敛后逐步打开 canny / skel 结构 loss。

## NaN 根因排查过程

### 症状
- 纯 diff 训练也在 step 148 / 165 出现 `diff=nan`
- 随后 `scaler.update()` 崩溃：
  ```
  AssertionError: No inf checks were recorded prior to update.
  ```

### 排查步骤（逐步排除）
1. **数据层**：扫描 2000 张图像 + VAE latent —— 全部 finite，latent 最大 |4.3|，
   远低于 fp16 上限 65504，排除坏样本。
2. **前向层**：fp16 前向 401×10 次（eval + train 两种模式）—— 0 NaN，前向稳定。
3. **反向层（关键）**：加入 backward 后，`use_checkpoint=True` 时 **step 0 梯度就在
   182 个可训练参数上全 NaN（grad=inf）**，而 loss 本身正常（0.10~0.20）。
   关闭 checkpoint 后 500 步 0 NaN，loss 从 0.10 正常收敛到 0.06。

### 根因
**`use_checkpoint=True`（gradient checkpointing）本身在 backward 时数值不稳定**，
与 fp16 AMP 无关（纯 fp32 + checkpoint 同样爆炸，gradmax 高达 2e37）。

原因：`torch.utils.checkpoint.checkpoint` 在反向时用 `no_grad` **重算前向**，但
训练模式下 **label dropout / LoRA dropout 是随机的**，重算的前向与原始前向中间激活
不一致，导致基于错误激活的反向梯度爆炸（inf/NaN），并污染全部可训练参数。

### 修复
将 `use_checkpoint` 置为 `false`：
```json
"use_checkpoint": false
```
- RTX 4090 有 24G 显存，batch=8 纯 diff 完全放得下（实际 ~12G），不需要 checkpoint。
- 关闭后 Steps/Sec 从 2.7 提升到 4.5，且训练稳定收敛（diff loss 0.07 附近）。

## 附带的代码 Bug 修复

原始 `train.py` 的 NaN 分支只打 warning、不调用 `scaler.scale/step`，导致
`scaler.update()` 断言 `No inf checks were recorded` 直接崩溃。修复：NaN 分支也
调用 `scale(loss).backward() + step()`，让 scaler 正确记录 inf、跳过该步并自动降 scale。

## 关于"显存为什么只用 12G"

因为阶段 1 已关闭 `use_canny/use_skel`，跳过了 VAE decode 结构 loss 分支
（不再生成 256×256×3 的 `x0_pred` 并算 Sobel 梯度），只走 latent（32×32×4），
省下了显存大头。之前 skel loss 那段才是显存峰值来源。

## 训练启动方式

远程单卡 base conda 环境，torch 2.6.0：
```bash
cd /root/Workspace/xy/DiT
nohup /opt/conda/bin/python train.py > /tmp/train_stage1.log 2>&1 &
```
注意：必须用 `/opt/conda/bin/python`（`python` 不在 PATH）。若 `diffusers` 导入报
`hf_cache_home`，需将 `huggingface_hub` 降到 `0.23.x`。
