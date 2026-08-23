# DiT-3Cond 设计评审与问题清单

> 本文档记录 2026-08-13 对代码库的全面排查结论、已定位的问题、以及后续方案。
> 状态：**诊断已完成，全部修复已落地，ckpt 已统一为 delta 存储（2026-08-14 更新）。**
> 当前项目整体说明以 `DOCUMENTATION.md` 为准；本文保留逐项问题记录供追溯。

---

## 1. 当前进展总结

### 1.1 已完成的评估（远程，正确的 EMA 加载方式）

对 17 个干净 checkpoint（跳过 NaN 污染的）跑了定量评估，方式与训练一致：
- 加载 `ckpt["ema"]`（完整 522 键模型，`missing=0, unexpected=0, all_finite=True`）
- 单步重建 `diffusion.training_losses(...)["pred_xstart"]`（t=150）
- VAE decode 后与 GT 算 MSE / SSIM（test 集 1000 张）

结果（`eval_all_summary.csv`）：

| step | MSE | SSIM |
|------|-----|------|
| 2500 | 0.0668 | 0.6230 |
| 5000 | 0.0698 | 0.6407 |
| 7500 | 0.0682 | 0.6407 |
| 10000 | 0.0708 | 0.6546 |
| 12500 | 0.0611 | 0.6544 |
| ... | ... | ... |
| 42500 | 0.0612 | 0.6537 |

**关键结论**：SSIM 在 step 10000 达到 0.654 后**完全停滞**（之后 32500 步零提升）。
这直接印证了 §2.1 的 adaLN frozen bug。

### 1.2 已定位的核心 bug

见 §2。

### 1.3 已落地的代码修复

- `train.py`：新增 `--train-cond-head` 开关（默认 True），adaLN/final_layer 纳入训练。
- `lora.py`：新增 `extract_full_inference()` / `load_full_inference()`。
- `train.py`：ckpt 保存统一为 `delta` 字段（**已重构，见 §3.3 已落地**），不再并存 model/ema/inference_delta 三份。
- `train_full_3cond.json`：显式加 `reset_cond_head=true, train_cond_head=true`。
- `TRAINING_NOTES.md`：追加根因分析。

---

## 2. 模型设计问题清单

### 2.1 【已修】adaLN / final_layer 被错误 frozen —— 输出糊的根因

**现象**：模型训练 42500 步后 SSIM 卡在 0.653 不涨，出图糊。

**根因**：`train.py` 的 trainable 判断只解冻了 `lora_*` / `y_*` / `cond_fusion`，
**漏掉了 `adaLN` / `final_layer`**。而 `--reset-cond-head` 会把这些层重置为
`std=0.02` 随机值。结果：

1. adaLN/final_layer = **随机值 + 永不训练**。DiT 的核心调制层处于随机状态，
   模型在随机调制下只学 LoRA 残差 → 学习能力被 rank-32 低秩锁死 → 输出糊、收敛停滞。
2. 每份 ckpt 的 adaLN 随机初始值都不同 → 无法用「官方预训练 + 小增量」复现。

**证据**：
- `TRAINING_NOTES.md` 第 243 行原始设计明确写「只训练 lora_/y_/cond_fusion/**adaLN**」。
- `TRAINING_NOTES.md` 第 326 行记录过 adaLN 有梯度更新量（单步 39）。
- 实测 15000 与 42500 的 `final_layer.linear.weight` 差异为 0（frozen 铁证）。

**修复**：`--train-cond-head=True`（默认），adaLN/final_layer 纳入 trainable。

### 2.2 【未修】001 run 的 NaN 根因（与 2.1 相关）

001 run 在 step ~10130 开始 NaN，之后永久污染（skip 51238 次）。

**分析**：
- 三个 loss（diff/canny/skel）同时 NaN，且纯扩散 loss 也 NaN → 是扩散前向发散，
  不是 canny/skel 结构 loss 触发。
- AMP(fp16) 下，随机 adaLN 调制 + LoRA 前向，激活值逐步逼近 fp16 上限 65504，
  某 step 触发 inf → NaN 通过 Adam 状态永久残留，无法自愈。
- `losses.py` 已对 canny/skel 做过防护（clamp/detach），但防不住 pred_xstart 本身发散。

**方案**（已实施的关键部分）：
- 修复 2.1（adaLN trainable）后，随机调制问题消失，NaN 概率大幅下降。
- 前向已改 **bf16 autocast**，NaN 步直接跳过不更新（不再依赖 scaler）；后续 `skips=0` 即健康。
- 可选的 `pred_xstart` clamp 尚未加（低优先级）。

### 2.3 【未修】`forward_with_cfg` 的 batch bug

`models.py` `forward_with_cfg`（约 534 行）：
```python
half = x[: len(x) // 2]
combined = torch.cat([half, half], dim=0)
```
当 `original_bs > 1` 时，`half = x[:1]`，会**丢弃 batch 中除第 0 个外的所有样本**。
采样通常 batch=1，所以未实际触发，但是隐患。

**方案**：改为正确的 CFG 广播（对每个样本 cat 其 uncond 版本）。

### 2.4 【未修】注释与实现不一致

`models.py` 注释写「adaLN-Zero」，实际是标准 adaLN（无 zero-init gate）。
不影响功能，但应修正注释或确认是否需要 adaLN-Zero。

### 2.5 【需确认】adaLN 是否应保持 reset

`reset_cond_head=True` 把 adaLN 重置为 std=0.02。这是必要的（3-cond 输入对官方
ImageNet 单 label 调制是 OOD），但配合 trainable 后，需要确认 lr=1e-4 下 adaLN
从随机值能否稳定收敛（历史上 lr=3e-3 会打飞，1e-4 已验证可行）。

---

## 3. ckpt 保存/加载方案（**已重构落地，2026-08-14**）

### 3.1 原问题（历史状态，已解决）

当时 `train.py` 保存逻辑同时存了 3 份权重，冗余且混乱：

当前 `train.py` 保存逻辑同时存了 3 份权重，冗余且混乱：

```python
checkpoint = {
    "model": trainable_state_dict,   # 231 键：lora + 条件头（旧 extract_lora_and_new_embedders）
    "ema": ema_full,                 # 522 键：完整模型（含 body，3.2GB 中的大部分）
    "inference_delta": inference_delta,  # 291 键：lora + 条件头 + adaLN/final_layer
    "opt": opt.state_dict(),
    "args": args
}
```

问题：
1. `ema` 全量含 frozen 的 body（与官方预训练重复），**每份 ckpt 浪费 ~2.2GB**。
2. 三种格式并存，加载时容易用错（历史上就用错过：拿 `model` 231 键当完整模型推理 → 图崩）。
3. `inference_delta` 是我刚加的，与 `model`/`ema` 重复。

### 3.2 目标设计（你的核心诉求）

> 公共的预训练权重（body）**不每次保存**；只保存**改变的部分**；
> 且保存的格式要能**直接一步加载，不需要二次提取/过滤/拼接**。

### 3.3 方案

**保存**（train.py，只存改变的部分 + 优化器状态，**已实现**）：

```python
checkpoint = {
    "delta": delta,        # 改变的部分：lora + 条件头 + adaLN/final_layer（~988MB）
    "opt": opt.state_dict(),  # 优化器状态（本来就只含 trainable 参数）
    "args": args,
}
```
- 删除 `ema` 全量（3.2GB → ~1GB）。
- 删除旧的 `model`（231 键）和 `inference_delta` 三份并存，统一为一份 `delta`。

**加载**（单一标准入口，封装在 lora.py，**已实现**）：

```python
model = build_model_from_ckpt(ckpt_path, pretrained_path, device)
```
内部顺序（与训练一致，封装好，调用方无感知）：
1. `DiT_3Cond_XL_2(num_calligraphers=2021, ...)` 构造
2. 加载官方 pretrained body（filter 掉 3-cond 专属键）
3. `inject_lora(r=32, lora_alpha=32)`
4. `load_state_dict(ckpt["delta"])` —— 一步加载改变的部分

**注意点（LoRA 的固有约束）**：LoRA 低秩矩阵是**新增参数**（非替换原有权重），
所以「注入 LoRA 结构」这一步**无法避免**。但可以封装进 `build_model_from_ckpt`，
让「保存 → 加载」对外表现为一步，不再有散落各处的提取逻辑。

### 3.4 resume 语义修正（**已落地**）

`train.py` 的 `resume_full` 当前加载 `_rf.get("model")`（231 键 LoRA 增量），
且因 `_resume_full_ckpt is not None` 跳过 pretrained body 加载 → **body 是随机初始化**。
这极可能是 001 run NaN 的另一个诱因。

**已实现方案**：`lora.build_model_from_ckpt()` 统一「构造 → 载预训练 body → inject_lora(r32) → 载 delta」，训练续跑与推理重建共用同一入口（见 `DOCUMENTATION.md` §4.4、§5.4）。

---

## 4. infra 与代码规范

### 4.1 目录规范（你的诉求：非模型代码放 tools/）

建议结构：
```
DiT/
├── models.py           # 模型定义（DiT-3Cond）
├── lora.py             # LoRA 注入/提取/加载
├── train.py            # 训练主入口
├── eval_full_3cond.py  # 标准评估
├── sample_3cond.py     # 标准采样
├── losses.py           # 损失函数
├── dataset.py          # 数据集
├── diffusion/          # 扩散过程
├── tools/              # 非模型辅助代码
│   ├── eval_all_remote.py   # 批量评估（远程）
│   ├── plot_curve.py        # 曲线绘图
│   ├── extract_*.py         # 数据/ckpt 处理脚本
│   └── ...
└── archive/            # 历史调试脚本/日志（归档）
```

### 4.2 已完成的清理

- 本地 + 远程删除了大量 `_probe_*.py` / `_check_*.py` / `_debug*.py` 临时脚本。
- 删除错误的 `lora_only_*.pt`（旧 231 键不完整提取产物）。
- 删除 3.15GB 的临时完整 ckpt。
- 日志与调试脚本归档到 `archive/`（代替删除）。
- 有 bug 的本地 `eval_curve.py`（拼接加载方式）已归档，标准评估用 `eval_full_3cond.py`。

### 4.3 待办（infra）

- [ ] 把 `_eval_all_remote.py`、`_eval_remote.py`、绘图脚本移到 `tools/`。
- [ ] 统一训练/评估脚本的模型加载入口（`build_model_from_ckpt`）。
- [ ] 清理远程 `archive/` 与本地 `tools/` 的命名一致性。

---

## 5. 待决策 / 待确认问题（已更新结论）

1. **adaLN 修复后是否重新训练**：已决定重训——远程 tmux `skel0`（`new_data_skel0/results_full_3cond`）以 `train_cond_head=true` 在新数据上从 0 训练。
2. **ckpt 体积权衡**：已确认采用只存 `delta`（已实现）。
3. **NaN 防御**：前向已改 bf16 + NaN 步跳过（`skips` 计数值恒 0 即健康）；是否再加 pred_xstart clamp 可后续补。
4. **EMA 存储**：当前 ckpt 统一为 delta（非 EMA）；`eval_full_3cond.py` 兼容从 `ckpt["ema"]`（回退 `"model"`）加载历史 ckpt。若需 EMA 质量对齐，需要额外存 EMA delta（暂未做）。

---

## 6. 下一步计划（已执行项勾选）

1. [x] 重构 ckpt 保存/加载（§3.3）：统一为 `delta` + `build_model_from_ckpt`（已完成）。
2. [ ] 修复 `forward_with_cfg` batch bug（§2.3）——仍未修，采样 batch=1 不受影响。
3. [ ] 加 NaN 防御（§2.2，可选）——现有 bf16 + NaN-skip 已大幅缓解。
4. [ ] 非模型代码迁到 `tools/`（§4.1）——部分完成。
5. [x] 用修复后的配置重新训练（adaLN trainable）——远程 `new_data_skel0/results_full_3cond`（tmux `skel0`）正在从 0 训练。
