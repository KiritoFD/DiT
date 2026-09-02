# 预训练实验进展与结论（2026-09-02）

> 汇总 base → skel → repa 三阶段预训练实验，量化各改动对生成质量 / 噪声类指标的影响。
> 数据看板: `tools/pretrain_eval_dashboard.html`（base/skel/repa 分组，横轴 steps，纵轴 SSIM/LPIPS/MSE/SkelIoU）
> 视觉对比: `tools/make_base_model_grid.py --group all --step 30000 --out _sync_work/grid_all_s30000.png`

## 0. 一句话结论

**对"噪声类"指标（LPIPS / MSE）改善最大的两个改动是：骨架控制（1-pix skel）和表示对齐（REPA）**。
它们把 LPIPS 从 base 的 ~0.44 降到 ~0.14、MSE 从 ~1.0 降到 ~0.12（降 3–8 倍），骨架吻合度（SkelIoU）从 ~0.014 升到 ~0.43。

## 1. 实验总览（best SSIM @ step，含 LPIPS/MSE/SkelIoU）

| 阶段 | 实验 | 改动 | best SSIM | LPIPS↓ | MSE↓ | SkelIoU↑ |
|---|---|---|---|---|---|---|
| base | s21 | 真迹 DINO (ln_only) | 0.4670 | 0.4416 | 1.034 | 0.0144 |
| base | s25 | IDS 部件码本 | 0.4618 | 0.4483 | 1.045 | 0.0142 |
| base | s28 | 标准字形 DINO+PCA+OT | 0.4476 | 0.4744 | 1.079 | 0.0115 ❌ |
| **base** | **s30** | **DINO char-strong + 清洗后数据** | **0.4841** | **0.4345** | **0.930** | **0.0161** |
| skel | s26 | GT skel 1px (b96) | 0.5031 | 0.4245 | 0.952 | 0.0142 |
| skel | s31 | GT skel 1px (b192) | 0.5631 | 0.4372 | 0.807 | 0.0072 |
| **skel** | **1pix** | **ctrl_fame 1-pix skel v1** | **0.7974** | **0.1579** | **0.149** | **0.3318** |
| **repa** | **s32b** | **repa-strong（接 1pix）** | **0.8177** | **0.1420** | **0.124** | **0.4021** |
| **repa** | **s32c** | **repa-chain/longconv** | **0.8204** | **0.1407** | **0.121** | **0.4313** |
| 在跑 | v8a_s30_base | s30 的 v8 变体（cu121） | — | — | — | — |

> s29（bench b96/b192）、v8_3stage 暂无 eval 产出；s32c 已结束，当前在跑 `v8a_s30_base`。

## 2. 各阶段加了什么、效果如何

### 2.1 base 阶段（s21 → s30）
- **s21 真迹 DINO**：CLS/patch 注入 callig/char embedding，base 基线 0.467。
- **s25 IDS 部件码本**：用 IDS 部件码本替换 char embedding —— **下游差**，SSIM 0.462，放弃该分支（用户明确不看 IDS）。
- **s28 标准字形 DINO+PCA+OT**：把 DINO 特征换成"标准字形"的 CLS 特征 + PCA 降到 384 + OT 直通 —— **失败**（0.448，且 step 3000 后停滞）。
  原因：印刷体标准字形与真迹书法存在域差，且 PCA 降维损失信息。
- **s30 DINO char-strong（推荐 base）**：用更强的 DINO char embedding，**并在用户清洗后的数据上重训**（见 §3）。
  达到 base 最佳 **0.4841**（比 s21 高 +0.017）。LPIPS/MSE 也略优。

### 2.2 skel 阶段（骨架 ControlNet）
- 在 base 的 DiT 上加 ControlNet，条件为标准字形 1-pix 骨架。
- **s26 / s31 失败**：SSIM 仅 0.50–0.56，且从 grid 看 s31 近乎全白（未收敛/训练崩）。
- **1pix（ctrl_fame_1pix_v1）成功**：SSIM 跳到 **0.7974**，LPIPS 降到 0.158、MSE 降到 0.149、SkelIoU 升到 0.33。
  **这是 base→skel 最大的单点提升**，前提是 1-pix 骨架 + 正确收敛设置。

### 2.3 repa 阶段（表示对齐，接在 skel/1pix 之后）
- **s32b repa-strong / s32c repa-chain**：在 1pix 骨架上加 REPA（表示对齐到干净数据分布）。
- SSIM 进一步到 **0.82**，LPIPS 0.14、MSE 0.12、SkelIoU 0.43（全链路最佳）。
- grid 视觉上 s32c 比 1pix 更稳定、伪影更少。

## 3. 数据清洗在哪一步起作用

- 用户在 **s30 重训前**对 fame 数据做了清洗（反相修复、黑框 crop、孤立噪点删除），重新 encode 后训练 s30。
- 清洗的具体内容见 `docs/system/28_data_cleaning_survey.md`：
  - 全量扫描发现真正需修复的仅占少数（反相 0.24%、黑边 1.69%、边界墨 2.84%、散落噪点 1.59%）；
    而 `foreign/main_frac` 高者**多为草书/飞白的正常多连通域，不是污染**（误判陷阱）。
  - 方案 B（基于标准字形 bbox）全量修复 18785/51322（36.6%），主连通域保留 0.9897，过度删除仅 2.49%，噪点基本清零。
- **效果**：清洗后的 GT 更干净 → 训练信号更准 → s30 比未充分清洗的 s28 显著更好，是 base 提升的间接贡献之一。
- 注意：用户后续又做了 `train_fame_clean_v8.csv`（v8 清洗，→ `final_imgs_fame_v8/`），是与 clean_v2(B) 并行的另一轮尝试，说明清洗是持续迭代项。

## 4. 对"视觉（噪声类）指标"的好处分析

| 改动 | LPIPS 变化 | MSE 变化 | SkelIoU 变化 | 是否对噪声类指标有益 |
|---|---|---|---|---|
| 真迹 DINO→char-strong (s21→s30) | 0.442→0.435 | 1.03→0.93 | 0.014→0.016 | 轻微有益（主要靠清洗数据） |
| 标准字形 DINO (s28) | 0.442→0.474 | 1.03→1.08 | ↓ | **有害**（域差+降维损失） |
| IDS 部件 (s25) | ≈持平 | ≈持平 | — | 无益 |
| **1-pix 骨架 (1pix)** | **0.435→0.158** | **0.93→0.149** | **0.016→0.332** | **极大有益** |
| **REPA (s32c)** | **0.158→0.141** | **0.149→0.121** | **0.332→0.431** | **极大有益** |

**结论**：
- **骨架控制（1-pix skel）和 REPA 是对噪声类指标最有益的两个改动**。它们的共同机制是
  "把生成约束到干净结构分布"——骨架钉死字形结构，REPA 让 latent 表示贴近干净数据，
  二者叠加使 LPIPS/MSE 下降 3–8 倍、SkelIoU 提升 30 倍。
- 纯 base 侧的 embedding 改动（含数据清洗）对 SSIM 有 +0.017 的温和提升，但对噪声类指标改观有限；
  其价值在于为 skel/repa 提供更好的初始化。

## 5. 哪些改动是"好的"（ adopt / drop 清单）

✅ **采纳**
- 真迹 DINO char embedding（s21 基线，s30 加强版）
- 数据清洗（反相/黑边/噪点，方案 B），提升 GT 质量
- 1-pix 骨架 ControlNet（1pix），base→skel 最大单点提升
- REPA 表示对齐（s32b/s32c），全链路最佳

❌ **放弃**
- 标准字形 DINO+PCA+OT（s28，域差+信息损失）
- IDS 部件码本（s25，下游差）
- 清洗方案 C（空间拓扑，误删正常笔画）
- s26/s31 的 skel 配置（未收敛）

🔄 **在跑 / 待观察**
- `v8a_s30_base`（s30 的 v8 变体，cu121 环境，已迁移 torch2.1.2+xformers）
- 环境已从 torch1.13/cu117 升级到 torch2.1.2/cu121，训练吞吐已提升（steps/s 由 ~3.4 升到 ~4.2）
