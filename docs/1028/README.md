# 2026-10-28 S/J 实验结果聚合

阶段基线入口：[250行结论与下一阶段报告](CONCLUSION_AND_NEXT_PHASE_20261028.md)。原始指标及逐样本表见[raw](raw/README.md)，本地归档包位于exports目录。

第二代训练已转入[24/48/64h完整队列](schedule64.md)：4090为S9及同起点FM/Pair分叉，48为J4–J7，后续条件晋级和S10/S11均已实现并挂起。旧矩阵结果仍在本页，队列最新状态见根目录training.md。

更新时间：2026-10-28 21:30（Asia/Shanghai）。本目录是当前 S 风格表征×注入矩阵和 J Decoder 联合训练的统一结果入口；训练现场仍以远端 `execution/status.json`、checkpoint `.done` 和评测 `complete.json` 为准。

## 一句话结论

- **S 系列没有出现全面超过朴素 S0 的新结构。** S0 仍有最高 SSIM/最低 LPIPS；S4（Writer×Local CNN + Spatial AdaGN）是新结构中最均衡的一组，提升 Ink-IoU、降低严重失败率并保持较强 Writer margin。
- **“低 Frag”不能单独作为连笔改善证据。** S1 的 Frag 更低，但笔画中位宽达到 GT 的 1.234 倍，并伴随 SSIM/LPIPS、墨量校准和 Writer margin 退化；海报显示主要是粗化与粘连。
- **S7 补齐后，Multi-Token × DynamicConv 没有叠加收益。** 它既没有超过同表征的 S3，也没有超过同注入的 S6；DynamicConv 当前更多改善部分低阶笔画统计，没有稳定改善字形、拓扑或书风。
- **J2/J3 的可训练 Decoder 带来小而稳定的像素/感知收益，但还不是拓扑或书风收益。** 两者相对 J0 的 SSIM/LPIPS改善可靠，容差骨架 F1和独立 Writer指标不升；J2强GT重建梯度相对J3没有额外生成收益。
- **最终模型选择仍需等 S8@50k 与 J@12k。** 当前结果是单 seed 过程筛选，不把未完成臂混入排名。

## S 系列：50k / val100

| Arm | 表征 × 注入 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ | Width/GT | Skel-F1 ↑ | Writer Top-1 ↑ | Margin ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | Vector × Global AdaGN | **0.6027** | **0.3395** | 0.2859 | 2.627 | **1.144** | 0.192 | **63%** | 0.0544 |
| S1 | Vector × Spatial AdaGN | 0.5866 | 0.3494 | 0.2806 | **2.265** | 1.234 | 0.179 | 53% | 0.0199 |
| S4 | Local CNN × Spatial AdaGN | 0.5988 | 0.3405 | **0.2951** | 2.299 | 1.196 | **0.196** | 62% | 0.0850 |
| S2 | Vector × Cross-Attention | 0.5949 | 0.3455 | 0.2835 | 2.489 | 1.193 | 0.191 | 54% | 0.0217 |
| S3 | Multi-Token × Cross-Attention | 0.5976 | 0.3426 | 0.2904 | 2.601 | 1.195 | 0.194 | 55% | 0.0169 |
| S5 | Local Attention × Cross-Attention | 0.5975 | 0.3425 | 0.2851 | 2.385 | 1.175 | 0.190 | **63%** | **0.1017** |
| S6 | Vector × DynamicConv | 0.5980 | 0.3428 | 0.2901 | 2.399 | 1.177 | 0.188 | 60% | 0.0356 |
| S7 | Multi-Token × DynamicConv | 0.5972 | 0.3431 | 0.2857 | 2.477 | 1.196 | 0.190 | 58% | 0.0467 |

完整 bootstrap 区间、失败率、BSD 与几何指标见 [S 指标](metrics/S.md) 和 [机器可读汇总](metrics/S.json)。

### 模型变量拆分

1. **只换注入：S0→S1。** Global AdaGN 换成 Spatial AdaGN 后，SSIM `-0.0162`、LPIPS方向 `-0.0099`，置信区间均不跨零；低 Frag 来自增粗粘连，不能判为结构提升。
2. **在 Spatial AdaGN 下换表征：S1→S4。** Local CNN 恢复大部分内容质量，并获得矩阵最高 Ink-IoU 和骨架 F1，说明局部表征与空间注入之间存在有益交互；不是 Spatial AdaGN 单独有效。
3. **在 Cross-Attention 下换表征：S2→S3/S5。** Multi-Token 只带来小幅 SSIM/LPIPS改善且 Frag变差；Local Attention改善 Frag和Writer margin，但没有提高Ink-IoU。更复杂的token并未形成统一收益。
4. **在 DynamicConv 下换表征：S6→S7。** S7的内容、几何和Writer均未超过S6，否定了“Multi-Token与动态卷积自然互补”的当前实现假设。
5. **同一 Vector 表征换注入：S0/S1/S2/S6。** S0整体最强；S6的BSD和笔宽W1最好，但这种低阶统计改善没有转化成更好的内容、拓扑或书风。

## J 系列：5k / heldout187

| Arm | Decoder目标 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ | Width/GT | Skel-F1 ↑ | Writer Micro ↑ |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| J0 | Decoder冻结，FM | 0.6149 | 0.3220 | 0.3437 | 2.397 | **1.046** | **0.242** | **64.17%** |
| J1 | Decoder冻结，FM+image | 0.6149 | 0.3222 | 0.3428 | 2.389 | 1.050 | 0.241 | 60.96% |
| J2 | Decoder可训，image+GT重建 | **0.6198** | 0.3150 | **0.3447** | 2.236 | 1.088 | 0.240 | 63.64% |
| J3 | Decoder可训，image、无GT重建 | 0.6193 | **0.3147** | 0.3433 | **2.186** | 1.086 | 0.238 | 62.03% |

J2 相对 J0 的 dSSIM `+0.0049 [0.0039,0.0060]`，J3为 `+0.0045`；两组LPIPS也稳定改善。J2/J3彼此不可区分，且容差骨架F1、笔宽、独立Writer没有同步提升。因此当前最稳妥的解释是：**解冻Decoder改善了局部外观/重建适配，但还没有重组笔画拓扑或增强真实书风。** 详见 [J 指标](metrics/J.md)、[机器可读汇总](metrics/J.json) 和 [Writer审计](metrics/writer.md)。

## 实际读图

![S系列对照](posters/S_comparison.png)

- S1在多行出现更粗、更密的墨块；低Frag主要来自相邻笔画合并。
- S4相对S1恢复可读性并减少局部墨块，是数值与视觉一致的最均衡新结构。
- S3有局部噪声/破碎尾部；S6/S7整体接近中间组，没有出现动态卷积带来的明确笔画组织优势。

![J系列对照](posters/J_comparison.png)

- J0/J1肉眼几乎一致，冻结Decoder时图像辅助损失没有有效改变最终生成。
- J2/J3也高度接近；变化集中在边缘、墨量和局部锐度，不像拓扑级重构。

海报按每位书家选择跨模型分歧最大的样本，同列固定STD、GT与噪声。选择索引见同目录 `*.indices.json`。

## 独立 Writer 评测边界

真实GT ceiling：S协议 86% Macro，J协议 75.85% Macro。评测器有辨识能力但并非完美书法学标准，因此只用于同协议相对比较。S0/S4/S5的Top-1差距在100张上不足以严格排序；J0领先J2/J3的1–3个百分点同样不能宣布显著。

## 当前训练现场

- **4090：** S8 约40.4k/50k，1.56 step/s，预计约1.7小时完成训练，之后还有100字评测；stage-2等待人工选择成对75k实验。
- **48GB：** S7已完成50k与全部评测。J0延长首次恢复因scheduler键名兼容错误中止；修复为兼容`_last_lr/last_lr`并校验参数组数量后，已从5k正确恢复，当前队列为J0→J1→J2→J3各到12k。
- J0恢复后已越过5150步，GPU约20.7GiB、98%；恢复调度LR从原5k值连续进入200步warmup，没有重置模型、Optimizer、EMA或RNG。

## 目录说明

- `metrics/S.md`、`metrics/J.md`：适合阅读的结论表。
- `metrics/S_full.md`、`metrics/J_full.md`：自动生成的完整bootstrap表。
- `metrics/S.json`、`metrics/J.json`：机器可读逐臂汇总与配对差值。
- `metrics/writer.md`：独立Writer原型评测、GT ceiling与实现修正。
- `posters/`：S/J跨模型海报及样本索引。
- `manifest.json`：本目录文件大小与小文件SHA；大海报仅登记path+size。

## 结论边界与下一步

1. 等S8@50k后补齐 Local CNN×DynamicConv，再决定S组延长对；当前优先候选是S0与S4，S5仅作为强Writer-margin对照。
2. J组按原计划到12k，至少比较5k/12k曲线；若J2/J3仍重合，后续无需保留GT重建锚。
3. 终评统一187字、至少3个noise seed，并加入Writer-swap与局部空间场/动态卷积冻结消融；本页100字/单seed结果仅用于架构筛选。
4. 不根据训练FM Loss选模型：S2尾段FM最低但生成并不最好，说明当前主要差异来自条件使用和输出几何，而非FM均值。
