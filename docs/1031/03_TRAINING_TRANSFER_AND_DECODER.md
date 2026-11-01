# 训练策略与后训练：P75、A/B/C、Decoder、预训练归因

> 研究目的：把“后训练长度不足”“Writer 应先适配”“通用预训练有净收益”“Decoder 应当单独微调”拆成四个不同的因果问题。证据主要来自 [1030p125.md](../../1030p125.md)、[1031.md](../../1031.md)、[Current](../../Current.md)、[A/B/C 配方源码](../../tools/p125/build_abc.py)、[J组严格对照](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)。

## 1. 当前 P75 的问题更像“条件曝光不足 + 训练合同混杂”，而不是通用预训练已被证伪

实际训练 P0通用50k→P1混合5k→P2 Top10 15k→P3A/B Top10 5k。文件名中的 `P2@120k` / `P3@125k` 是原100k P0计划保留下来的别名，真实累计步数 **50/55/70/75k**。

每一阶段按有效batch归一化的 Top10 曝光**期望**（尚未从随机训练日志逐行点数）：

| 阶段 | 实际步数 | Top10比例 | Writer非空率 | Top10图像等效步 | 带Writer等效步 |
|---|---:|---:|---:|---:|---:|
| P0 | 50k | 0（目标Top10 Writer） | — | 0 | 0 |
| P1 | 5k | .30 | .75 | 1.5k | 1.125k |
| P2 | 15k | 1.00 | .90 | 15k | 13.5k |
| P3A/B | 5k | 1.00 | .90 | 5k | 4.5k |
| **合计** | **75k** | — | — | **21.5k** | **19.125k** |

“75k更新”不等于“75k步的Top10已知Writer训练”。S0/S5从头50k在目标Writer训练的条件曝光远高于19.125k；P0仍可能提供通用结构先验，不能因为当前P75分数偏低就判预训练负迁移。

P75 val100：P0 **SSIM .5573 / E-G AUC53.69**→P1 **.5636/72.22**→P2 **.5857/77.77**→P3A **.5857（E-G待补）**。P1的E-G跳升 **+18.53pp** 恰好发生在Top10 Writer开始接受监督时，**不等于混合P1优于直接Top10**；P2之后内容和风格继续改善，且仍落后成熟scratch S组。单seed P3B相对P3A：SSIM **+.0024**、LPIPS **−.0023**、IoU **+.0020**，不证明E-G监督带来独立风格收益。

## 2. 必须将实际实现错误纳入因果分析

### 2.1 P75旧 Style LR 真值是 1e-4

[训练参数分组](../../tools/style_matrix/runtime/common.py) 的旧 `rate(kind,legacy)` 拼出 `lr_style_representation` 和 `lr_style_injections`，但Config字段名是 `lr_style_repr`、`lr_style_inj`，导致回退至 legacy **1e-4**。P3A checkpoint `initial_lr=1e-4` 已证实，并非猜测。

2026-10-31 A/B/C 修复了此映射、restart `base_lrs`、optimizer 3→4组的 moments 处理及 horizon 检查。因此 **A首点评测的任何改善都不能归为仅仅多训5k**：真正变化还包括参数组LR、生效方式和 cosine restart。

### 2.2 Writer槽位与阶段权重的静默失效曾在预检中出现

[原记录](../../1030p125.md) 披露：`num_writers` 未进入模型构造时，通用 Writer 10–46槽折到 null，`valid=False` 使条件注入梯度为零；另外早期跨阶段 `source=None` 导致权重迁移错误。后续作了修复。不能由 loss下降推出 Writer表确已训练；下一轮必须反查checkpoint里的 Writer行数、有效 ID、EMA、真实参数变化与条件反事实 `ΔL`。

### 2.3 P75命名中的架构身份冲突

`1030p125.md` 一处称 P75 为矩阵S5，却在 ``2.5.5` 明确写 P0 spec 为 `R2(LocalCNN)×I2(CrossAttention)`，并另提 `I3`。而矩阵 [1028架构表](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md) 中 S5 是 **LocalAttention×CrossAttention**。**不按“P75=S5”做同架构比较**；以 actual checkpoint config/模型权重为准，并把不一致记入实验身份合同。

## 3. A/B/C：已经实际启动，先收口

| Arm | 起点 | 新步数 | 真实内部 counter | LR（backbone / style_repr / style_inj / writer_table） |
|---|---|---:|---|---|
| **A** | P3A内部20k（外部累计75k） | 25k | 20k→45k | `1e-5 / 5e-5 / 5e-5 / 2.5e-5` |
| **B** | P0@50k | 25k | 50k→75k | 同A |
| **C1** | P0@50k | 5k | 50k→55k | `2e-6 / 5e-5 / 5e-5 / 1e-4` |
| **C2** | C1 | 20k | 55k→75k | 同A |

均在4090、Top10-only、drop_writer约.1，真实数据/weight来源由 `assets/abc_20261031/plan.json` 指定。**A vs B/C的目标域曝光不同，B vs C更接近 Writer-first 学习率日程对照。**

首个 A内部25k（新增5k）在 val100 上 **SSIM .5914 / LPIPS .3478 / IoU .2829**，较旧 P3A **.5857/.3528/.2725** 全面改善。这是一个**新配方联合干预的正结果**，不是长期未见字或者书家改进已证。A剩余35k/45k、B@75k、C2@75k在当前GitHub快照没有结果；不得填入预计值。

### A/B/C回答的真正问题

- **A**：随着目标域曝光继续增加，内容/风格是否持续改善，还是只在LR重启后短暂改善？
- **B**：跳过P1的整个“直接Top10＋新LR”流程是否有实际优势？即使赢了，也不能只归因为“P1有害”，因为曝光和LR同时变。
- **C vs B**：同P0起点、同25k目标域训练，用5k Writer-first→20k联合，能否提高独立 E-G/Writer-swap 且保持 OCR？这是本轮最接近干净单变量的比较。

早期 [2026-10-25 P1/P2](../../ANALYSIS/p1p2_writer_aware_20261025.md) 曾在**另一个P0(kxl-100k池75k)** 上得到：20k目标预算直FT SSIM **.5800** vs混合+专项 **.5864**；30k预算直FT **.5829** vs混合+专项 **.5908**。因此不能由新P75一条链否定早期混合正结果；两者架构、预训练点与真实LR并不相同。

## 4. Decoder独立配方：来自J0–J3的强信号

Deep-Sym4@85k共同起点，额外5k、187固定面板：
- J0 FM/冻结Decoder：SSIM **.6149**、LPIPS **.3220**、InkIoU **.3437**；
- J1 冻结Decoder+图像损失：**.6149/.3222/.3428**，基本无收益；
- J2 解冻Decoder+图像Loss+GT重建：**.6198/.3150/.3447**；
- J3 解冻Decoder+图像Loss（无GT重建）：**.6193/.3147/.3433**。

J2−J0 SSIM **+.0049**，配对95% CI **[+.0039,+.0060]**；Writer Micro Top1 J0 **64.17%** vs J2 **63.64%**，不支持“Decoder解冻改善书家”。J2和J3几乎相同，也不支持GT重建锚在5k必不可少。

下一轮 A/B/C完成后，**从同一最佳成熟点K**分叉 D0普通FM+5k vs D1同样FM+可训Decoder/弱图像Loss+5k，并保存配套EMA。再用**同一 latent × 两种Decoder**做交叉解码：检验画质提升是否主要来自Decoder而非U-Net分布、是否损伤OCR/连通、是否造成E-G“风格变好”的虚假信号。D1的结果先独立验收，不同时增加E-G/Pair/复杂条件Decoder。

## 5. 怎样验证预训练：不要从A/B/C猜净迁移

A/B/C **都依赖P0预训练**，没有 scratch 对照，因此无法检验“P0是否必需”。下一阶段同时报告：
1. **目标样本效率**：相同目标Writer曝光 `k`，比较 `P0+FT_k` 与 `scratch Top10_k`，明确前者享受额外通用训练算力；
2. **算力净效益**：同总更新或更严格同FLOPs，比较 `P0 50k+FT25k` 与 `scratch75k`；
3. **开集净效益**：按 `Unseen-in-Top10` 和 `Unseen-in-All-Stages` 分开统计，绝不把P0见过的字称“全部阶段未见”。

正式比较冻结模型/EMA、训练数据及 Writer remap、STD字形字体、sampling NFE、OCR和 E-G 评测器、来源留出和随机seed。先完成 A/B/C 与 D0/D1，再谈扩充P0预训练。

详细推导：[math/04](../math/04_TRANSFER_EXPOSURE_AND_STATISTICS.md)；具体操作：[下一阶段决策](05_NEXT_EXPERIMENTS.md)。
