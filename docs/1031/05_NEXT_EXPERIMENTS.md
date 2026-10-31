# 下一阶段：以最少训练回答最重要的五个问题（4090执行版）

> **排序原则**：现有 A/B/C 在4090串行推进；**先验收已投入的训练，再增加实验分叉**。本页是建议的**实验设计/决策门**，没有在当前请求中实际启动GPU或更改运行计划。状态以 [Current.md](../../Current.md) 和原 `assets/abc_20261031/plan.json` 为准，勿依据本页推断任务已完成。

## 0. 我们真正需要区分的原因

当前失败表现有四类，处理顺序不能倒过来：
- **F0 测量错误**：CSV顺序/类别映射/LPIPS骨干/EMA/实际LR；这类错误可以把正确模型评成0%。
- **F1 内容覆盖/结构**：未见字、复杂字、部件与连通，OCR/InkIoU/人工读图不一致。
- **F2 Writer 条件没学足**：目标Writer有效曝光少、错误槽位/注入短路、Writer条件响应小或不稳定。
- **F3 Decoder 画质瓶颈**：latent已包含目标结构，但细笔、飞白、边缘在解码时受到损伤。

不能看SSIM低就优先训更大主干；先定位失败属于哪一类。

## 1. P0：CPU先完成三项可复现审计（无需抢训练GPU）

| 任务 | 输入 | 输出 | 失败即停止什么 |
|---|---|---|---|
| Q1 协议/Seen审计 | `validation_val100.csv`、各阶段训练CSV、所有增强映射 | `protocol_identity.json`、逐字 `seen_top10/seen_all/source`、有效n | 停止把 val100 称“全未见”或“全已见” |
| Q2 ckpt合同审计 | A/B/C各阶段 checkpoint/config/optimizer、`common.py` | `run_identity.json`：真实LR、EMA、Writer table行数/ID、parent、hash | 停止“P75与矩阵S5同架构”的未经核定归因 |
| Q3 OCR强JOIN重评 | 各eval `eval_stdskel_batch.csv` 与 `g{i}/gt{i}`，OCR权重/vocab | 显式按idx JOIN、GT sanity、错误列表、evaluator_version | 发现索引缺失/重复即不发该panel排名 |

**重点解决的两个当前冲突**：① [eval_style_v2](../1030/eval_style_v2/FINAL_REPORT.md) 写val100中48 seen/50 unseen、另一 [P75报告](../../1030p125.md) 写100全seen；② P75名义S5与 P0 配方R2×I2，以及 `I3` 说明不一致。不要选某一报告覆盖另一报告，而是输出逐字符和resolved config的证据。

## 2. P0：A/B/C完整收口后，先做一次同面板排序

**现有计划**：[A/B/C代码](../../tools/p125/build_abc.py)，训练最多仍在4090运行；此任务**不增启动步骤**。

| arm | 何时对比 | 关键问题 | 若结果出现…… |
|---|---|---|---|
| A内部25/35/45k | 纵向，严格相同val100和h187×3seeds | 目标域延长是否持续提升？ | 若OCR/IoU上升、Writer不退，保留最长有效点；若只有GT-SSIM升而独立OCR/Writer停滞，进入协议/数据诊断 |
| B@75k | 与P75末端（多混杂），并与C同口径 | 跳过P1的“整体配方”怎样？ | B赢也不能声称P1混合必有害 |
| C2@75k | **优先与B**（同P0起点、同25k目标训练） | Writer-first优化是否有效？ | E-G提升但OCR/结构下降时不是净改进；先找Pareto边界 |

**公开结果格式**（每arm每协议一行）：SSIM(impl)、LPIPS(Alex/VGG)、InkIoU、OCR有效n与GT ceiling、E-G raw/macro AUC与GT ceiling、未见字分层、Writer-swap（若具备）、单字汇总后的paired bootstrap CI；附原生protocol_id。

建议将 **OCR Top1和Writer宏AUC**预先定为两条主要方向，LPIPS/InkIoU为画质与伤害保护；不要把二者任意归一成一个总分。单seed的SSIM±.002不能直接宣布微小新损失晋级。

**验收门建议（设计阈值，不是既有实验规律）**：为避免“微小提升后另轴灾难”，至少要求同panel的OCR与E-G均无明显退化；若采用等效性界限，预注册OCR −2pp、Writer宏AUC −2pp作为探索性容忍界，必须同时给字符簇CI；CI含负界不得称证明无害。只有差值和置信区间能真正决定晋级，不使用小样本选图分数作硬门。

## 3. P1：只读Writer的时间轴审计，成本低于再训练一个50k

**目的**：确定Writer条件真正在生成速度场的哪个时段起作用、是否真使用了正确Writer，而不是单纯因输出图更像某个Writer统计特征。

输入 S0/S4/S5@已有EMA及当前 A/B/C 最佳 checkpoint（只评能准确加载者）；至少在 [容量探针](../../exp.md) 先前使用的 `t={.10,.25,.40,.60,.80,.95}` 上操作。

为每张GT latent `z_0`、固定独立高斯 `ε` 构造 **同一个** `z_t=(1−t)z_0+tε`；其余STD/Script不变，分别输入 `w_true`、`w_wrong`、`w_null`，测：
1. `ΔL_wrong(t)=MSE(v(z_t,w_wrong), ε−z_0)−MSE(v(z_t,w_true), ε−z_0)`；
2. `ΔL_null(t)` 同法；
3. `D(t)=‖v(z_t,w_true)−v(z_t,w_wrong)‖`（统一按latent维数归一）；
4. 分组：seen/unseen字符、书家/书体、作品来源、复杂度。

**必做负控制**：在相同 Writer 与输入下重复推理应该逐位同值；洗牌writer标签后 `ΔL` 应接近0或下降，错误Writer在同书体且训练支持范围内抽取；不允许为不同Writer重新采样 `ε` 或新 `z_t` 后说“干预Writer”。

**判据**：
- 如果正确/错误Writer在多数t上几乎等价（并且swap响应弱），优先修条件注入/Writer训练曝光，而非增加主干宽度；
- 如果中间t表现出强Writer边际风险但未见字OCR受损，怀疑 Writer/字交互和训练数据分布，进一步按字符归因；
- 如果终点 `t≈1` 有响应而中间弱，考虑当前网络可能只学到Writer条件均值而没有复杂分布差异（**假说**，见数学 [FM端点推导](../math/01_CONDITIONAL_FLOW_AND_TIME.md)）。

这是只读checkpoint的前向风险诊断，**不是新的训练任务**。

## 4. P1：Writer-swap 的共字同噪声反事实面板

对共享字符/同书体/不同Writer配成真实目标差异，固定同一STD与噪声，仅替换Writer ID。至少预注册下列三项：
- **方向**：E-G特征的生成Writer差与真实 Writer差向量余弦（真实差必须同一字或配对估计并给置信区间）；
- **幅度**：生成/真实差向量的范数比；
- **内容稳定**：两Writer生成结果都OCR为目标Unicode的比例，配合人工审核断笔/草化合法性。

同时做：
- `w_a→w_b→w_a`条件重复确认可复现；
- null/随机错误Writer负控制；
- 严格未见字子集与 seen 子集分开；
- 使用同一真实Writer参考池，排除目标字符，报告 GT pair AUC。

此前 J关系几何方向约**0.85**、幅度比**0.66–0.75**，只能作为**这组旧协议下的参考**，不能当所有模型应达到的理论目标。S组100字面板多槽位不完整，必须通过真正共字swap才能检验关系几何。到这一步仍有Writer优势且OCR稳，再考虑多层Writer新架构；否则不要再开 S5++。

## 5. P2：Decoder D0/D1单变量对照（新训练）

从A/B/C中按预注册C/W/Q选择的同一个ckpt `K` fork：
- D0：K后再FM 5k，Decoder冻结；
- D1：K后FM 5k + 原Decoder部分层可训 + 弱图像Loss，保留其自身Decoder权重与EMA；
- 数据、真实LR组、采样seed、更新步与 eval 全部固定。

**必须进行同latent交叉解码**：同一个生成 latent输入原/新Decoder，测画质、OCR、InkIoU、笔宽/飞白、E-G和失败局部。只有固定latent对照中的画质收益，才可以归因到Decoder本身；若改变了生成latent则属于端到端联合收益。J2已证明一轮+0.0049 SSIM、约−0.007 LPIPS的匹配改善，尚未证明在新的K上仍然有效。[J对照](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)。

如果 D1 的独立 Writer不提升，不继续给Decoder加Writer条件门控；先确认Decoder目标就是画质，而不是“风格分布纠偏”。

## 6. P3：后续真正预训练研究的最小矩阵（暂不执行）

| Arm | 计算预算 | 用途 |
|---|---|---|
| Scratch目标域 | 75k与预训练总预算对齐 | 直接测试总算力净收益 |
| P0_50k + Top10_25k | 同75k总预算 | 迁移路线 |
| Scratch目标域25k | 同Top10目标监督量 | 迁移样本效率（预训练多花50k算力） |
| P0_50k + 混合P1 + Top10足量 | 与上一行分开 | 单独检验混合阶段是否有价值 |

所有arm必须同实际架构和Writer ID定义，报告GT/STD分布、对所有阶段的严格OOV字符和目标域Writer匹配。A/B/C本身不能回答预训练净收益，禁止借它们跳过对照。

## 7. 终止条件、数据资产、阅读顺序

**不要自动扩大下一实验矩阵**。优先补齐当前GPU已投入实验的逐图证据；如果评测合同失败，就先修评测再谈架构；如果 Writer时间轴对正确条件不敏感，就先找有效训练曝光/真实参数组；只有这两项都通过后，才讨论复杂损失/新网络。

每次运行产物最少为 `run_identity.json`、`eval_manifest.csv`、`metrics_per_image.csv`、`metrics_summary.json`、`writer_swap.csv`（若做）、`20_selected.json`和按同idx的poster。不要删除checkpoint唯一副本或静默用新VAE复用旧权重。只搬运20张图和小CSV，不主动搬运GB级大模型。

配套理论章节：[条件FM与时间](../math/01_CONDITIONAL_FLOW_AND_TIME.md)、[Writer可辨识性](../math/02_WRITER_INFORMATION_AND_IDENTIFIABILITY.md)、[VAE/辅助损失](../math/03_LATENT_DECODER_AND_OBJECTIVES.md)、[迁移/统计](../math/04_TRANSFER_EXPOSURE_AND_STATISTICS.md)、[预测清单](../math/05_PREDICTIONS_AND_FALSIFICATION.md)。
