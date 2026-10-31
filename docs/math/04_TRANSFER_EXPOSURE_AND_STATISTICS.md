# 预训练迁移、后训练曝光与实验统计：如何获得能信任的结论

> 数学任务：区别“迁移让目标任务训练更快”“总算力更值得”“单个Writer表现好”“总体GAN/图像指标更好”四件事。针对 [P75链](../../1030p125.md)、[A/B/C](../1031/03_TRAINING_TRANSFER_AND_DECODER.md)、[评价勘误](../1031/04_EVALUATION_AND_ERRATA.md)。

## 1. 定义目标风险和两种不同的预训练效应

定义目标分布 `P_T`（目标Top10 Writer、指定数据来源和严格未见划分），训练后模型 `\theta` 的目标风险为

\[
R_T(\theta)=\mathbb E_{(X,G,S,W)\sim P_T}
[\ell(\theta;X,G,S,W)].
\]

`\ell` 可以是训练用 FM MSE，也可以是**固定的** OCR错误率、Writer pair损失或图像重建损失；对不同 `\ell` 的风险不可直接相加。

对有同架构参数族 `\mathcal F` 的模型，形式上：

\[
R_T(\theta)-R_T^*
=
\underbrace{R_T(\theta)-\inf_{f\in\mathcal F}R_T(f)}_{\text{该模型族内剩余差距}}
+\underbrace{\inf_{f\in\mathcal F}R_T(f)-R_T^*}_{\text{表示/模型族差距}}.
\]

这是**按定义成立**的恒等式；第一项还含训练数据估计、优化、LR调度等多重因素，不能未经实验再拆成几个非负数并赋予固定比例。“数据不足”“容量不够”“优化不对”是待区分的解释，不是已有数值分解。

### 两个必须分开报告的迁移估计量

令 `A(P_0,T)` 表示通用预训练 `P_0` 后在目标域再训练 `T` 步，`B(T)` 表示完全从头在目标域训练 `T` 步。

**样本效率问题（不等总算力）**：

\[
\Delta_{\rm sample}(T)=R_T(B(T))-R_T(A(P_0,T)).
\]

若差值为正，预训练提供了目标监督样本效率收益，但 `A` 额外用了 `P_0` 的算力。

**总算力问题**：

\[
\Delta_{\rm compute}(P_0,T)=
R_T(B(P_0+T))-R_T(A(P_0,T)).
\]

这才回答：在相同总体训练预算下，把前 `P_0` 步花在通用数据上是否值得。若两个架构或批量不同，应改为匹配**训练FLOPs/相同显存/实际时间**而不只数steps。

它们可能符号相反：通用预训练帮助低样本适配，却因目标数据质量高而**不如把相同算力直接投入目标Writer**。没有这两个对照，就不能一句话裁定“预训练有效/无效”。

## 2. 为什么有效 Writer 曝光比累计步数更能解释 P75

设阶段 `k` 总训练 `T_k` 个等batch更新，目标Top10被采到的概率 `p_k`，其Writer条件未被drop的概率 `1-d_k`。定义

\[
E_{\rm image}=\sum_kT_k p_k,\qquad
E_{\rm writer}=\sum_kT_k p_k(1-d_k).
\]

这两者是**同batch归一的训练样本量期望**，不是模型参数只更新了那么多步，也未包含同一张样本重复曝光、重加权和信息重复等问题。

P75已确认：P0 50k无目标Top10 Writer曝光；P1 5k、`p=.3,d=.25`；P2 15k和P3 5k、`p=1,d=.1`。所以

\[
E_{\rm image}\approx1.5+15+5=\boxed{21.5k},
\qquad
E_{\rm writer}\approx1.125+13.5+4.5=\boxed{19.125k}.
\]

这比 S0/S5 从头50k–85k在Top10的Writer条件训练少得多，因此 P75 落后不能直接说明通用先验差。**同时还有真实Style LR事故**：旧 P75 Style组实际 `1e-4`，不是配置表里预期的较小值；A/B/C修复后不能把改进仅归于步数。

另一个数学上的风险：当 `P_0` 的Writer标签与目标Top10无关，模型可能学习到好的字符骨架，但将 Writer差异在内部分布中边缘化；这只是潜在机制，**需要正确/错误/null Writer的时间分段FM风险证明**。

## 3. 训练分布与评测分布不等同：一般化边界

对某模型损失 `\ell_\theta`，将训练和目标评测分布差写成

\[
R_T(\theta)-R_{\rm train}(\theta)
=\int \ell_\theta(x,c)\,[p_T(x,c)-p_{\rm train}(x,c)]\,\mathrm dx\,\mathrm dc,
\]

只要密度和积分有定义。此恒等式表明：若训练与测试的 Writer×Script×Character×Source频率不同，即使训练FM损失不断下降，也可能只改善**被过量采样的seen条件**。

泛化风险还受到：
- **字符覆盖**：Unicode在训练出现与否；
- **Writer条件覆盖**：同字符是否被目标Writer写过；
- **作品来源**：同书家同帖/扫描器可能被近似复现；
- **生成条件支持**：错误 Writer 置换是否产生训练中没有出现的 `(W,S,C)`组合；
- **评价器域偏移**：读光在GT上80%不等于对近形缺笔有80%检测率。

因此“Strict”必须明确是哪一层级：
- `Seen image`；
- `Unseen (W,S,C)`组合；
- `Unseen-in-Top10`字符；
- `Unseen-in-All-Stages`字符；
- `Source-held-out`书帖/采集来源。

旧187有**75/187**组合在训练中出现，不能称作187严格未见；val100有**48seen/50unseen**（E-G有效98）与“全部已见”的文档矛盾，必须由原始 CSV/训练字符并集重建，不许用猜测修正历史结果。

## 4. 正式指标应测“什么”：四目标与配对统计

模型在同一评测协议上的性能向量可写成：

\[
M(\theta)=(
A_{\rm OCR},
A_{\rm EG}^{\rm macro},
-\mathrm{LPIPS},
\mathrm{InkIoU},
A_{\rm unseen},
\operatorname{ContentStability}
).
\]

**Pareto改进**：一模型在所有预设重要指标上不弱于对照，且至少一项有可复核提升；否则称为内容/书风/画质的**权衡**。不能用任意权重把不相容单位硬加总，再以“总分更高”掩盖Writer变差或错字。

**三类身份的统计误差必须分开**：
1. 字符/Writer/Source的**抽样变异**（可用分层或层级bootstrap）；
2. 训练随机种子/优化器/模型初始化的**训练变异**（只有多个完整训练seed才能估计）；
3. OCR/E-G评价器与缓存、真实GT的**测量误差**（评估实现、vocab、GT ceiling、来源偏差）。

在187个字符×3个采样seed的面板，同字符三seed不是三个独立字符。对于逐字符函数 `d_{c,k}`：
- 先对 `k` 做平均 `\bar d_c=\frac1K\sum_k d_{c,k}`，得到字符水平效应；
- 按字符（必要时Writer×Script分层）重采样 `\bar d_c`，报告两模型的**配对差**和CI；
- 如果需要评估噪声稳定性，独立报告 `k` 内方差，而不是把 `3N` 当独立样本凭空缩小误差条。

当Writer只有10个且书体不平衡时，Writer宏指标区间和pooled AUC区间的 estimand不同：前者每Writer近等权，后者多样本Writer获得更大权重。要预先声明主目标，不能看哪个显著就选哪个。

**置信区间不跨0也不自动代表已经跨训练随机性复现**，更不意味着所有预设指标均有显著效应。多个指标/超参数同时选优，要额外用验证集冻结选择、独立Test验收。

## 5. GT“天花板”与评价器相对饱和度：不要当数学上界

对E-G二分类核验，在同一参考池、相同Writer正负比和同字符排除协议下，可作经验归一：

\[
S_{\rm EG}=
\frac{\mathrm{AUC}_{\rm gen}-0.5}
{\mathrm{AUC}_{\rm GT}-0.5},
\quad \mathrm{AUC}_{\rm GT}>0.5.
\]

例如旧S5 AUC **87.98%**、其100字面板GT AUC **92.38%**，`S≈0.896`；Deep85 AUC **85.57%**、旧187面板GT **89.76%**，`S≈0.895`。**这说明跨协议原AUC的名次不稳定**，不说明S5或Deep85已经达到某个不可越过的数学上限。若参考池、同字排除、Writer混合权重变动，连饱和度也可能不可比。

OCR同理应同时报告 `A_{\rm gen}`、`A_{\rm GT}`、字表覆盖以及在 `OCR(GT)=target` 的预先固定子集上的 `A_{\rm gen\mid GTcorrect}`。后者不能静默替代全体，否则对难字产生筛选偏差。

**SSIM同条件真迹对的均值`0.6503`不是上界**。以平方差为例，两个同分布独立GT `X,X'` 有

\[
\mathbb E\|X-X'\|^2=2\operatorname{tr}\operatorname{Cov}(X),
\]

它解释多写法会有正常差异；但 **SSIM不是MSE，不能据此推出任何固定SSIM最大值**。

## 6. 研究结果的来源审计和可重复性

一个可信实验点至少记录：
`code_hash`、`model_config+architecture_resolved`、`checkpoint EMA/raw及真实steps`、`optimizer param_groups/base_lrs`、`VAE/Decoder/STD/font hash`、`train/eval CSV hash`、`Writer remap和source id`、`sample idx/char/g/gt image hash`、`sampler/NFE/CFG/SDE/noise seed`、`OCR/E-G evaluator_version` 与指标定义。

真实已发生的反例：
- 评测按 `g0,g1,g10...`字典序配错，OCR错报0%；
- PIL把(height,width)当(width,height)，OCR输入被拉扁；
- `S0@85k`同批PNG两种SSIM实现报告**0.5963 vs0.6081**；
- MoYi `t{i}.png`骨架误当生成图；
- v68 Heun50≈100NFE vs MoYi Euler50 NFE50；
- P75 Style LR配置字段无效，实际变成`1e-4`；
- P0曾出现Writer合法槽全部折叠到null，条件梯度为0的预检失效。

**若这些合同不冻结，任何统计显著都可能只是在评价错误。** 下次实验先CPU核对身份，GPU只做确实必要的采样和前向诊断；所有新结果另开protocol_id，不覆盖历史曲线。

对应执行：[评测勘误](../1031/04_EVALUATION_AND_ERRATA.md)、[4090计划](../1031/05_NEXT_EXPERIMENTS.md)。
