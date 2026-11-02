# 4090 下一轮诊断与后训练执行方案（2026-10-31）

> **定位**：可直接交给训练执行者的实验合同。以 `main@bbbfc4230f38` 及 `assets/abc_20261031/plan.json` 为基线；**本文是计划，不表示已经执行**。主机只用 **4090**；不操作 48、不搬运 GB 权重；不开展新一轮通用预训练、不立刻重训 VAE/DiC。
>
> 配套：[1031实验总览](README.md)、[证据矩阵](01_EVIDENCE_MATRIX.md)、[原始CSV勘误](06_RAW_DATA_RECONCILIATION.md)、[FM理论](../math/01_CONDITIONAL_FLOW_AND_TIME.md)、[Writer可辨识性](../math/02_WRITER_INFORMATION_AND_IDENTIFIABILITY.md)、[可证伪命题](../math/05_PREDICTIONS_AND_FALSIFICATION.md)。
>
> **科学目标**：定量判断剩余误差由 (a) Writer曝光/数据采样、(b) Writer条件注入/时间作用、(c) 内容/OOV、(d) Decoder/VAE，还是 (e) 主干容量造成；只对确诊的瓶颈新增训练。重要：当前S/P75主干是 **32×32 latent CNN U-Net**，部分旧线才以“DiC”命名，禁止在报告中把其称为同一DiC架构。

## 0. 一句话决策与预算

**执行顺序**：P0训练与评价身份审计 → **完成已在跑的 A/B/C** → 只读 Writer时间轴/注入诊断 → 同噪声Writer-swap + GT→VAE→GT诊断 → **按照诊断最多启动一个主要新训练主题**（数据均衡、时间采样、注入位置或Decoder），每主题保留同起点FM对照。

| 优先级 | 工作簇 | 额外训练 | 任务结果 |
|---|---|---:|---|
| P0 | 合同与样本索引修复 | **0** | 协议、真实step/LR、Writer remap、S0/S5/P75结构身份可追溯 |
| P0 | A/B/C既有队列 | 已计划 **75k更新**，不重复新开 | 内容/Writer/画质三轴的真实后训练Pareto选择 |
| P1 | T：Writer正错null随t风险 + 逐层干预 | **0** | 证实条件是否真被网络使用，确定时间与空间作用位置 |
| P1 | S：同字符同噪声 Writer-swap | **0** | Writer改变方向、幅度、内容保真是否正确 |
| P1 | V：真实GT-VAE重建与生成缺陷分解 | **0** | 区分Encoder/Decoder瓶颈和生成latent瓶颈 |
| P2 | E：数据/时间采样**二选一**新配方 | **同起点 10k×2** | 无附加Loss的可归因改善 |
| P2 | I：局部注入位置（仅T/S指向时） | **同起点 10k×2** | 验证改变Writer作用的尺度是否改善 |
| P2 | D：Decoder图像域后训练（仅V显示收益时） | **同起点 5k×2或×3** | 复现J2/J3而不过度损伤OCR/Writer |
| P3 | 新VAE/大主干/通用预训练 | **本轮不启动** | 等证据明确再设计 |

**不要把所有P2分支同时排满**。P1诊断后选一个最有可能解瓶颈的P2主题；若有两种独立瓶颈，各自分开、各保留同起点FM对照。

## 1. P0：训练/评测环境的硬合同（CPU先做）

### 1.1 优先锁定实际checkpoint身份

复核 `P0@50000`、`P3A@20000`、`A内部25000` 与后续 A/B/C 完成点：
- checkpoint完整SHA、架构实际类/参数、style representation的 R0–R4 / injection的 I0–I3、Writer table和null行；
- EMA/raw、optimizer参数组、`initial_lr`、`lr`、`base_lrs`、restart origin/horizon、训练步数、RNG/数据采样；
- STD-soft字库/TTF版本、VAE scaling/shift、Decoder与EMA、训练样本宇宙；
- ***加载必须严格检查*** weights key集合，不允许`strict=False`默默吞主干参数；新增注入模块仅允许显式白名单新键，并要求零初始化时与原模型输出数值等价；
- 特别核对 `P75` 名称“S5”与 P0 spec的 `R2(LocalCNN)×I2(CrossAttn)` 冲突；不以run名替代架构ID。

**历史勘误**：P75 Style两组实际 `1e-4`（旧字段名bug），A/B/C新阶段的LR才按spec生效。A的增益包含重启+LR更正+新增训练，不等于纯延长步数。C1与C2分段各有restart，**B vs C2 比较的是 Writer-first学习率+两阶段restart联合配方**，不是仅Writer学习率单变量。

### 1.2 必须修好指标的身份

- **每次评测都以本次`eval_stdskel_batch.csv`为权威**，由`(panel_id,idx,writer,script,Unicode)`精确join `g{idx}`/`gt{idx}`；检查重复/缺失/异体映射并计算文件hash，不允许`zip`位置配对、字典序文件名或外部CSV回退。
- 对`p125_metrics_full_4090.csv`执行 schema fail：当前**15列表头、后5行18列、S0两seed重复**。修字段名/真实主键再入统一表，不覆盖原文件。
- `val100` seen/unseen身份存在文档冲突（“全seen” vs“48seen/50unseen”）；冻结训练字符并集，给每行增加`seen_writer_combo`、`seen_top10_char`、`seen_all_stages_char`、`source_heldout`。
- 同时保存 `ssim_impl`、`lpips_backbone`、`solver/NFE`、`CFG`、`noise_seed`、`GT_OCR_ceiling`、`EG_GT_ceiling`；不得合并`S0@85k SSIM .5963`与另实现`.6081`。

**本关交付**：`contract/run_identity.json`、`contract/eval_identity.csv`、`contract/metric_implementation.json`、`contract/seen_flags.csv`、`contract/raw_csv_schema_audit.json`。若身份未通过，本次严禁发布任何“第一名”。

## 2. P0：收口4090已经在跑的A/B/C，先看后训练有效性

不修改正在运行队列，不重启已完成节点。来源：[实际生成脚本](../../tools/p125/build_abc.py)。

| Arm | checkpoint/起点 | 新训练 | 实际内部停止点 | LR backbone / style_repr / style_inj / writer_table |
|---|---|---:|---|---|
| **A** 延长 | `P3A@20000`（外部累计75k） | +25k | `25000/35000/45000`（最终累计100k） | `1e-5 / 5e-5 / 5e-5 / 2.5e-5` |
| **B** 直达Top10 | `P0@50000` | +25k | `75000` | 同A |
| **C1** Writer-first | `P0@50000` | +5k | `55000` | `2e-6 / 5e-5 / 5e-5 / 1e-4` |
| **C2** 联合 | `C1@55000` | +20k | `75000` | 同A |

训练数据 `assets/p125_20261030/stage_p2.csv` (Top10-only)，drop_writer约0.1，单一FM主Loss，restart warmup200。现有 A 首点 `val100` SSIM **.5857→.5914**、LPIPS **.3528→.3478**、IoU **.2725→.2829**，只是**新整体配方的正结果**，尚缺同协议Writer/OCR判定。

**统一评测**：
1. 完全一致的 `heldout187×3 seeds`（按仓库`S_187_seed{0,1,2}.csv`），保留 val100历史协议；把所有模型重评在**同一份真正的样本manifest**；
2. **内容**：读光 OCR Top1/Top5、target margin、GT Top1、有效n/词表覆盖，按字符复杂度、seen层级分组；
3. **风格**：冻结 E-G同书体异字 Writer宏AUC、EER/mAP和真实GT参照；因P3B曾用E-G监督，它必须额外走独立的swap/人工验收；
4. **画质/结构**：同版本SSIM、LPIPS-VGG（或Alex另列）、InkIoU、墨量、宽度/GT、Frag、容差Skeleton；输出RGB图和固定20字poster；
5. **统计**：同字符先跨seed平均，再做按字符/Writer分层的paired bootstrap（95%CI）；三个seed不是三倍独立字符。

**判读**：
- A@25k→35k→45k同协议持续改善且Writer/OCR不退：继续Top10适配有价值；否则在第一个**三轴Pareto明显最优**的点停。
- B vs C2：检验**整套Writer-first+额外restart配方**；若C胜后才补“C0两阶段restart但普通LR”的精确控制（需要额外5k+20k），否则不能只归因到Writer-first。
- A vs B：从不同训练历史来，目标Writer曝光不同；只用于比较**实际部署路线**，不能用于证明P1混合阶段单独有害或通用预训练有/无效。

**选取后续共同起点K**：先筛除明显错字/异常墨量和身份错误，再在同面板内容OCR、Writer宏AUC、LPIPS/IoU的Pareto前沿选一个；若两模型互有胜负保留两个候选，但**新训练只从一个K开始**。建议预注册主要目标`OCR_Top1`和`EG_macroAUC`，其他为保护指标，不能事后按SSIM挑赢家。

## 3. P1诊断T：Writer到底在FM哪一个t和哪一层生效

### 3.1 全程不训练，只需GT latent的只读前向

候选：至少`K`、`P3A`，若4090已有且可严格加载，附加`S0@50k/S4@50k/S5@50k`作架构横向诊断；不用48端唯一权重。样本以`heldout187`为主，额外选train_half，作者/书体平衡；固定每张GT潜变量`z_0`、同一个高斯噪声`ε`及对应STD和Script。

取 `t={0.10,0.25,0.40,0.60,0.80,0.95}`；每样本**2个噪声seed**；每个`z_t=(1-t)z_0+tε`用**完全同一z_t**查询：
- `w_true`；
- `w_wrong`（在当前Script有数据、同字符共用时优先；先1位固定对照，需要时再增加2位）；
- `w_null`。

计算：

\[
L_t^w=\frac1d\|v_\theta(z_t,t,g,s,w)-(\epsilon-z_0)\|^2,\quad
\Delta L_t^{wrong}=L_t^{wrong}-L_t^{true},\quad
\Delta L_t^{null}=L_t^{null}-L_t^{true}.
\]

另计算 `\|\Delta v\|/\sqrt d`、`\|\Delta v\|/\|v\|`（分母设eps保护）、真实Writer与错误Writer的各尺度特征/梯度响应；报告分t均值、按Writer与字符簇CI。

**绝对必要的正确性限制**：
- 同噪声**不等于同输入**：不同真迹`z_a,z_b`在中间t产生不同`z_t`，两者速度差不能被称为纯Writer干预。
- `t=0`最优Writer端点差必为0，`t=1`只含条件均值差；不能因为两端响应弱，就推断Writer没用。
- `wrong Writer`如果是根本未支持的`(char,script,writer)`组合，风险差可能体现OOD；主表优先使用跨Writer共同字/共同Script，其他作单独泛化面板。
- 正确/错误对比可能受训练作者/扫描作品来源共线影响，**FM风险差本身不证明真实书法身份**，必须和S诊断结合。

### 3.2 注入层消融（同模型、无需新权重）

在现有 MatrixUNet：内容stem+encoder使用STD；Writer独立分支实际作用在**Middle与Decoder不同分辨率**（核对模型实际配置）。对`K`以及可加载的S4/S5：
- 记录每层 `delta_rms/base_rms` 与空间方差、cross-attn entropy/variance；
- 在保持其余结构不变的情况下，依次临时关闭 **Middle(8×8)、Decoder(16×16)、Decoder(32×32)** 中一个Writer residual；
- 在 `t=.25/.5/.8` 上的同一`z_t`测 `L_t^{true}`、`ΔL_t^{wrong}`，再对典型字符做少量完整采样，核对OCR和E-G。

**这叫推理期机制ablation，不是训练因果消融。** 关闭已训练模块可能触发失配，因此只有“哪个尺度影响最大”的诊断意义；不能直接据此宣布删除该层训练必然更好。

### 3.3 符合数据的判读分支

| 观察 | 更可能的问题 | 对应动作 |
|---|---|---|
| `ΔL_wrong≈0` 且swap幅度弱，Writer参数/梯度正常 | Writer条件有效学习不足，或数据Writer差被平均 | P2优先 **数据均衡/训练时间采样** |
| `ΔL_wrong>0`，但生成swap方向错 | 条件确有影响，但未按真实书家关系输出 | 查Writer remap/来源/注意力/关系学习，不急加辅助Loss |
| 中间t条件有效，低分辨率Writer分支消融对结构影响大 | Writer需要正确的低尺度结构通路 | P2可试**指定尺度注入** |
| 只在train_half有Writer gap、heldout弱 | 泛化或来源捷径 | 数据/来源与长尾采样优先，暂不改架构 |
| 正确Writer效果强、OCR仍差，GT重建良好 | STD内容或未见字符分布问题 | 内容/字频数据分析，避免再加style强度 |
| Writer与内容均有效，但GT重建已丢细笔 | Decoder/VAE存在局部质量瓶颈 | P2优先 **D Decoder试验** |

## 4. P1诊断S：同字、同噪声的Writer-swap（真正生成，0训练）

**面板**：从此前1609共有字中选**30个冻结字符**，分简单/复杂、训练组合见过/未见、楷/行/隶；实际只有同Writer×Script可比的组合才入主表。尝试每字符**10个已知Writer×3 seeds**，上限约900张生成图；若脚本或Writer覆盖不齐，输出有效组合并下调n，**不制造不存在的GT**。选择一部分样本作独立、全部训练阶段均未见的严格OOV（若有合格可用字），并标明这部分没有GT同字对时只能做OCR，不做真实同字风格配对。

固定每个 `(char,g,s,noise_seed)`，只改变Writer；不要改变NFE、CFG、VAE/Decoder。参考书家原型从**同书体且排除该字符和同作品来源的真迹**估计（例如现有E-G K64），必要时辅以同字符真实配对，但稀疏时用跨字原型并带CI，避免强行宣称“同字真迹均值”。

必须报告：
1. **Content stability**：每个swap组所有输出OCR都为目标字符的比例；记录错字/缺笔/粘连的人工盲审。
2. **Writer fidelity**：原有E-G核验AUC、宏AUC，错误Writer/null负控制，参考GT上界。
3. **Direction**：在同E-G特征中 `\cos(\phi(\hat X_w)-\phi(\hat X_{w'}),\mu_w-\mu_{w'})`。
4. **Amplitude**：生成Writer差范数/真实参考Writer差范数，真实差接近0的配对不计算不稳定比值；按Writer对/字符重采样CI。
5. **Per-char failures**：按复杂度/seen/source切片，不把同一字的45对 Writer差异视为45个独立字符。

本组主要判定：有没有“Writer确实使图像变化，但方向与真实Writer不一致”；有没有“方向正确但幅度压缩”；有没有“书家更像而字写错”。只报告宏AUC不足以回答以上三个问题。

## 5. P1诊断V：VAE/Encoder/Decoder到底损失了多少质量

从4090已有真实 GT 中选**300张**（尽量覆盖10 Writer、各书体、复杂字及细笔/飞白，若可用数量不足则显示真实n）。使用每个checkpoint原配的Encoder/Decoder和scale/shift，以训练一致的posterior模式进行：

\[
X_{\rm GT}\rightarrow E_\eta(X_{\rm GT})=z_{\rm GT}\rightarrow D_\psi(z_{\rm GT})=X_{\rm rec}.
\]

对原图与重建图比较：
- OCR Top1/GT Top1差值、错字集合，近形字/残笔分类；
- E-G真实Writer AUC/原型方向变化，避免“纹理变少导致AUC低”与“真风格丢失”混同；
- InkIoU、容差SkelF1、Frag、Ink amount、宽度、细线connected components及细节高频；
- 若已有两版Decoder，**固定同一个latent**交叉解码，不混用错误scale/shift。

**判断**：
- 若 GT VAE重建的字符和拓扑与GT接近、而生成图明显更差，**不要重训VAE**，优先模型条件/训练数据；
- 若连 GT 重建也有系统性缺笔/粘连/飞白崩坏，再做J2式Decoder对照，先不从头训VAE；
- 只有图像域Decoder微调无法纠正且错误跨大量GT/Writer/Script普遍存在，才立项新VAE。

**注意**：重建OCR“没有下降”不保证微细笔锋没丢；必须看重建质量而非单一OCR。

## 6. P1数据诊断：决定训练应怎样采样

对Top10训练语料与同协议生成错误，按 **Writer×Script×Character频次、Unicode复杂度/笔画数、作品source** 建立表：
- 样本数/可用共享字覆盖、每Writer有效条件曝光、Writer×Script贫乏格与主导书体；
- 逐字符 `L_FM(t)`、OCR错误率、InkIoU、E-G身份失误、真实GT重建错误；
- 检查错误是否集中在**低频字、复杂字、特定Writer/Script或来源**；对于跨阶段所谓“未见”，区分是否只在Top10未见而通用P0见过；
- 特别筛查稀疏 writer/script格的重采样是否只是同一件作品反复复制。

**两条决策**：若Writer低频格导致身份差，先改Writer×Script采样；若错字集中在复杂低频字且GT VAE正常，则改字符覆盖/频次；若无频次相关而出现系统形变，先查条件机制，不用简单重采样解决结构性问题。

## 7. P2训练分支：只启动和诊断一致的一组

所有新训练在 **4090** 上按序串行，共用选定ckpt `K` 和数据/EMA/RNG/optimizer规则；从K分叉，新段统一`10k`（Decoder`5k`），warmup`200`、相同cosine视界、同一模型源与optimizer处理。以下LR**仅作为复用已验证A/B/C配方的初始候选**：backbone/style_repr/style_inj/writer_table = `1e-5 / 5e-5 / 5e-5 / 2.5e-5`，必须以A/B/C完整结果和真实参数组为准，不重新拍超参。

### E：数据利用是主因 → 原采样 vs Writer/Script均衡（10k×2）

| 组 | 起点 | 唯一主变量 | 数据与Loss |
|---|---|---|---|
| **E0** | K | 原样Top10采样，基准 | 常规FM，无新损失 |
| **E1** | K | Writer×Script分组**温和均衡** | 相同样本池、FM、LR、总步与dropout |

推荐`q(ws)∝[(1-\alpha)p_{\rm empirical}(ws)+\alpha p_{\rm uniform,observed}(ws)]`，`\alpha=0.5`；再对每样本相对原采样的倍率**截断于[1/3,3]并重新归一化**，杜绝极稀格过采。每Writer Script真实可用才采，不合成无标签新图。不要同时加形态增强、额外E-G/Writer Loss。

预言：若数据暴露问题是主因，E1提升贫乏Writer/Script的**宏AUC与correct-writer ΔL**，同时OCR/墨量基本保持；若只提升训练作者/已见字符则不晋级。

### T：时间利用是主因 → 原时间采样 vs 中段加权（10k×2）

| 组 | 起点 | 唯一主变量 | FM 时间分布 |
|---|---|---|---|
| **T0** | K | 当前训练合同 | `t∼\pi_{\rm original}` |
| **T1** | K | 时间采样加权 | `t∼0.5\pi_{\rm original}+0.5U[.25,.8]` |

其余结构与Loss不动，不添加“错Writer故意高损失”的ranking目标；新采样不会增加训练步或改变GT语料，但会重新分配主干与Writer分支的时间权重。**必须同时检查FM在高噪/低噪端是否退化**，不能只看中段Writer响应增强就晋级。

只有P1显示 `t≈.25-.8` 的Writer条件风险显著且模型未学充分、或端点响应偏均值而swap幅度压缩时，才考虑T1；若数据没有这个特征，暂不跑T组。该时间窗是候选而非已证明最优，不再网格搜索多个窗。

### I：注入层次是主因 → 保留原结构 vs 指定低尺度注入（10k×2）

当T层消融证实低分辨率 Writer影响结构但当前路径作用不足时：
- **I0**：K +10k，原条件注入结构（与E0/T0可复用，若起点、随机性、训练合同完全相同）。
- **I1**：仅在 **Encoder低分辨率8×8** 增设一个小型Writer residual，复用现有Writer-ID/StyleState，不引入新参考图Encoder。输出零初始化、残差增益零或近零，开训前逐元素比对**I1未训练输出≈K原输出**；新增参数不超过主干~2%作为设计上限。
- 除新增注入外，数据、FM loss、训练步、LR等不变，新增参数的 optimizer组显式登记。若该层在当前模型没有相同shape/scale，不能借名字强行硬接，必须检查 forward shape/CFG-null/梯度。

若T层消融指向**Decoder32×32过强调制**，则不同时做I1，改为**有界降低该尺度注入增益**的单变量分支（同样10k对照）；同一轮只验证一种结构假说。

**判据**：未见字OCR不降、Writer-swap方向/幅度改善、E-G宏AUC上升；只提高training Writer CE或SSIM的小效果不算结构晋级。

### D：Decoder实际构成瓶颈 → J2/J3式图像域对照（5k×2或×3）

| 组 | 起点 | Decoder | 目标 | 作用 |
|---|---|---|---|---|
| **D0** | K | 冻结 | FM | 自然继续训练的基准 |
| **D1** | K | 冻结 | FM+与D2同量级的弱RGB loss | 隔离图像Loss本身 |
| **D2** | K | 可训（先按J3范围） | FM+与D1相同RGB loss，暂不加GT重建锚 | 隔离解冻Decoder的额外作用 |

可先做 D0+D2两组复现画质收益；若需**严格区分图像Loss和Decoder训练**，补D1。旧J2/J3的∆SSIM约+.0045~.0049、LPIPS下降约.007，是**同Deep85起点的先例**，不是本次K的预期数值。所有D输出配套保存Decoder，进行**同一个生成latent×旧/新Decoder**交叉解码与GT重建，核验OCR、E-G、宽度/连通是否被破坏。

只有D1/D2在真实GT重建、生成图和内容/Writer保护下稳定改善，才允许继续扩充图像监督；不要一开始再加Writer条件Decoder或E-G Loss。

## 8. 不做什么：避免又一次多变量不可归因实验

- **不新训VAE**，除非GT→VAE→GT确实发生系统性拓扑/笔画质量丢失，且D分支无效；
- **不盲目加宽/加深主干**。20L×384 vs 20L×256等曝光：FM MSE`.35787 vs .37087`、Writer Top1`51.87 vs41.71%`，而复杂字细骨架召回**34.20 vs34.24%**几乎相同，结构容量并非已有证据里的优先瓶颈；
- **不回12ch/SkelNet级联**。旧4ch同50k对12ch无劣势，辅助目标有梯度竞争；新骨架先做误差诊断再立项；
- **不加两张参考图Encoder或人工Writer×字形码本**。DINO K32原型Top1仅24%，严格留字的几何调制没有增益；
- **不直接用E-G给生成器增加强ranking loss**，其训练教师与评测器重合、来源混杂尚未彻底隔离；若未来需要，也须固定teacher、梯度占比受限、用独立评审和OCR防Goodhart；
- **不在当前队列外追加新的通用P0预训练**。迁移价值需要等目标曝光与等总FLOPs的两种scratch对照，A/B/C都以P0起点，不能回答它。

## 9. 统一验收、选择与停止规则

**固定评测至少两层**：历史`val100`与`heldout187×3`用于快速一致性检查；若要区分细微OCR差别，另用**足够多独立Unicode的未见字面板（优先≥300字，取决于现有可用GT/字库且不得从冻结Test调参）**。187字二分类OCR准确率的抽样波动可能达数pp，单看1–2pp差不能下结论。若没有300字可构造，应如实报告不确定性，而非重复同字seed伪造独立性。

- **主要目标预注册**：OCR Top1（全样本、GT可识别子集和严格未见层分列）与 Writer宏AUC（同书体跨字、独立真实GT ceiling）；生成艺术性最终需盲审代表性失败例。
- **保护指标**：LPIPS-VGG、InkIoU、笔宽/GT、Frag、近白异常、Writer-swap内容稳定、数据来源分层。
- **统计**：同字符多seed先聚合、paired bootstrap (≥2000 resamples)，Writer层分组并单列；不能把同一字符的10Writer×3seed当30个独立汉字，也不能以20字poster的局部SSIM称全量领先。
- **停止条件**：指标变坏且重复同协议测量成立；条件有效LR/Writer remap不对；生成blank/墨量坍缩；GT sanity不通过；未见字OCR明显退化。**“无伤害”必须由配对区间支持，不能只看均值差很小**。
- **工程设计非劣界建议**：为避免微小画质增益牺牲内容/书家，可在Val预先考虑OCR Top1和Writer宏AUC各`−2pp`的保护界；但187字样本通常不足以统计证明这么小的界，不能据无显著差异宣告满足非劣性。

**最终结论分类**：
1. **确定有效**：同协议、匹配训练对照、至少主要指标与保护指标满足预设判据，配对置信区间支持；
2. **有希望但未证明**：改善小、跨作者异质大、仅单seed或未有独立Writer/人工验证；
3. **不值得扩展**：改善只在训练/单GT SSIM/教师分数而内容/Writer泛化不升，或明显伤害；
4. **评测失效**：身份/样本/词表/配置不一致；不产生科学排名。

## 10. 每个实验最终产物（固定路径建议）

建议存于 `assets/next4090_20261031/{family}/{arm}/`（只建议命名，不覆盖现有路径）：

```text
contract/run_identity.json        # checkpoint + EMA/raw + resolved config + true LR
contract/eval_identity.csv         # idx,char,writer,script,GT/STD/hash,seen/source
diagnostics/writer_time_per_sample.csv
diagnostics/layer_ablation.csv
diagnostics/vae_reconstruction_per_image.csv
diagnostics/writer_swap_per_pair.csv
metrics/per_image.csv
metrics/summary.json
metrics/paired_bootstrap.json
gallery/selected_20.json
gallery/poster_20.png
decision.md                       # 采用/保留/淘汰，理由及下一分支
```

每次测完先写`decision.md`（对照、主要效应、CI、失败样本与解释边界），再把真正里程碑迁入[Milestones](../../Milestones.md)，全量run落[All](../../All.md)，当前簇落[Current](../../Current.md)。任何已训练checkpoint唯一副本不得自动清理。

## 11. 执行完P1后如何决定P2（最短决策树）

1. **Writer条件风险在真迹和未见字都弱，VAE重建正常** → 优先E（Writer×Script平衡），若时间曲线指出中段缺口再T；不重训Encoder/VAE。
2. **Writer风险有信号，但swap方向错或表现只在来源内** → 先修数据/Writer remap/参考评价，暂不加强Loss或大模型。
3. **Writer风险有效、swap方向对，但幅度弱；t中段和层消融指出一个尺度缺口** → T或I二选一，并保留K+同预算原FM对照。
4. **Writer/结构正常，细笔/墨迹在GT重建阶段已丢** → D，先可训Decoder；D失败后才考虑新VAE。
5. **字符结构在GT重建完好，但生成OOV错字集中低频/复杂字** → 先改Top10字符覆盖/采样，避免“提高Writer强度”破坏内容。
6. **Writer和内容都正常，仅主干FM在train/val双高且增加容量有匹配收益迹象** → 才准备更深/更宽DiC/U-Net重训消融；**本轮数据不足以直接下此结论**。

**本计划最重要的输出不是多几个ckpt，而是诊断清楚每一次错误发生在哪个子系统，让下一次新结构的改动只有一个因果解释。**
