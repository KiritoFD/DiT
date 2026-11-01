# 研究假设登记：每一条理论如何被下一次实验支持或推翻

> 定位：**不是未来工作的清单，而是可证伪的预注册草案**。理论等式、现有测量与下一步实验必须一一对应。此时不启动新训练；先完成4090现有A/B/C与评测合同修复。操作详见 [1031实验顺序](../1031/05_NEXT_EXPERIMENTS.md)。

## 1. 预测列表：把“理论正确”和“模型真的做到”分开

| ID | 数学或机制命题 | 当前仓库证据 | 必做试验（不新增训练时优先） | 若未观察到什么，应修改什么 |
|---|---|---|---|---|
| **H1 条件有信息** | Bayes最优 `R_0^*−R_1^*=E‖Δv_W^*‖²≥0` | S5 E-G身份优于S0；P1 Writer接入时AUC +18.53pp | 固定 `z_t,U,g,s` 置换正确/wrong/null Writer，分t FM风险差；同样本来源留出 | 若正确/错Writer无区分，优先查Writer ID/remap、注入短路和曝光；不否定Bayes恒等式 |
| **H2 端点只知均值** | `v_w^*(ε,1)=ε−μ_w` | 旧S9 Pair仅噪声端；K-shot均值原型差 | t≈1的Writer速度差跨噪声是否近常量；中t是否随z变化且与真实Writer差一致 | 若t≈1的差依赖噪声很强，查模型端点误差/高噪条件注入；不能解释为理想端点“多峰已学会” |
| **H3 中段能表达分布形状** | 条件score恒等式与高斯协方差反例 | DINO Writer×字残差5.6%，但QDA logloss差1.34 | 同位置中tWriter Δv 随z/t的统计，与最终多seed多写法分布及方向关联 | 中t变化若与Writer身份/GT分布不相关，只能称噪声或过拟合 |
| **H4 局部条件改善身份** | S5 vs S0 E-G宏AUC +5.73pp CI[1.65,9.84]；S5 vsS4跨0 | S0内容强、S4局部CNN、S5局部Attention | 同协议/同来源的S4 vs S5、未见字符配对、真实writer swap和t扫描 | 若优势只在来源或已见字上，放弃“局部表征更可迁移”的概括 |
| **H5 Decoder主要改善RGB质量** | J2−J0 SSIM +.0049 CI[.0039,.0060]、Writer未同步升 | D1可训Decoder提供额外图像自由度 | 同latent两Decoder交叉解码，比较OCR/Writer/LPIPS/InkIoU/笔宽 | 若OCR/拓扑显著变化，则“只改纹理”假说被证伪；若同latent无增益，则收益来自U-Net联合路径 |
| **H6 Aux可能与主图像冲突** | 12ch等权梯度占比约51%；4ch同50k不输12ch | `ΔL_img≈−η(‖g_i‖²+λ〈g_i,g_a〉)` | 同batch分梯度内积、每层/每通道贡献；同计算4ch vs12ch | 若梯度方向一致且同算力有净增益，则禁止使用“辅助目标天然有害”叙事 |
| **H7 纯STD是开集输入，不是开集正确性保证** | v83未见13字InkIoU .1586 vs全体.2990；v83仍混有char table | 连续g能传新字形 | 纯STD条件同架构按All-Stages-Unseen大样本OCR/人工判错、复杂度分层 | 真开集与seen一致才可宣布该维瓶颈解决；若差距持续，则优先字形覆盖/条件训练 |
| **H8 Writer-first适配更高效** | P75条件曝光约19.1k；P1早期AUC上升；A首点改善 | Writer条件学习可能滞后通用字符能力 | **B vs C2**同P0、同25k目标更新，比较OCR/E-G/Writer-swap | 若C2不优，不再为Writer表设计更激进日程，转审注入与数据 |
| **H9 通用P0有净迁移** | P75 P0→P2内容升，但未有scratch总算力对照 | P0可能改善结构表征 | 预训练50k+FT25k vs scratch75k同架构+数据合同；另做同Top10曝光对照 | 如果同总算力/严格未见皆无益，转向直接目标域；若只低曝光有益，则报告“样本效率”而非“算力效率” |
| **H10 独立OCR是必要内容轴** | 修Bug后S0 50k val100 Top1 71.11，GT94.44；h187 S0@85k81.29但**不同面板** | OCR能发现一部分结构问题 | 6975真实Test验收+人手校验Corruption、GT分层与词表完整覆盖 | 若OCR对真实缺笔不敏感，则只能作字身份代理，不能作为局部正确性真值 |
| **H11 风格幅度被压缩** | J关系几何方向约.85、幅度 .66–.75 | 模型条件差可能偏均值/保守 | 严格同字同噪声swap＋真实同字参考均值；按Writer/char给CI | 真实来源平衡或swap后幅度恢复，则旧0.72仅是面板/估计偏差 |
| **H12 SDE不是通用补丁** | v80 corrected固定γ SSIM差+.000576 CI跨0；Frag−.072探索性 | exact SDE对理想场同边缘，不保证近似场提质 | 同NFE、固定ckpt/STD/CFG，在验证集选时间窗、冻结Test与seed | 若画质/内容同时稳健改善，才发布采样收益；旧v68大涨不能直接作证 |

## 2. 一条具体且廉价的条件速度场检查

如果能严格加载 S0/S4/S5或 A/B/C checkpoint，在冻结的真实GT latent上，生成对应同一字符的 `g,s`，固定噪声 `ε`。取

\[
\mathcal T=\{0.10,0.25,0.40,0.60,0.80,0.95\}
\]

（仓库 [v90容量探针](../../exp.md) 已经使用这一组时间）。对每个t和样本：
1. 构造**唯一一个** `z_t=(1-t)z_0+tε`；
2. 记录 `u=ε-z_0`、`v_\mathrm{true}`、`v_\mathrm{wrong}`、`v_\mathrm{null}`，核对完全一致的输入；
3. 计算 `L_\mathrm{true}(t)`，`ΔL_\mathrm{wrong}(t)`、`ΔL_\mathrm{null}(t)`、`‖v_\mathrm{true}-v_\mathrm{wrong}‖/√d`，及两者输出差的层/尺度分量；
4. 在Writer×Script×字复杂度×Seen标签分层出中位与CI，并做Writer标签置换负控制。

**理想端点附加检验**：对同一个 `(g,s,a,b)` 和多个高斯 `ε_j`，在 `t=1` 计算 `v_\theta(ε_j,1,a)-v_\theta(ε_j,1,b)`。Bayes理想值 **与 `ε_j` 无关**；若实际方差较大，意味着有限模型没有达到理想端点，或其正交条件/时间实现不符合假设。这个测量**不需要训练**，但不能据一个端点一致就说完整Writer分布已经学对。

**特别提醒**：不能将 `t=1` Pair真实GT `z_b-z_a` 的单对随机性当作必须逐样本复现的Writer均值真值。用多真迹/多字均值或bootstrap才可估计Writer条件均值；若只取一对，其差包含单张作品变化。

## 3. 如何选下一组训练而不是一直扩军

**决策树（源自现有数据，不是新的训练已经执行）**：

**第一关：协议不可信？** 若OCR按idx强JOIN不通过，或 val100 seen定义冲突未解、P75实际架构/LR不可核，停止发布排名，先CPU修数据和跑同一GPU复评。

**第二关：A/B/C正确完成？** 比较同一187×3、原val100、严格unseen；若A继续改善且Writer/OCR不退，延长有价值；若B/C更好，要先确定是否只是更高Top10曝光或更激进Writer LR。不要用“P75 75k”与“S5 50k”仅看总步数。

**第三关：Writer条件是否真正起作用？** H1/H2/H3的 t扫描及Writer-swap若弱，优先样本采样/注入/Writer映射，不先写S9++、更复杂GPT-DiT条件分支；若中t条件强但OCR不稳，优先内容保护和未见字数据，而非再扩大E-G loss。

**第四关：画质差来自Decoder吗？** 在同latent交叉解码时若D1改善 LPIPS、边缘、笔宽且未损伤内容，再保留5k后期Decoder阶段；若未改善，停止“图像监督越加越好”的扩张。

**第五关：预训练值得重新设计吗？** 只有当前后训练已找出合理配方且严格 unseen 的评测可复现时，再用 [math/04](04_TRANSFER_EXPOSURE_AND_STATISTICS.md) 的总算力与条件曝光两组对照评价。

## 4. 预注册每个节点的结果，不再凭单指标提名冠军

正式报告按协议给出：

| 维度 | 主估计量 | 保护指标/反事实 | 必报身份 |
|---|---|---|---|
| 内容 | OCR Top1（GT正确率/覆盖单列）、严格未见字准确率 | Top5、Margin、人工缺笔/错字、InkIoU | 样本idx、Unicode、OCR vocab/ckpt |
| Writer | E-G跨字 Writer宏AUC与GT ceiling | EER/mAP、正确/错条件FM风险、swap方向/幅度 | 参考库K、E-G权重hash、书体/来源 |
| 画质 | LPIPS特定骨干、SSIM特定实现 | 墨量/宽度/Frag、近白、异常率、固定latentDecoder交换 | VAE/Decoder/sampler与EMA |
| 泛化 | Unseen-in-Top10与Unseen-in-All-Stages分列 | 同来源 vs跨来源、复杂度分层 | 全阶段训练字符集合与来源标签 |
| 统计 | 同字符paired effect、预先定义主要指标 | 分层bootstrap CI、训练seed稳定性 | 协议与评测实现hash |

**建议的工程守护界限（不是实测定理）**：新模型若Writer宏AUC或OCR下降明显，应至少给出95%CI并展开失败字，不以小于1%绝对SSIM收益直接晋级。若需要用于自动筛选，可在独立Val上预注册非劣限，如OCR−2pp、Writer AUC−2pp，再在冻结Test验证；没有CI不能声称“无害”。

## 5. 数学研究在本阶段应停止在哪个边界

- 已有数据**允许**说：Writer条件拥有可泛化信号；某些局部注入改善身份；Decoder解冻可以改善图像域；额外辅助目标可能与主生成竞争；预训练净收益未证。
- 已有数据**不允许**说：Writer低维度已被数学证明最优；Writer与Char独立；S5是唯一最优结构；E-G=完整书风真值；只靠 t=1 Pair就能恢复完整Writer分布；QDA失败意味着Writer协方差必然不重要；OCR能检测所有拓扑错；P0通用预训练一定有害。

**最终理论工作的责任**：新数学模型应产生**可测量且能被实际数据反驳**的预测，而不是为已获得的实验排名寻找事后解释。每次证据与预测矛盾，优先复核协议/代码和统计条件，再修正理论假说，绝不能改测量去配理论。

导航：[FM端点与score](01_CONDITIONAL_FLOW_AND_TIME.md)、[Writer表示](02_WRITER_INFORMATION_AND_IDENTIFIABILITY.md)、[VAE/目标](03_LATENT_DECODER_AND_OBJECTIVES.md)、[预训练与统计](04_TRANSFER_EXPOSURE_AND_STATISTICS.md)、[对应实验顺序](../1031/05_NEXT_EXPERIMENTS.md)。
