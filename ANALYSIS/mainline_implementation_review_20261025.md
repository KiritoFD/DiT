# 新主线实施审查：先校正证据，再固化 P0/P1/P2

审查日期：2026-10-25。用户要求审查附件建议与仓库新状态；本轮不修改训练/推理实现、不启动任务。
采用 code-comprehension 的调用链核对流程，产出本报告。仅新增分析文档。

## 1. 核对范围与当前身份

- 本地 HEAD `dc41bcc`，工作区审查前无已跟踪改动。附件引用的公开 `3f13f34` 快照已经不是本地状态。
- 4090 HEAD仍显示 `ac2c25e`，但源文件有未提交演进。不能据远端Git标签断言代码陈旧。
- 本地/4090的 `tools/u100_pretrain/run_p1p2.py` SHA256一致：
  `e486a4c9b94654f5f3e896f43a1884b77bd1064124fe20262c3e692502de3ba5`。
- 本地/4090的 `exp_writer_swap.py` SHA256一致：
  `3b760ad8f6281328e2eeec5e8c338645839e50042407bc49edb37fd67dae3941`。
- 核对根文档、阶段脚本、U-Net配置/条件/参数组、数据与配对采样、FM、统一评测、离线汇总及远端实际协议。
- P1/P2训练已完成。远端 `assets/u100_p1p2_20261025/status.json` 仍把离线评测记成failed，
  `pipeline_summary.json`缺失；重评生成的逐集CSV存在，但`offline_eval.json`里的两个结果均为`{}`。

## 2. 最重要的证据校正

### 2.1 P1/P2并非已完成“同预算、同协议”因果验收

| 实际设置 | 直接FT | P1/P2 |
|---|---|---|
| Batch | 256 | 512 |
| 总下游步数 | 30k | 10k + 20k |
| 总样本曝光 | 7.68M | 15.36M |
| 数据 | 80% Top10 +20%通用回放 | P1通用70%/Top10 30%；P2纯Top10 |
| Top10名义曝光 | 约6.144M | 约11.776M |
| LR分组 | 全局1.5e-5 | 主干1e-5，部分条件参数5e-5 |
| Writer dropout | .25 | P1 .25；P2 .10 |
| val100 STD sigma | **0** | **.65** |
| val100 SSIM/LPIPS | .5829/.35807 | .5908/.34975 |

曝光量为步数×实际batch与混合比例的名义计算，不声称各图片均匀出现。
P1/P2的15.36M总曝光还高于直接FT50k的12.8M，不能据“30k超过50k”宣告更省曝光/算力。
P1/P2结果可保留为**表现良好的组合候选**，但必须撤下“同预算假设已成立、机制已证实”的强结论。
同协议重评可以排除sigma混杂，不能排除batch/曝光/LR/采样混杂。

远端权威协议：

- 直接FT：`assets/u100_pretrain_20261024/ft_runs/20261024-140020-Unet-flow/eval_protocol/step0025000_eval200fix.json`，sigma0。
- P2：`assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/eval_protocol/step0020000_eval200fix.json`，sigma.65。
- 两者条件hash不同，分别以`9deb7c09...`和`5ef21d73...`开头。
- P2 test200离线CSV为.5895/.34656/IoU.28744，eval200fix为.5834/.34880/.29180；不是数据丢失。
- 直接FT50k的test200为.5874/.34886/IoU.28594。与P2同为sigma0、seed0、CFG1、Heun50，
  且STD条件hash同为`c4ad6c14b4c07fcd9e6b6940fa16d8893f118efe0de6d7030fc8abead3dad0e7`。
  因而P1/P2仍有同推理协议下的小幅正向候选证据；但预算不同，且尚无配对置信区间或独立风格验收。

### 2.2 阶段迁移与参数组不能只按文档名称理解

`src/unet/transfer_shared.py` 和 `train.py` 的实际行为：

1. P0 EMA权重严格初始化P1，P1 EMA权重严格初始化P2。
2. **每阶段新建optimizer、scheduler、EMA时钟**；不是保留原Adam历史的resume。
3. `writer_cond_lr`高LR组实际仅包含`writer.`、`writer_proj.`、`script.`。
   `script_proj.`与共享`condition.`仍在主干LR组。不能笼统写“全部条件MLP高LR”。
4. 当前新主线`freeze_writer_table=False`，整表可训练；不同于早期Reference实验的冻结独立书家表。
   后续应显式保留这个差异，而不是悄悄套回旧合同。

### 2.3 现有Writer-swap工具必须先修，不能直接用于选Teacher

`tools/u100_pretrain/exp_writer_swap.py`：

- 第60行从0积分到1；主FM定义是噪声`t=1`向图像`t=0`积分。
- 第63/67行直接把0..1送给U-Net；训练及统一采样器使用`t*1000`。
- 第92行将`gray[N,1,H,W]`按`gray[ci,:,:,0]`保存，得到1×H而非H×W。
- LPIPS在生成前加载且没有实际参与此脚本的有效评分，成为无关依赖。
- `spread_ratio`分子分母是同一批跨Writer距离，构造上约等于1；目前虽未输出，也应移除。
- 不应把所有书家都放入未观测过的script组合；候选集应由训练支持的writer×script组合决定。

因此，“LPIPS加载后死亡”不是这个工具唯一的问题。修复原则是**删除独立ODE，复用统一sampler**，
先保存正确PNG/latent和索引，再单独运行内容、笔宽、Writer评价。不是靠修改几处常数继续维护第二套采样器。

## 3. 应当立即实施：小范围闭环，不重造训练框架

### A. 固化当前候选与评测合同（最高优先）

已有资产必须复用：`Config`、严格stage transfer、`RunRecord`、`EvalRequest`、protocol DONE、权重hash。
附件中“eval永远固定187张/sigma.65”的描述已过时：`eval_args`会读取实际CSV长度，sigma来自Config。
但CFG1仍是adapter默认值，离线脚本还存在多套映射入口，需要收束。

最小改动：

1. 在`tools/u100_pretrain`下区分`pipeline/`、`evaluation/`、`diagnostics/`，各保留约5个职责文件；
   老CLI仅作兼容转发。只整理主线与此次要复用的工具，不全仓重构。
2. 阶段配置明确数据hash、Writer标签空间、null行、采样器、可训练参数组、LR/horizon、初始化来源及重置策略。
3. 阶段状态绑定resolved-config hash；请求改变配置时不复用旧done。
   选择预期步数的确切checkpoint，不用任意目录下“最后一个pt”当阶段终态。
4. 基于已有EvalRequest扩充明确的sigma、STD/有序样本hash、时间尺度、Writer映射hash、guide窗口和NFE。
   校验需要新字段时按现有规则升级schema；不另写第二个协议系统。
5. 修`eval_offline.py`汇总：共享评测返回按集合嵌套的结果，当前只筛最外层数字，造成空字典。
   显式取目标集合并校验关键指标有限、样本数匹配；protocol复用时从已完成指标恢复，不能写空汇总。
6. 汇总成功后原子更新pipeline状态；不得把“训练完成”“生成完成”“指标完成”合成一个布尔值。

首先用现有权重完成P2/直接FT的相同val100、sigma0与.65成对重评，不需要重新训练。
有效checkpoint即使当前机制解释未完成，也可以保留为推理候选，不必阻塞全部研发。

### B. 标签空间与实际曝光（最高优先）

- 保留`raw_writer_id`、`writer_group`、`writer_row`，数据manifest声明`label_space`及映射hash；
  原始ID→模型行的转换只有一个入口，显式标记已转换，而不是靠数值大小猜测。
- P1当前用42,970条重复Top10行达到30%比例，实际仅覆盖**19,720个独立Top10 img_id**，
  并非该阶段覆盖全部23,485张。应记录这一事实；改为两池概率采样会改变实验，需新run身份。
- 复用RunRecord：每次实际优化累计总/通用/Top10曝光，按writer/script计数；以CPU img_id计数或bitset记录独立样本覆盖。
  保留日志边界的执行探针，但不能用稀疏探针估算全程精确覆盖。
- 同时记Writer非null的**有效**曝光，区分原始数据混合与dropout后的条件曝光。
  当前默认drop_all=.05、P1 drop_writer=.25时，真实Top10 Writer条件约占全batch的
  `.3*.95*.75=21.375%`，而不是名义30%；用实际mask计数验证。
- `PairMixtureSampler`是配对采样，不是对比loss。保留现有普通FM；不因此顺手加rank/CE。
- `src/train/data_setup.py:150`目前拒绝配对采样的非零步resume。正式长训前应支持确定性epoch/offset恢复；
  不以重置Adam/EMA冒充完整恢复，也不承诺CUDA训练逐位一致。

## 4. 下一项最值得运行：P2 Writer CFG，再决定蒸馏

不改U-Net结构；新P2不是Reference模型，现有`forward_with_cfg`已支持只置空Writer、保留STD与Script。

1. 修swap后复用统一sampler。冻结P2终态EMA、val100与seed；扫1.0/1.2/1.4/1.6。
2. 先用单seed筛选，再只对baseline和优选项做额外seed复核，避免完整笛卡尔积消耗。
3. 全程记录SSIM/LPIPS/IoU、墨量、骨架脊线宽P10/P50/P90与W1、Frag中位数/尾部、Writer macro/margin。
4. 固定STD和噪声换Writer，候选仅限同script的已支持书家；先测真实GT及VAE round-trip的评估器基准。
   不把Teacher自身分类器涨分当作真实风格证据。
5. 全程CFG有收益后，只加一个中噪声窗口候选，例如归一化t∈[.2,.8]；其数值是工程试点，不是最佳区间结论。
   主采样方向1→0；传给模型的t为1000倍，窗口判断必须显式还原。
6. 窗外s=1时跳过null前向。Heun50基础NFE100，全程Writer CFG是200个单样本等效分支前向；
   窗口引导介于两者。记录solver评估次数、等效分支前向、总延迟/显存，不仅记录Python调用次数。

通过后再做5k蒸馏试点：Teacher冻结，Student由同一P2初始化；相同xt/t/STD/script下，
Teacher给出`v_null+s(t)*(v_writer-v_null)`，Student只做一次条件前向拟合该速度。
保留相同步数与sampler，不同时做少步蒸馏。先比较P2 CFG1、Teacher CFG、Student CFG1三者。
若风格、宽度、内容不能同时保持，停止蒸馏，不因“单次前向”就默认成功。

## 5. 数据扩展：值得做，但不能按附件直接开40万

用户此前明确限定楷/行/隶。本轮复核原始metadata为393,486张，其中三书体总共**101,678张**。
现有100k池已经接近该范围的可用全集；其余不是三书体的额外30万张。

因此：

- 现在优先修错误极性、错标、重复和损坏，并补合法写法/复杂字/稀缺作品覆盖清单。
- 只有新增可信的楷行隶来源，才能在不改任务范围的前提下做真正数据扩展。
- 若要纳入其他书体，须单独确认；这是script空间、STD字体、映射和采样协议的联合扩展，不能只换CSV。
- 来源目录不是作品ID；保存source与work_id_known两种字段，未知不伪造。
- 飞白、重墨、草化、偏离STD只标写法属性，不直接判脏。已有`tools/data_clean`候选/人工决策/隔离流程可复用。
- 新测试字符应先于数据扩充确定并阻断所有训练阶段；同时报告全局未见字、writer×script×char未见与来源留出。
- 数据池比较固定总曝光/墙钟及下游配方，再检查短程下游收益，不能只看P0单GT SSIM。

## 6. 后排候选与暂缓事项

| 方向 | 建议 | 最小边界 |
|---|---|---|
| 冻结latent的Decoder微调 | 中优先独立支线 | 固定Encoder、scale/shift、latent缓存；相同GT latent与生成latent分别比较旧/新Decoder |
| 内容监督或REPA | 中优先、按错误类型二选一 | 生成分支有梯度；单位置小权重；不能让STD分支直接读标签代替生成内容 |
| Style loss | 暂缓训练 | 先校准独立评价器的跨字/来源与生成梯度，不能复活强Critic堆权重 |
| Writer专用调制/更大骨干 | 暂缓默认启用 | 当前条件入口可继续；只有同协议明确缺口时单独消融 |
| 新VAE/16ch/f4 | 低优先 | 重建提高不代表生成提高，需另立latent合同与下游对照 |
| Qwen通道/合成数据 | 保留支线，不扩量 | 已有A/C对照未显示可靠增量；生成字对、书家标签可信后再谈训练数据 |
| Reference/CrossAttention/码本 | 暂缓 | 当前路线无可靠净收益，不等于理论上不可行 |

Decoder试点不能只对预训练GT重建涨分：还要检查生成latent、旧Decoder风格锚定、灰背景/碎墨伪影与内容退化。
如无合法配对监督，不把生成latent直接硬拉向某张不对应的GT。

## 7. 实施顺序与验收

1. **第一批工程**：统一Writer ID空间、修swap采样与落盘、修离线汇总与状态、对齐现有模型评测协议。
   只做三个关键检查：映射幂等/原始语义保留、常量速度oracle复用主FM、协议复用返回非空指标且图像维度正确。
2. **第二批工程**：固化P0/P1/P2参数组/重置合同、精确曝光计数、配对采样resume。
   现有P1/P2作为候选保留；若要证明“更省预算”，补同曝光/同协议的直接FT对照，而非反复扫架构。
3. **首个推理实验**：P2 CFG有限扫描+独立Writer/笔宽/内容评价；不启动新的大预训练。
4. **通过后**：固定优选Teacher做5k CFG合并蒸馏；少步采样另开后续实验。
5. **新增三书体数据确实存在后**：固定下游配方做P0数据扩展；Decoder小支线与主线不捆绑。

P1/P2数值可作为当前候选证据；“同预算全面优于”“EMA状态沿用”“val100 sigma0”“参考路线严格证伪”
应在后续文档勘误中分别改成实际曝光/重置/协议/适用范围。不要在本轮静默重写历史记录。

## 8. 补充审查：共享书家条件不等于固定图像偏移

本节核对用户追加附件及当前`model.py`、`blocks.py`、`exp_m1m2.py`、`calliflow_v2.py`；不启动新实验。

### 工作假设与结论边界

- 建模对象为`p(x | STD, script, writer)`。共享Writer embedding通过非线性空间网络，可以产生局部、字符相关响应。
  `scale/shift`不直接读取STD，不代表最终输出中没有Writer×STD交互。
- 当前AdaGN在STD产生的空间特征上工作，已存在这条交互路径；有表达入口不等于已学会正确风格。
- M1/M2确有三个`.01`尺度因子相乘，并使用Adam和weight decay。没有拟合收益既可能来自弱可预测信号，
  也可能来自参数化/优化；小初始化本身不足以确定根因。
  因而不再以该阴性结果断言“字符相关交互客观不存在”。无需为了这个勘误重启大型交互网络。

### 优先加入一项免训练的Writer条件效用审计

扩展现有`tools/cohort_eval/probes/fm.py`的时间分桶和固定噪声逻辑，不再另造FM/ODE实现。

1. 固定P2 EMA、真实GT latent、STD、script、噪声及t；仅改变Writer。
2. 三组：正确Writer、同script有训练支持的错误Writer、Writer-null。
   错误Writer可取所有合法候选的平均；不得用另一个GT或另一个噪声替换目标。
3. 分别报告`Delta_wrong = mean(L_wrong)-L_correct`、`Delta_null = L_null-L_correct`，
   绝对值/相对值、胜率与按字符聚类的配对置信区间。先小样本，无须扩大成新一轮探针研究。
4. 复用统一FM：`z_t=(1-t)z0+t*eps`、target=`eps-z0`、模型时间`t*1000`。
   VAE/scale/shift与latent合同固定；重新编码时明确使用posterior mode，不假装是旧训练缓存的逐位重放。
5. 分层区分P0/P1/P2联合训练字符全局未见、仅writer×script×char未见、已见组合新图。
   若严格全局留字样本不足，按真实数量报告，不把普通val100称作严格留字集。
6. 按t及预先固定的字符复杂度分桶；低t已包含大量GT信息，Writer收益小不直接证明条件无效。
   高/低t收益也不能直接命名为结体/笔锋机制，最终需完整采样验证。

这里有一个精确的解释限制：同一`z_t`下`z0_hat=z_t-t*v_pred`，所以一步clean-latent MSE等于
`t^2 * velocity MSE`。不同t的原始Delta不能直接解释成相同大小的图像改善。
反之，正Delta表示当前留出分布中Writer有助于预测，不证明完整条件分布、书风或CFG已经正确。
这项审计与修复后的swap/CFG互补，不作为暂停所有主线工作的强制门槛。

### CalliFlow-v2只保留为后备，不与主线混为CFG实现

当前分解`v_w=b+d_w`，null输出若为`v_null=b+d_null`，则真正的Writer CFG是：

`v_cfg = b + s*d_w + (1-s)*d_null`。

代码的单次`b+s*d_w`仅在`s=1`或`d_null=0`时与上式相同。输出头零初始化只保证初始残差为0，
不保证训练后的null残差为0。不得用“单次CFG”名称掩盖这项语义差别。
现阶段称其为“Writer残差强度缩放”更准确；不改旧模型，也不据此切换主线。

后续决策不变：**修评测 → 固定P2 → Writer效用小审计与CFG → 符合独立指标后蒸馏**。
若条件效用弱或CFG无收益，先核对标签、dropout、训练覆盖、评价器和优化冲突，再考虑条件结构。
探针阴性和已有U-Net的表达能力，都不能单独决定书法风格的本质或模型上限。

## 外部依据的适用边界

- [有限引导区间，NeurIPS2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/dd540e1c8d26687d56d296e64d35949f-Abstract-Conference.html)：支持将窗口作为超参数，不提供本项目最佳窗口。
- [引导蒸馏，CVPR2023](https://arxiv.org/abs/2210.03142)：先合并条件/空条件输出，再研究少步；不承诺本项目无损或墙钟严格2倍。
- [REPA](https://arxiv.org/abs/2410.06940)：主要证据来自扩散/Flow Transformer，不是本CNN U-Net的已验证收益。
- [Diffusability，ICML2025](https://proceedings.mlr.press/v267/skorokhodov25a.html)：支持区分重建与latent可建模性；不证明保持旧latent的Decoder微调必然改善本项目。
