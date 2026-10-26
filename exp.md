# 实验总账

## 2026-10-27：S组恢复编译，从零50k（4090已启动）

[完整设定、队列与infra纠正](ANALYSIS/style_repr_injection_matrix_20261026/compile_cold50k_20261027.md)。S0 R0/I0、S1 R0/I1、S4 R2/I1、S6 R0/I3依次50k；同seed90、B512/micro512、LR6e-5/warmup1500、WD0.02、EMA0.999。全部随机初始化，旧Writer路径固定null，无梯度检查点。运行根目录`assets/style_matrix_20261027/cold50k/4090/execution`；各组checkpoint/日志/统一评测独立保存。新入口曾强制eager，已修正；旧eager耗时不再用于删实验。当前S0首次编译中，无生成质量结论。历史编译吞吐估算四组约37h，待稳定计步核实。提交34e06e4。

## 2026-10-26 续实施：S/J预检、真实校准和串行落地（未长训）

[逐步证据与预算](ANALYSIS/style_repr_injection_matrix_20261026/README.md#7-续实施最快安全batch真实恢复串行方案)。首个实现提交`ae14347`。48 J0/J1/J2/J3分别1.0213/0.6541/0.6069/0.4264step/s，图像micro8与4短测后选8，J3峰值40.279GiB；四臂梯度连通/隔离均符合设计。4090 FP32非等长累积与B512检查通过，BF16仍有0.226%梯度/1.90%更新差；S2-large容量与真实P1两步保存恢复通过。
J3从实际P2 EMA及配对GT标定初始Image梯度为FM的10%，真实两步保存恢复、Decoder raw/EMA/优化器状态与独立VAE导出加载已核对；只做预检，不形成质量实验。
路径两端`assets/style_matrix_20261026/plan_final/`；S=10k→29k→43.5k、J=4k→11.5k→19k，模型分配/源权重与CSV哈希/spec/阶段任务入计划。默认dry、ready未封存；自然墙钟预测63.95h含8h余量。正式长训、测试集生成和风格比较均未进行。


## 2026-10-26 最新：S/J风格表征与注入消融（实现、容量短测，未正式训练）

[设定/每步状态/数据路径/原始JSON](ANALYSIS/style_repr_injection_matrix_20261026/README.md) · [代码入口](tools/style_matrix/README.md)。S0–S8+S2-large从P1@10k EMA比较表征/注入；J0–J3从P2@20k EMA独立比较冻结/训练Decoder及有/无生成图像loss。R4与旧20格长训不执行。
有效B512、无检查点重计算；batch以安全吞吐最快为准。两端4类搜索完成；4090真实缓存短测1.0328step/s、reserved20.379GiB。J3容量测试0.4285step/s、46.145GiB，梯度连通而显存配置未定型。所有这些均为短测，无新质量指标/生成海报；正式步数和64h总ETA未锁定。
实现包含唯一Writer路由、累积、顺序反向、两类参数独立裁剪、Decoder raw/EMA保存；严格累积、真实J配对、联合loss校准、完整恢复/成套评测、鲁棒串行仍需完成。


## 2026-10-25：P1/P2 writer-aware 混合训练验收（假设成立，路线锁定）

设计：P0=kxl-100k 75k EMA（复用，40 万池替代已记录）→ P1 10k（70% 通用 null 行 +30% Top10，主干 LR1e-5/条件组 LR5e-5，
drop_writer .25）→ P2 20k（Top10，PairMixtureSampler 30% 同字同书体异书家配对，纯 FM 采样，drop_writer .10）。
B512/EMA/强增强沿用/全解冻/每5k存评。实现 `tools/u100_pretrain/run_p1p2.py`（null_writer_rows、writer_cond_lr 参数组、
mainline 放行 pair_fraction 仅采样）。

| 预算 | 路线 | val100 SSIM/LPIPS/IoU/Frag |
| --- | --- | --- |
| 20k | 直 FT | .5800/.3603/.2632/2.27 |
| 20k | **P1+P2@10k** | **.5864/.3535/.2725/2.02** |
| 30k | 直 FT 延长 | .5829/.3581/.2638/1.91 |
| 30k | **P1+P2@20k** | **.5908/.3497/.2814/2.18** |
| 50k | 直 FT 终态 | .5844/.3558/.2660 |

离线：eval200fix σ.65 .5834/.3488/.2918/2.79；test200 σ0 .5895/.3466/.2874/2.87。
**假设成立**：同预算 P1/P2 全面优于直 FT，且 30k 预算超过直 FT 50k 终态。路线锁定（详见
`ANALYSIS/p1p2_writer_aware_20261025.md`）；P0 用 kxl-100k 替代 40 万池（偏差已记录）。

## 2026-10-25：风格结构诊断系列（决定上述路线）

纯分析六轮（DINOv2-B，17,569 图/1,609 字/10 书家；严格留字；方法三轮自我修正）：
方差分解（主效应 .5%/交互 5.6%/共享项相关=基线 0.4987 vs 0.500）、patch 谱+ridge（无结构、零增量）、
多尺度 G/C/S/L（G .436 留字 Top1；C/S 零增益；L +1.9pp）、分类器阶梯（质心.284<LDA.328<QDA.361<logistic.426，
MLP 无增益）、结构测量（K90≥32、交互留字负增益、d_eff95=7、QDA log-loss 更差）、M1/M2 双线性（ΔR²_test≈0）。
**结论：书家风格=跨字稳定、线性可读、弥散于 ~7 维+协方差的分布；无可泛化的字形条件调制；参考路线与
字形调制路线关闭；Writer Embedding 容量足够。**方法修正与全部数字：
`ANALYSIS/style_structure_diagnosis_20261025.md`、`ANALYSIS/results_20261025/`。

## 2026-10-25：eval200fix 极性损坏样本（数据质量）

idx16「留」（何绍基隶书）外框极性错误、笔画粘连不可读；poster 自 v5 剔除该列；
`ANALYSIS/eval200fix_polarity_corrupted_gt_20261025.md`；final_imgs_fame_v8 来源待极性全量审查。

## 2026-10-25：4090 深度磁盘清理（DiT 992G→399G）

旧代 ckpt 487 个（423G）、dino_cache 108G、v90 scratch 70G、归档白名单外 330 个（260.6G）、
_quarantine/_models/baseline/dataset 等；保留 19 个代表性归档 ckpt + 现行基线终版。
清单：`assets/disk_cleanup_*.json`；代表 ckpt 索引：`docs/experiments/archive_representative_checkpoints_20261024.md`。

## 2026-10-24：U100 纯通用预训练 75k → Top10 后训练 25k 完赛报告

管线 `tools/u100_pretrain`（4090 单卡 10.7h 全程，含三次可查修复：FT 导入层级/离线评测 inference_mode/eval200fix char 映射扩展）。

**最终指标（FT 25k EMA，三口径）**：

| 集 | SSIM | LPIPS | Ink-IoU | Frag | 空白 |
| --- | ---: | ---: | ---: | ---: | ---: |
| val100（σ0，可比 0.56 系） | **.5817** | .3579 | .2654 | 2.12 | 0 |
| test200（σ0） | .5826 | .3527 | .2804 | 2.84 | 0 |
| eval200fix（σ.65，可比 U100） | .5771 | .3536 | .2790 | 2.50 | 0 |

**横向**：val100 超过 reference 25k（.5775/.3612/IoU .2631/Frag 2.72）与 generic48 全解冻（.5612）；eval200fix 低于 U100 从零 .6132 约 .036（对方 60M 曝光、纯 Top10-aug 数据、18.6h@48；本线 25.6M、纯通用预训练、10.7h@4090——预算与数据构成双重不同，不能归因单一变量）。
**FT 动力学**：0步 .4916 → 5k .5712 → 10k .5767 → 25k .5817；增益 92% 在前 5k（行 0-9 安装 + 轻适配），此后缓升无回落。
**预训练段**（val100 σ.65、行 0-9 未训练＝近内容基线）：.4291@5k → .4974@55k 峰 → .495@75k；Frag 20.6→5.3、零空白。
**机制结论**：(1) 通用预训练+可训练行 0-9 的路线在 1/2 预算下达到与最强域内从零线可比较的水平并全面超越此前所有课程方案；(2) 行 0-9 从 SupCon 初始化在 5k 步内即完成安装，验证了"读取器预训练+FT 训行"的修正后设计；(3) 距 0.6132 的差距是预算还是路线，需等预算延长实验（FT 50k / 预训练 150k）分离。
配置与产物：`assets/u100_pretrain_20261024`（manifest/config/15+5 ckpt/离线评测）。

## 2026-10-24：U100 纯通用预训练 → Top10 后训练管线（4090，进行中）

决策链：Reference 线单向量注入证伪（0步.5586→25k仅.5775）→ 回归 U100-Deep-Sym4 114M 设计
→ 用户修正"书家表可独立对比学习训练"，推翻"Top10 行必须进预训练"的论证（行=320参数可FT学；难学的是读取器通路，
读取器由 248 通用组预训练解决）→ 最终单臂方案：**纯通用 100k 预训练 75k + Top10 后训练 25k，drop_writer 统一 0.25，ckpt/评测每5k**。
对比论证与根因分析见 [设计文档](ANALYSIS/u100_generic_pretrain_design_20261024.md)；实现契约见 [管线README](tools/u100_pretrain/README.md)。

关键实现事实：
- 书家编号 259 行：0-9 Top10 预留（SupCon 冻结表初始化，预训练不训练）、10-257 通用 source-scoped 组（≥32图）、258 OTHER（3,604行）、259 null。
- STD：std_bbox64 store，clean σ.65 + strong 形变增强（14,999 字 bank，256 raster 与缓存逐位一致后才生成）。
- 114,011,012 参数，B256，LR6e-5，实测 3.21 step/s / 10.78GiB（远轻于 reference 线 22.5GiB——其含 DINO 银行与 SupCon 开销）。
- 预训练 75k ≈ 6.5h；FT 25k（全权 EMA 初始化、整表可训练、20% 通用回放、LR1.5e-5、eval σ0、0步基线）≈ 2.5h；离线 eval200fix(σ.65)/test200(σ0)。
- 判读锚点：FT 后 eval200fix vs U100 从零线 0.6132；val100(σ0) vs 课程线 0.56 系；val100 内联（预训练期）兼测读取器×SupCon 方向互操作性。

## 2026-10-24：Reference Top10 10k→30k续训（已停止）

实际停于25,375/30k。续训段：15k .5661/.3702/.2511 → 20k .5713/.3662/.2542 → 25k .5775/.3612/.2631（SSIM +0.001/k，
增益微弱且注入机制单向量加法已定性为弱）；不再续训。0步 .5586、5k .5631、10k .5651 同前。

## 2026-10-24：Reference Top10 10k→30k完整续训

输出4090 `assets/reference_top10_30k_20261024`，runner1832859。
恢复原后训练10k的raw/EMA/Adam/scheduler/RNG；保持数据23485/100/200、参考库和FM+.05SupCon不变。
总30k、额外20k，B256，每5k保存评测，无EMA/optimizer重置。
LR显式延长原余弦horizon10k→30k，base2e-5、min_ratio.1不变，10k恢复点约1.56e-5。
旧run保留，新run单独归档；不声称DataLoader逐位重放，不声称LR轨迹未变。
原10k验证SSIM.5651/LPIPS.37204/IoU.24681/Frag3.2969作为续训前基线；0步SSIM.5586/LPIPS.39050。
活动run `runs/20261024-025517-Unet-flow`，trainer1832923；已执行10025步。
resume_record确认EMA未重置、完整RNG恢复、LR1.560911e-5；FM.1541、SupCon约3.91。

## 2026-10-24：显式书家/书体Prompt，停止训练后同字重评

非Turbo runner1642156/trainer1645256已停止于270步，无已保存LoRA；因此本组非Turbo严格标作未微调基模。
新增前缀`书法家：{calligrapher}。书体：{script}。请用该书法家的{script}风格书写。`，保留原字符Prompt。
六字：博、縱、願、衣、難、蜿；颜真卿/智永/赵佶，均楷；seed90，512输出统一LANCZOS至256，对同GT评测，无GT输入。
非Turbo40步；Turbo LoRA@6000为8步并加载原条件投影。Turbo旧组复用保存输出，其余三组共新生成18图。

| 模型与Prompt | SSIM↑ | LPIPS↓ | Ink-IoU↑ |
| --- | ---: | ---: | ---: |
| 非Turbo基模，原Prompt | .50776 | .54429 | .19006 |
| 非Turbo基模，显式书家/书体 | .51621 | .53641 | .22447 |
| Turbo LoRA@6000，原Prompt | .49015 | .45414 | .29938 |
| Turbo LoRA@6000，显式书家/书体 | .49239 | .45130 | .31781 |

只解释各模型内部Prompt变化，不将跨模型差异归因于Prompt。读图：颜真卿部分字墨量更接近GT；Turbo縱改善较明显，其他字微变或局部退化，尚不能确认风格迁移成功。
48路径`/home/ds/Workspace/xy/qwen-prompt-explicit-20261024`，状态completed；未接续训练。
[海报](assets/qwen_prompt_explicit_20261024/poster.jpg)、[汇总](assets/qwen_prompt_explicit_20261024/summary.json)、[代码](tools/qwen_lora/prompt_eval.py)。

## 2026-10-24：Qwen非Turbo vs Turbo条件LoRA

非Turbo新run：48 qwen-lora-base-20261024，trainer1645256，正式首步已验证。
模型为已SHA校验的Qwen-Image-2.1；LoRA/条件MLP从零初始化，同seed/冻结3×32表、数据2730/18/38。
2786图posterior与1694字符文本特征用本基模重新缓存，按源图SHA验证；与Turbo缓存隔离。
rank16 Attention q/k/v/out，512/B8，目标6000更新，保存/同6字验证每500。
LR完全复现Turbo训练历史，见tools/qwen_lora/base6000.json；40步原生采样，Turbo为8步，质量与采样墙钟分开解释。
LoRA16,777,216+条件4,749,712参数，初步峰值18.4GiB；尚无检查点质量结果。

## 2026-10-24：Reference FM 25k EMA → Top10 10k

路径4090 `assets/reference_top10_20261024`；启动`tools/reference_style/finetune.py`。
源`reference_kxl_20261023/runs/20261023-195603-Unet-flow/checkpoints/0025000.pt`的EMA；
全部权重strict加载并逐张量核对，不新增模块，局部step重置0，optimizer/EMA新建。
数据train23485、validation100、test200，复用generic48 curriculum中Top10标签行的审计ID及VAE缓存；
不混入generic行，不使用val/test图片作为参考；以named writer标签仅进行采样/SupCon。
114298852参数U-Net+投影头全量微调，DINOv2冻结；STD+Script+Reference条件，无Writer-ID。
LR2e-5、warmup200、B256、10k附加步数，每5k保存/评测；FM+.05 SupCon。
0步先评validation100，后续只与同集同参考协议比较；test200留存，不参与优化。
runner1831644、trainer1831960，run `runs/20261024-010707-Unet-flow`；340张量strict且逐张量相等校验通过。
0步验证100图SSIM.5586/LPIPS.39050/IoU.23333/Frag3.0614/SoftInk1.1580；这不是后训练收益。
已完成25步，FM.3373、SupCon约4.56、有效anchor256，PyTorch显存22.47GiB；尚无后训练质量结论。

## 2026-10-23：K/X/L 100k Reference FM，DINOv2 + 114M U-Net

4090：`assets/reference_kxl_20261023/runs/20261023-195603-Unet-flow`；配置`config.json`，
数据/缓存契约`audit.json`、`features.json`、`run_manifest.json`，外层日志`training.log`。
训练100095，楷35935/行35057/隶29103；不引入其他书体或Top10额外数据。
source-scoped Writer标签仅用于采样和监督；参考严格同组同书体异字，1–2张，训练库独占。
冻结DINOv2-B/14最终CLS768D，MLP256→128，参考归一化平均，输入STD+Script+Reference。
U-Net192/384/576、每尺度4块，114298852参数；无Writer条件、无Brush、无REPA。
B256、LR6e-5、warmup500、30k步、每5k保存/评测；FM+.05 SupCon、tau.1、硬负分母权重2。
采样50%同书体2×2四元组，50%同书体writer-balanced跨字对；后者覆盖非共享字符。
验证固定训练参考，仅1044/1046有合格参考，不能与历史1046行指标直接比较。
失败B512 run保留；正式B256于10-24 00:38完成30k/6轮评测，末段FM.2037、SupCon约2.20。
25k：SSIM.4558、LPIPS.46147、IoU.23664、Frag5.039；30k：.4551/.46107/.23702/6.665。
30k Frag中位数1.183、P90为6.5，均值受极端碎墨样本影响；仍缺独立Writer与reference-swap验收。
图片`assets/reference_kxl_20261023/progress_10k_25k_30k.png`，后训练选择25k EMA。
[实现说明](tools/reference_style/README.md)。

## 2026-10-23：Turbo LoRA 2k→6k续训

从checkpoints/0002000完整恢复LoRA、条件适配、AdamW两组状态和随机状态，逐张量校验加载结果。
trainer927137，已执行2001步；原512/B8/rank16/冻结表/数据划分不变，额外4000步，每500保存验证。
续训LR：当前1e-5→100步升到5e-5→余弦降回1e-5。调度变化单独留档，不声称与未改LR纯续训等价。
配置tools/qwen_lora/turbo_extend6000.json，48日志qwen-lora-turbo-20261023/train_extend6000.log。
首轮6字指标仅显示后期形态与粗细继续变化，尚无0步/完整test/独立Writer验收；延长训练不是确认风格已成功。

## 2026-10-23：Turbo条件LoRA正式首轮

按用户要求切换Turbo，不再等待非Turbo。48任务`qwen-lora-turbo-20261023`，trainer614522。
GT/冻结书家表/划分不变：2730训练、18验证、38测试，仅颜真卿/智永/赵佶楷书。
完整VAE posterior缓存2786图、文本缓存1694字；输入只有目标字符及writer/script条件，无STD或GT输入。
rank16、LoRA16,777,216参数、条件4,749,712参数；512、B8、不累积、LR1e-4、warmup100、2000步。
每500步保存及6字8步验证，并检查实际采样步数。沿用原生1000时间步FM监督，是否保留蒸馏八步质量由验证决定。
60步实测4.10秒/步、18.58GiB峰值；不是生成质量收益。旧非Turbo改为仅下载，GPU编码与训练均停掉。

## 2026-10-23：BC FM Top10新路线

`assets/brush114_top10_20261023`，114M+5维Brush条件，P0固定VAE重建属性审计运行中。
源为旧楷行隶通用池与冻结Top10，非Top20；数据只声明实际约10万候选，首轮不引入其他书体。去重复、严格未见字符留出后记录最终规模。
P1 B512/LR6e-5/warmup1500/60k调度停5k/75%通用25%Top10，Brush dropout.1/noise.075，FM-only。控制只用独立指定属性，不从测试GT取属性。
P0及P1（含人工读图）通过后才resume60k并Top10微调30k；分支LR8e-5/其余1e-5，Writer表冻结，每5k保存评估。
与原40万、Top20泛称严格区分；不同时启动从零/无Brush对照，不能预先宣称正迁移。[完整门槛](tools/brush_fm/README.md)。

## 2026-10-23：Qwen-Image-2.1冻结书家条件LoRA

48 `/home/ds/Workspace/xy/qwen-lora-20261023`，不占4090当前训练GPU。
三位楷书writer，冻结Top20划分及独立SupCon表按raw-ID取行；保留2730/18/38，规则剔除444/3/4，逐图SHA与源CSV SHA留档。
输入为目标字符文本特征+冻结writer表投影4 tokens+1个楷书token，无STD、参考图或GT条件。
GT经Qwen原生RGBA VAE编码，posterior mean/std缓存、训练中重新采样；标准化沿模型latents_mean/std。
非Turbo正式底模下载33.13GB，与GT编码/文本编码阶段依赖并行；完整校验和缓存后启动2000步FM。
rank16 Attention q/k/v/out LoRA，512、B8、LR1e-4、warmup100、cosine至10%、writer-null10%，每500步存LoRA/条件/优化器/RNG并生成6字验证。
Turbo仅做两步反向可训练性预检：通过，峰值18.56GiB；不将短测loss下降视为生成改善，也不保证蒸馏8步能力保持。
PEFT0.21.2装任务vendor，复用现有pytorch环境；GT传输用临时SSH agent转发直传，不复制私钥、不永久授权新密钥。
当前无正式LoRA生成质量结果，状态及限制见[README](tools/qwen_lora/README.md)。

## 2026-10-23：P1全解冻Top10迁移对照

用户停止部分解冻50k续训（末次确认18,450步），改从通用预训练50k EMA重新起10k。
全网可训练，唯独冻结独立书家表与无效共享writer_proj；取消Writer-only阶段。
Writer分支/主干/STD与Script峰值LR=1e-4/2e-5/1e-5，warmup200，10k余弦，B512，每5k保存评估。
其余复用原Top10验证100张、10%回放、bbox STD sigma0和SD VAE。配置top10_full_unfreeze_config.json，目录top10_full_unfreeze_runs。
这组同时修改解冻范围、组LR与取消1k冻结，属于迁移策略组合对照，不是纯单变量冻结消融。附件中的迁移前向等价性已在前轮CPU检查通过。

## 2026-10-23：48M Top10微调10k延长至50k

停止114M预训练及串行；最后日志34,950步，保留已保存权重，不自动微调114M。
从`generic48_kxl_top10_20261022/top10_runs/20261023-055043-Unet-flow/checkpoints/0010000.pt`完整resume，额外40k。
配置top10_extend_50k_config.json：max/stop/LR horizon均50k，保存评估间隔5k；其余微调设定不变。
新output top10_extended_runs，log top10_extend_50k.log，status extension_status.json，不覆盖5k/10k已有结果。
延长余弦使10k处LR回升到约4.6e-5（分支）、9.1e-6（主干）；保留优化器和EMA，不重做Writer-only warmup。

## 2026-10-23：114.003M同口径通用预训练与Top10微调排队

4090 `assets/generic114_kxl_top10_20261023`，持久runner PID1821400，02:15确认waiting_for_48m_complete。
先等48M两阶段完成，再B512编译短预检→114M从零预训练50k→Top10 Writer-only微调10k，每5k保存/评估。
channels192/384/576、每尺度4块，通用模型114,003,044参数；其余直接复制48M最新两阶段配置及缓存路径，不重新切分数据、不加载48M权重。
LR、STD bbox/sigma0/增强off、SD VAE、独立冻结表、回放与阶段解冻设置一致。容量/ETA待空闲后的预检实测；失败不静默降低batch。
入口[queue_wide.py](tools/generic_pretrain/queue_wide.py)，预训练曲线与Top10条件拟合曲线不宜直接作为容量证据。

## 2026-10-22：48M通用楷行隶→Top10两阶段（GPU缓存中）

**最新修订**：用户要求预训练50k→Top10微调10k，两阶段每5k保存/评估。
预训练实际已启动，缓存全部124,894行/25分片、0读取失败；稳定2.66step/s、约15GiB。
不丢弃当前进度：活动30k进程正常结束后完整状态续至50k，原50k LR调度不变，之后加载50k EMA微调。
串行接管入口 `tools.generic_pretrain.full --extend-running`；新配置 `pretrain_50k_config.json` 与 `top10_50k_config.json`。
以下原30k配置与“缓存中”记录为历史启动状态，不代表最新预算。

4090目录 `assets/generic48_kxl_top10_20261022`；runner.log位于同名目录外。
候选101,678张；保留101,309张，train100,263/validation1,046，排除369张留出近重复。保护Top20与legacy留出；不做新的审美过滤。
预训练48M/B512/LR7e-5/warmup1500/30k停止/50k余弦调度，每5k保存评估。无Writer、保留Script。
Top10 train23,485/validation100/test200；独立对比学习表按raw-ID取10×32并冻结。
自动加载预训练30k EMA，新优化器和步数；新增rank16 Writer-only FiLM，前1k仅分支、后9k联合低LR微调，10%通用回放。
STD真实墨迹bbox等比fit256→64，sigma0/增强off，VAE SD EMA/mode/scale0.18215；无REPA。
当前GPU编码124,894行（含Top10训练/验证，独立CSV避免预训练混入），FM步数0，暂无质量指标或生成poster。
模型构建确认generic48,325,668参数，Writer-only48,504,772参数。[完整设定](tools/generic_pretrain/README.md)、[冻结清单](assets/generic48_kxl_top10_20261022/manifest.json)。


## 2026-10-22：Qwen-Image-2.1-Turbo跨字合成小样（准备中）

暂停原数据诊断路线，用户授权32张pilot。48GB主机复用现有pytorch环境；只在任务vendor加载diffusers0.41，原torch/transformers/diffusers安装不改。
目录 `/home/ds/Workspace/xy/qwen-calli-20261022`。官方32.45GB权重ModelScope下载，SHA与HF fc13ba1986fcb905854dc8de2f52eee395f49b5b权重一致；下载完成后再做全文件校验。
最终输入 `assets/synthetic/qwen32_20261022_final`，3书家（颜真卿/智永/赵佶）×4共享目标（泉梅岳霞）×2seed=24，另STD-only8张。
只给新bbox STD及同书家其他字训练真迹，不给目标GT；保持同书体，目标排除所有相关留出字符。
模型设置1024/BF16/CPU model offload/CFG1/官方8步sigmas/KV cache，无量化。推理已挂等待下载，尚无质量或成功率结果。
原始输出、白底256、逐图身份、人工review.csv和匿名海报将分别保存，不自动混入真迹数据。
[代码与许可记录](tools/synthetic32/README.md)。

## 2026-10-22：STD bbox几何修复，训练/推理暂停

无随机模糊训练已启动于20:56、用户随后叫停，最后确认300步；保留日志，无5k权重，不当作完整实验。
旧STD形变本已off，仍存在sigma0.4–1随机模糊；随后sigma0推理也因用户要求优先修正bbox而停止。
旧STD固定em、未按实际墨迹缩放，最长边覆盖中位70.3%、最小35.9%。
已按原font index重绘8945字：bbox等比fit256，最长边顶满、短边居中，再BOX64用于模型；新缓存std_bbox_v1，不覆盖旧std_soft_final。
[对照图](assets/audits/std_bbox_20261022/preview.png)、[审计](assets/audits/std_bbox_20261022/audit.json)、[代码说明](tools/std_bbox/README.md)。
全量跨度256/64检查与3项CPU回归通过。这里只修复缓存，尚未对新几何进行生成评测，不能据此解释旧模型全部质量差距。

## 2026-10-22：48.33M@25k/30k seen/test补评

训练20:37:41正常完成30k，书家表raw/EMA不变核验通过；X2未启动。
固定Top20 train_half200张与test400张，25k/30k共4项，EMA/CFG1/Heun50，与旧114M同路径和噪声比较。
[完整结果](assets/audits/extreme48m_seen_20261022/comparison.csv)、[拟合差距](assets/audits/extreme48m_seen_20261022/gaps.csv)。
30k新48.33M：seen/test SSIM0.7753/0.5646、LPIPS0.16194/0.35732、IoU0.70894/0.29258。
旧114M：seen/test SSIM0.83018/0.57033、LPIPS0.12474/0.35396、IoU0.77715/0.29073。
SSIM gap缩小18.9%、LPIPS gap缩小14.8%，但主要来自seen拟合下降；test SSIM/LPIPS没有改善，IoU仅略高。
25k→30k新模型SSIM gap0.1888→0.2107，LPIPS gap0.17936→0.19538，仍有拟合收益无法转移到留出样本的问题。
架构和独立表冻结同时变化，不单独归因于变窄；训练/测试内容并非逐字难度匹配，属于拟合差距诊断而非严格因果证明。

## 2026-10-22：Top5/10/20@30k代表性海报

[海报](assets/posters/cohort_20261022/poster.png)：19组原图，EMA/CFG1/Heun50，同字同GT/STD/噪声。
共同Top5测试10组（每书家P50/P90）、训练5组（P50）、Top10及Top20新增书家各2组。
顶栏展示共同Top5完整测试100张/训练50张指标，逐生成格附SSIM/LPIPS/软墨量比。
不调粗细、不裁切字形；按selection.json确定选样，89张原PNG及SHA保留。只读取已有CPU/磁盘产物，未启动新推理或干预当前训练。

## 2026-10-22 17:22：X1最终正式配置为B512

复核发现编译路径显存显著低于eager；不再用eager预检否定B512。
完整compiled优化/EMA短测：B512、13.61GiB allocated、14.94GiB reserved、2.69step/s。
停止B256短跑并保留日志；正式从零运行 `assets/results/extreme48m_top20_x1/20261022-172221-Unet-flow`。
runner/训练PID1812862/1814666；实际配置、预检、日志和状态 `assets/results/extreme48m_top20_x1_launch_20261022_compiled`。
原B512/LR7e-5/warmup1500/50k调度/先30k，15.36M曝光保持不变；X2未启动。
独立书家表冻结与EMA574/574已确认；17:24正式到100步，Diff2.0921，首个编译后区间2.67step/s、显存14.67GiB、GPU100%。
初估剩余约3.2–3.5小时含评测；不是长期稳定吞吐保证。

## 2026-10-22 17:16：48.326M X1正式启动（B256）

按最新用户指令在4090从零启动X1，输出 `assets/results/extreme48m_top20_x1/20261022-171554-Unet-flow`。
runner/训练PID1810203/1810350，日志与实际配置 `assets/results/extreme48m_top20_x1_launch_20261022_b256`。
B512前向容量预检OOM；B256完整优化/EMA短测通过，峰值allocated17.90GiB、reserved18.15GiB，eager3.05step/s。
正式配置开启compile，首次编译尚无稳定吞吐；短测不作为长训ETA或架构加速结论。
LR7e-5、warmup1500、50k调度、首阶段30k；样本曝光7.68M而非B512的15.36M，不进行等曝光误比较。
独立表冻结/映射和完整EMA配对574/574已由启动日志确认；X2未启动，旧30k换表续训亦未启动。

## 2026-10-22：独立Top20 SupCon表与48.33M架构准备

确认旧FM训练更新书家表违反契约；额外10个书家的旧初始行未独立预训练。
独立预训练 `assets/writer_table_repair_top20_20261022/table_v2` 完成5000步；冻结DINOv2-small CLS，
仅train30386优化，validation200选点，test400不进入；20书家均衡跨字采样，双向table-feature SupCon及排斥项。
选中2250步，训练Top-1 50.19%、验证52.00%，异类最大余弦0.19639。
正式原表 `assets/data_cohorts/top20_20261021_v1/writer_tables_supcon_v1`：20×32单位范数，无旧生成器坐标对齐。
provenance/audit记录输入与映射身份；这不是跨作品可靠性或生成梯度有效性的证明。
首次DINO输出类型错误保留pretrain.log；修复后成功日志pretrain_v2.log，不覆盖失败记录。

下一代X1/X2已实现，均48.326M（96/192/256、每尺度8块、50 ResBlock/5 Attention），
冻结独立Writer表、无REPA/Critic/额外条件支路，STD形变off；X2加入DropPath0.08与分组WD0.05。
配置 `src/unet/configs/extreme48m`；B512、LR7e-5、warmup1500、50k调度、先停30k。
6项轻量测试通过；尚未GPU定batch或启动训练，没有速度/质量结果。
旧114M表契约同时被修正，不能把新旧比较的全部差异归因于架构。
旧Top20@30k→33k换表续训仅实现、未启动，服从用户最新“先正确训练表、准备下一架构”指令。

## 2026-10-22：112项完整结果与阶段报告

全部评测16:07:37结束（本批15:39:06启动，复用3组已有完整test），112/112项，20个checkpoint。
[阶段报告](ANALYSIS/cohort_generalization_20261022/README.md)按Top2/5/10/20四个集合组织，含训练探针/测试曲线、统一10k比较、配对区间与解释边界。
完整CSV及逐图指标本地同步至`assets/audits/cohort_all_ckpt_20261022`。
训练拟合明显强于测试，但不同线路走势不同：Top5后期回退，Top10仍改善，Top20趋平台；不能概括为全部模型越训越坏。
后续Top20书家表训练请求：先核实writer表实际是否已训练，避免将无效解冻登记为新消融。
核查完成：requires_grad=True，进入AdamW，slot step30000，exp_avg范数0.00068784，相对初始表L2变化2.18253、最大变化0.28801。
证据`assets/audits/cohort_all_ckpt_20261022/writer_table_audit.json`。不启动无效“解冻”；仅训练表需另明确，不冒充原请求。

## 2026-10-22：全部20个已保存checkpoint的test/half-seen评估

用户要求停止Top2训练并优先全面评估。已在约13k停止；仅5k/10k完整权重纳入。
另三条线各6个5k间隔checkpoint至30k；20个checkpoint×支持档位×test/train_half=112项。
队列产物：`assets/audits/cohort_all_ckpt_20261022`，固定EMA/CFG1/Heun50，batch32/VAE8。
不运行train_equal。half-seen指冻结的train_half探针，严格是训练样本，不是混合seen/unseen集合。
最新权重优先；已有30k test验证权重SHA、源CSV及原生产物完整性后复用。
每个checkpoint完整生成对应最大档位，再按嵌套样本路径、seed及逐图指标汇总所有较小档位。
结果来源为原生4–5位小数逐图CSV；保留原始PNG/完整汇总供追溯，不声称未舍入精确值。
增量结果results.csv/json，拟合差距fit_gaps.csv；计划plan.json，进度status.json。
旧10组runner标记superseded并停止，不与新队列竞争GPU，不删除旧结果。
15:42首批20项完成。最新最大档位train_half/test SSIM分别为Top20@30k 0.8302/0.5703、
Top10@30k 0.8509/0.6022、Top5@30k 0.8461/0.6009、Top2@10k 0.7825/0.6227；
LPIPS分别0.12474/0.35396、0.11558/0.33733、0.12884/0.34990、0.18333/0.33372。
属于各自不同档位的诊断，不作为跨模型横向排名；等待全部checkpoint曲线和同列比较。

## 2026-10-22：训练拟合探针（数据准备，未启动额外GPU实验）

固定训练探针 `assets/data_cohorts/top20_20261021_v1/train_probes_v1`；从冻结train.csv选取，
不改变原划分。Top20/10/5/2等量集=400/200/100/40，减半集=200/100/50/20。
以书家均衡和接近对应test的书体比例抽样，seed90，半量集及跨级集合嵌套。
8份CSV及manifest记录源/输出SHA，均严格属于相应训练集且与全局验证/测试路径不交叠。
工具 [tools/cohort_eval](tools/cohort_eval/README.md) 支持单checkpoint+任意集合，
每次输出独立身份、有效CSV、PNG、逐图指标、原生协议及汇总，不静默覆盖已有目录。
仅用于诊断拟合，不用于验证选点。3项CPU测试通过；本轮不额外启动20组train GPU评测。

## 2026-10-22：30k正式测试矩阵（10组）

前三组30k训练已完成：Top20/Top10/Top5分别于02:54:56/08:06:32/13:16:23正常结束。
Top2仍在训。测试固定checkpoint30k EMA，CFG1，Heun50；测试量Top20/10/5/2为400/200/100/40。
矩阵：模型20→测试20/10/5/2；模型10→测试10/5/2；模型5→测试5/2；模型2→测试2。
正式产物：远端 `assets/audits/cohort_matrix_30k_20261022_v2`，逐组权重文件SHA/测试CSV SHA/协议/PNG/逐图指标齐备后才写RESULT.json。
父任务自动汇总10行至results.csv/results.json，Top2未完成时等待，不使用中途checkpoint。
初次尝试目录（无v2后缀）因本地/远端公共评测接口参数差异失败，未产生有效结果，保留追溯。
修复仅在新增独立工具tools/data_cohorts/evaluate.py内，不修改在训模型或训练循环。
比较以相同测试列为准；训练内验证指标不替代本次测试结果。

## 2026-10-21 21:41：预算修订，先各30k，LR适度提高

用户将70k改为50k，再指定各组先跑30k。执行：max_steps=lr_decay_steps=50000，stop_at_step=30000，LR6e-5→7e-5（+16.7%），warmup1500不变。
旧队列在Top20约600步warmup时停止，无checkpoint；保留旧日志，各组按新配置统一从零开始。
新持久队列：`assets/data_cohorts/top20_20261021_v1/launch_30k_lr7e5`；顺序20/10/5/2。
每组首阶段15.36M曝光；30k后自动切换下一组，不自动训练到50k。
输出：`assets/results/clean_cohorts_20261021/launch_30k_lr7e5/topN_u114/`。
数据、架构、batch512、EMA及5k验证间隔不变。初估总21–23h。
下面70k登记为被本修订替代的历史计划。

## 2026-10-21：固定清洗数据，Top20 → Top10 → Top5 → Top2

- Host：4090；21:32 CST 启动持久队列，先运行Top20，其他三个排队。
- 21:35:48首段验证：150步，Diff1.8185，稳定1.63step/s、显存19.10GiB；未发生OOM/NaN。
- 初始ETA：每组纯训练11.9h，含评测/存盘约12–13h；总49–52h。仅早期估算，不是完整墙钟实测。
- 同配置从零训练：普通114M FM U-Net、channels192/384/576、对称4ResBlocks、共享Writer/Script条件。
- B512，max/lr_decay=70000，LR6e-5，warmup1500，EMA0.999，每5k保存及验证；每组35.84M曝光。
- 不启用REPA/Critic、Writer硬分离、独立Writer分支、Gate、DiC或STD形状强增强；保留软STD模糊。
- 数据：`assets/data_cohorts/top20_20261021_v1/datasets_final`；Top20/10/5/2训练量30386/23485/14069/6226。
- 相应验证量200/100/50/20，测试量400/200/100/40；跨级嵌套、全局保留组隔离、逐图固定评测噪声。
- 目标latent缓存：同根`shards_img`；STD：`std_soft_final`；接口核验：`ready_final/READY.json`。
- 完整配置、SHA、运行PID/状态/日志：同根`launch_70k`；输出：`assets/results/clean_cohorts_20261021/topN_u114_70k/`。
- 新划分结果不能直接与旧187字评测数字混排。尚未有新训练评测结果；不把历史结果填进本轮。
- 原图保留；细线边框不再扩大清洗。协议及工具见 [说明](tools/data_cohorts/README.md)。

核对日期：2026-10-18 05:15 CST（Asia/Shanghai）。本文记录所有实验身份、详细设定、数据与模型路径、评测数据结果、可视化图表链接与结论边界；实时执行快照见 [state.md](state.md)。

## 0. 覆盖范围与查找入口

**2026-10-18 19:03补充**：最新Writer CFG结果已只读核对，见本文件末尾的Writer CFG审计。
凌晨`live_snapshot_20261018.json`仅代表其记录时间，不能用作下午/晚上的实时机器状态。
19:09进程复核：4090的DiC-768到76.6k，48无GPU计算进程，见state.md；下方原53k进展是15:25历史登记。
下方阶段表保留原始观测，但“显著超越/容量无瓶颈/FiLM浅化”等强归因未经单变量或统计检验支持，不作本账本结论。
尤其DiC B768@50k与Base B512@55k的曝光量不同；65k热启臂也缺相同预算续训Control，不能直接归因结构。

| 明细层 | 覆盖范围 | 核心入口 |
| --- | --- | --- |
| 历史逐run原始总表 | 2026-10-05 原始清单 499 行，含中断、smoke、失败和配置缺失项，逐行保留 | [逐run展开](docs/experiments/ledger/early_runs.md) · [完整设置CSV](docs/experiments/all_runs_20261005.csv) · [870点评测曲线](docs/experiments/all_runs_curves_20261005.csv) |
| 本地运行产物与镜像 | 本地 126 份 resolved_config，含旧 run 镜像与配置追溯 | [设置、评测、原图逐项索引](docs/experiments/ledger/local_runs.md) |
| 最新远端在训两线 | 48 机器 U-Net 100k 对照、4090 机器 DiT 32×256；实际配置、运行身份与指标 | [实时快照](docs/experiments/ledger/live_snapshot_20261018.json) · [当前状态](state.md) |
| 历史决策与演进 | 整理前 state 全文，包括 VAE、OT、SDE、增广、停启及被替代计划 | [原state归档](docs/experiments/ledger/state_history_20261018.md) |
| 早期演进编年史 | 2026-08 起的研究阶段及代表实验完整脉络记录 | [旧编年史](docs/04_experiments/00_exhaustive_experiment_chronicle.md) |

### 账本规范与记录规则
- **唯一实验身份**：实验 ID 必须连同 host、时间戳目录和 checkpoint step 一同确认；严禁单独使用模糊简称（如 A/B/D0 等在多阶段均重复出现）。
- **指标口径严格分离**：结果分为原生浮点评测（训练器内置 evaluator）、PNG 磁盘图重评、VAE 重建、风格编码器留出测试与免训练探针；不同口径与数据划分不混合排名。
  - 原生 LPIPS 默认骨干为 **VGG**；离线历史横评通常使用 **AlexNet**；SSIM 计算亦受图像量化位深影响。
- **样本曝光量对齐**：$\text{Exposure} = \text{global\_batch} \times \text{step}$。不同 Batch Size 的模型必须基于严格的样本曝光量对齐比较，不能直接拿相同步数对比。
- **图像与断言约束**：仅对本地或远端实际存在的图片建立链接；缺失海报或测试集的项目明确标注“缺失”，不造假链接。

---

## 1. 当前执行：两条主力架构线

### 线 1：Node 48 —— U100-Deep-Sym4 从头预训练（纯U100实现，无Gate，对称深度翻倍至 114M 参数，预算 60M 样本曝光）

- **主机环境**：Node 48（RTX 4090 48GB VRAM）。
- **训练方式**：**完全从零冷启预训练（Scratch 0 step）**。
- **架构设计原则（完全回归原版 U100 纯净机制，大幅加深）**：
  - **宽通道保持**：Channels `[192, 384, 576]`，保持原版宽通道所具有的高频细节拟合带宽。
  - **残差门控彻底关闭**：`residual_cond_gate = false`，完全移除条件门控对残差流的高频衰减，保留最纯净的恒等残差通路 $x + \text{residual}$。
  - **多尺度 Adapter 关闭**：`stage_specific_cond = false`，回归单阶段全局调制。
  - **激活函数**：`act_fn = 'silu'`（保持 U100 原生平滑激活）。
  - **深度翻倍加深（Symmetrical 4+4）**：
    - Encoder 每个尺度各 **4 个 ResBlock**（共 12 块，原版为 2 块）；
    - Middle 保持 2 个 ResBlock + 1 个 Attention（576维，9头）；
    - Decoder 每个尺度各 **4 个 ResBlock**（共 12 块，原版为 2 块）；
    - 全网总 ResBlock 数量达 **26 个**，总参数量达 **114,003,044（114.00M）**。
- **训练动力学与运行指标**：
  - Global Batch = 512，总步数 **117,200 steps**（精确对齐 **60.0M 样本曝光**：$117,200 \times 512 = 60,006,400$）。
  - LR: 6e-5（Cosine 衰减至 0.1，对齐 117,200 步调度），Warmup 1500 步，EMA 0.999（interval 2）。
  - 显存占用：**20.08 GiB / 48 GiB**（显存利用率 41.8%，极其安全健康）。
  - 步速：**~1.75 step/s（896 samples/s）**，预计每 5000 步耗时 ~47 分钟。
  - 启动进程：PID **1340603**，输出目录 `assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/`，日志 `logs/train_unet_deep_sym4_114m.log`。
- **全阶段实时评估轨迹（eval200fix，N=187，Heun 50 steps，CFG=1.0）**：

| 步数 (Step) | 实际样本曝光量 | SSIM Mean | SSIM Med | LPIPS-VGG | Ink-IoU | Skel-IoU | Frag (碎笔率) | SoftInk | 近空白率 | 状态与突破描述 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5k** | 2.56M | 0.5535 | 0.5483 | 0.3758 | 0.2537 | 0.0162 | 1.9888 | 1.2654 | 0.53% | 起步阶段，骨架迅速建立 |
| **10k** | 5.12M | 0.5690 | 0.5694 | 0.3596 | 0.2805 | 0.0181 | 1.7788 | 1.2434 | 0.53% | 字形结构稳定化 |
| **15k** | 7.68M | 0.5802 | 0.5789 | 0.3501 | 0.2973 | 0.0210 | 2.2305 | 1.1683 | 0.53% | 突破 0.58 |
| **20k** | 10.24M | 0.5857 | 0.5800 | 0.3448 | 0.3102 | 0.0253 | 2.2202 | 1.1598 | 0.53% | Ink-IoU 破 0.31 |
| **25k** | 12.80M | 0.5930 | 0.5886 | 0.3380 | 0.3163 | 0.0292 | 2.0809 | 1.1193 | **0.00%** | 完全消灭空白样本 |
| **30k** | 15.36M | **0.6012** | 0.5920 | **0.3312** | 0.3231 | 0.0317 | **1.7218** | 1.0765 | 0.53% | **仅 30k 追平旧 Base 55k 峰值** |
| **35k** | 17.92M | 0.5992 | 0.5906 | 0.3318 | 0.3223 | 0.0349 | 2.0104 | 1.0979 | 0.53% | 稳定阶段 |
| **40k** | 20.48M | 0.6022 | 0.5984 | 0.3285 | 0.3293 | 0.0352 | **1.7111** | 1.0919 | **0.00%** | Frag 保持极低水平 |
| **45k** | 23.04M | 0.6031 | 0.5968 | 0.3283 | 0.3267 | 0.0374 | 2.1805 | 1.0788 | **0.00%** | 持续爬坡 |
| **50k** | 25.60M | 0.6029 | 0.5989 | 0.3275 | 0.3320 | 0.0391 | **1.7023** | 1.0937 | **0.00%** | Frag 创自身新低 1.702 |
| **55k** | 28.16M | **0.6067** | 0.6017 | **0.3251** | **0.3326** | 0.0404 | 2.0732 | 1.0666 | **0.00%** | **等曝光全面超越旧 Base 55k** |
| **60k** | 30.72M | **0.6101** | 0.6003 | **0.3223** | **0.3369** | 0.0423 | **2.0081** | 1.0711 | **0.00%** | **首次突破 0.610 大关！** |
| **65k** | 33.28M | **0.6108** | 0.6051 | **0.3213** | **0.3361** | 0.0428 | 2.1591 | 1.0593 | **0.00%** | 全面超越 DiC 三臂峰值 |
| **70k** | 35.84M | **0.6130** | **0.6048** | **0.3207** | **0.3381** | **0.0441** | **2.1782** | **1.0427** | **0.00%** | 突破历史天花板 |
| **75k** | 38.40M | **0.6132** | **0.6048** | 0.3228 | 0.3309 | 0.0443 | **2.1453** | **1.0237** | **0.00%** | 绝对极值点，进入饱和平台 |
| **80k** | 40.96M | 0.6113 | 0.6050 | 0.3220 | 0.3346 | 0.0441 | **2.1404** | 1.0550 | **0.00%** | 平台微幅震荡 (±0.001) |
| **85k** | **43.52M** | **0.6129** | 0.6019 | **0.3209** | **0.3378** | **0.0450** | **2.1527** | 1.0531 | **0.00%** | **确认收敛：各指标连续1.5万步横盘锁定** |

- **阶段性观测里程碑**：
  - **30,000 步（15.36M 曝光）**：SSIM 突破 0.6012，仅用 30k 达成旧 Base 55k 效果。
  - **60,000 步（30.72M 曝光）**：SSIM 突破 0.6101，全面超越项目历史所有架构。
  - **70,000~85,000 步（35.84M~43.52M 曝光，当前）**：SSIM 连续锁定在 0.613 极值平台，LPIPS 稳定在 0.3208，确认进入最终收敛饱和期。
  - **117,200 步（60.0M 曝光，终点）**：预计今日 20:30 完赛。

---

### 历史归档：Node 48 —— 下一代 U-Net 基模：CalliFlow-v2 从头预训练（50k 步平稳停机归档）

- **主机环境**：Node 48（RTX 4090 48GB VRAM）。
- **训练方式**：**完全从零预训练（Scratch 0 step）**，不依赖任何 DiC checkpoint 或外部风格 Teacher。
- **核心数学机制**：
  $$\boxed{ v_\theta = v_{\mathrm{base}}(z_t, t, g, s) + \Delta v_{\mathrm{writer}}(z_t, t, g, s, w) }$$
  - **Base Field $v_\text{base}$**：负责汉字内容、基本笔画连通与结构合理性。条件仅接收时间步 $t$、骨架 $g$、书体 $s$，**物理屏蔽 Writer ID**，理论极限自然收敛至无书家偏置条件期望 $\mathbb{E}[v^* \mid z_t, t, g, s]$。
  - **Writer Field $\Delta v_\text{writer}$**：多尺度分支（$8^2 \to 16^2 \to 32^2$），窄通道卷积读取主干特征，经 DiC 条件门控自适应调制，零初始化输出头。理论极限自然收敛至书家速度残差 $\mathbb{E}[v^* \mid z_t, t, g, s, w] - \mathbb{E}[v^* \mid z_t, t, g, s]$。
  - **单次前向极速 CFG**：$v_{\mathrm{CFG}} = v_{\mathrm{base}} + \alpha \Delta v_{\mathrm{writer}}$，单步前向直接获得 CFG 速度，免除两倍网络计算开销。
- **训练目标（闭式双 FM 监督）**：
  $$\boxed{ \mathcal{L} = \|v_{\mathrm{base}} + \Delta v_{\mathrm{writer}} - v^*\|^2 + 0.5 \|v_{\mathrm{base}} - v^*\|^2 }$$
- **架构参数与规模**：
  - 骨干：CNN Hourglass U-Net，Channels `[128, 256, 384]`，Encoder 4 ResBlocks/尺度，Decoder 4 ResBlocks/尺度，Middle 3 ResBlocks + SDPA。
  - 激活函数：GELU；内存格式：Channels-last；Torch Compile 开启。
  - 总参数量：**63.23M**（主干 60.92M + 多尺度 Writer 分支 2.30M）。
- **训练动力学与运行指标**：
  - Global Batch = 512，总步数 120,000 steps（总曝光量 61.44M 样本）。
  - LR: 6e-5（Cosine 衰减至 0.1），Warmup 1500 步，EMA 0.999。
  - 实时吞吐：**2.83 step/s（1,449 samples/s）**，显存稳健驻留 **16.42 GiB**。
  - 启动进程：PID 714015，输出目录 `assets/results/calliflow_v2_120k/`，日志 `logs/train_calliflow_v2_120k.log`。
- **阶段性观测里程碑**：
  - **25k 步（12.8M 曝光）**：检查内容轮廓收敛性与 Writer 残差模长 $\|\Delta v_{\mathrm{writer}}\|$ 是否有效脱离 0。
  - **75k 步（38.4M 曝光）**：与 Node 4090 DiC-768@50k 做严格等曝光量对标（38.4M 样本）。
- **50k 步完赛实测数据表（eval200fix，N=187，Heun 50 steps，CFG=1.0）**：

| 步数 (step) | 样本曝光量 | SSIM Mean | SSIM Med | LPIPS-VGG | Ink-IoU | Frag (碎笔) | SoftInk | 近空白率 | 状态与特征总结 |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **5k** | 2.56M | 0.5418 | 0.5421 | 0.3854 | 0.2443 | 1.8591 | 1.3323 | 0.53% | 初始内容与骨架建立 |
| **10k** | 5.12M | 0.5531 | 0.5565 | 0.3734 | 0.2664 | 2.4660 | 1.2969 | 0.53% | 笔画骨架清晰化 |
| **15k** | 7.68M | 0.5635 | 0.5656 | 0.3648 | 0.2734 | 2.5806 | 1.2081 | 1.07% | 局部字形连通收敛 |
| **20k** | 10.24M | 0.5672 | 0.5677 | 0.3625 | 0.2796 | 2.5170 | 1.1998 | 0.53% | 结构稳定性增强 |
| **25k** | 12.80M | 0.5725 | 0.5723 | 0.3598 | 0.2852 | 3.0183 | 1.1554 | 0.00% | 完全消灭空白样本 |
| **30k** | 15.36M | 0.5803 | 0.5796 | 0.3535 | 0.2890 | 2.2991 | 1.1347 | 0.53% | SSIM 突破 0.58 |
| **35k** | 17.92M | 0.5808 | 0.5799 | 0.3526 | 0.2875 | 2.3493 | 1.1180 | 0.53% | 笔势形态逐步沉淀 |
| **40k** | 20.48M | 0.5816 | 0.5806 | 0.3527 | 0.2872 | 2.5552 | 1.1076 | 0.00% | 墨迹逐步逼近基准 |
| **45k** | 23.04M | 0.5863 | 0.5860 | 0.3483 | 0.2869 | 2.3389 | 1.0631 | 0.53% | SSIM 跳升，LPIPS 破 0.35 |
| **50k** | **25.60M** | **0.5860** | **0.5947** | **0.3475** | **0.2905** | **2.0987** | **1.0710** | **0.00%** | **完赛封存：中位数 0.5947，Frag 创历史最低 2.0987** |

- **执行结项与停机**：已于 2026-10-19 00:49 响应用户指令在 50k 节点（实跑 52,550 步）完成优雅停止。显存全部释放，已落盘 `0050000.pt`（966MB）、海报 `eval200fix_poster.png` 与评测明细。

### 历史参考线：Node 48 —— DiC 三臂对照串行队列（50k Checkpoint 热启至 75k 步，已全量完赛）

- **主机环境**：Node 48（RTX 4090 48GB VRAM）。
- **统一热启起点**：U100-Base 的 50,000 步 Checkpoint（`0050000.pt`，Base 55k 峰值 SSIM 0.6017 / LPIPS 0.3317）。
- **执行状态**：**三臂已全部跑满 75,000 步完赛归档**（耗时 7 小时 49 分钟，无损串行流转）。
- **全三臂完整评估轨迹（eval200fix，N=187，Heun 50 steps，CFG=1.0）**：

| 实验臂与架构设置 | 步数 (step) | 样本曝光量 | SSIM Mean | SSIM Med | LPIPS-VGG | Ink-IoU | Frag (碎笔) | SoftInk | 状态与结论 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **U100-Base (旧基线)** | 55k | 28.16M | **0.6017** | 0.6057 | 0.3317 | 0.3192 | 2.679 | 1.062 | 旧主线收敛峰值 |
| —————————————— | ——— | ———— | ———— | ———— | ———— | ———— | ———— | ———— | ——————————— |
| **Arm 1: DiC_residual_gate**<br>(75.04M, 门控参数28个) | 55k | 28.16M | 0.5995 | 0.5996 | 0.3338 | 0.3173 | 2.653 | 1.075 | 门控自适应起步 |
| | 60k | 30.72M | 0.6023 | 0.6039 | 0.3323 | 0.3192 | 2.844 | 1.061 | 跨过基线 |
| | **65k** | **33.28M** | **0.6054** | 0.5996 | **0.3287** | **0.3214** | **2.461** | 1.045 | **全线超越基线** |
| | 70k | 35.84M | 0.6021 | 0.6054 | 0.3318 | 0.3163 | 2.637 | 1.056 | 平台期震荡 |
| | 75k | 38.40M | 0.6044 | 0.6054 | 0.3298 | 0.3180 | 2.458 | 1.040 | 完赛收敛归档 |
| —————————————— | ——— | ———— | ———— | ———— | ———— | ———— | ———— | ———— | ——————————— |
| **Arm 2: DiC_stage_adapter**<br>(72.88M, 多尺度轻量残差) | 55k | 28.16M | 0.6002 | 0.6003 | 0.3330 | 0.3171 | 2.654 | 1.070 | 平滑起步 |
| | 60k | 30.72M | 0.6032 | 0.6042 | 0.3315 | 0.3210 | 2.972 | 1.061 | 稳定超越基线 |
| | **65k** | **33.28M** | **0.6059** | 0.6041 | **0.3286** | **0.3214** | **2.368** | 1.043 | **综合最优（断笔最少）** |
| | 70k | 35.84M | 0.6024 | 0.6036 | 0.3319 | 0.3163 | 2.695 | 1.054 | 平台期震荡 |
| | 75k | 38.40M | 0.6043 | 0.6055 | 0.3303 | 0.3174 | 2.445 | 1.040 | 完赛收敛归档 |
| —————————————— | ——— | ———— | ———— | ———— | ———— | ———— | ———— | ———— | ——————————— |
| **Arm 3: DiC_stage_indep**<br>(72.80M, 多尺度独立投影) | 55k | 28.16M | 0.6003 | 0.6006 | 0.3334 | 0.3165 | 2.683 | 1.067 | 平滑起步 |
| | 60k | 30.72M | 0.6026 | 0.6015 | 0.3324 | 0.3196 | 2.885 | 1.060 | 超越基线 |
| | **65k** | **33.28M** | **0.6061** | 0.6040 | **0.3288** | **0.3212** | **2.377** | 1.040 | **SSIM 峰值冠军 (0.6061)** |
| | 70k | 35.84M | 0.6025 | 0.6038 | 0.3322 | 0.3159 | 2.733 | 1.050 | 平台期震荡 |
| | 75k | 38.40M | 0.6045 | 0.6059 | 0.3300 | 0.3181 | 2.454 | 1.039 | 完赛收敛归档 |

---

### 线 2：Node 4090 —— U100-DiC-768（从头训练，全功能 DiC 架构 U-Net，目标 100k 步）

- **主机环境**：Node 4090（RTX 4090 24GB VRAM）。
- **当前运行 PID**：`1738543`；日志：`scratch/unet_dic_100k_b768_20261018_01/train.log`。
- **配置文件**：`src/unet/configs/connectivity/DiC_U100_b768.json`。
- **模型设定**：75.83M 全功能 DiC 架构 U-Net，全开 `residual_cond_gate` + `stage_specific_cond` (adapter) + `act_fn='gelu'`。
- **训练超参**：Global Batch 768，LR 7e-5（cosine 调度至 100k，warmup 1500，min_ratio 0.1），WD 0.02，EMA decay 0.999（interval 2）。STD-shape strong 增广，Latent 预加载。
- **进度与状态**：已推进至 **Step 53,000+ / 100,000**，速度稳定在 **1.71 step/s（1,313 samples/s）**，显存占用 22.24 GiB / 24.56 GiB，Diff 损失降至 **0.0942**。
- **50,000 步关键收敛评测（eval200fix，N=187，Heun 50 steps，CFG=1.0）**：

| 指标 | U100-Base @ 55k (基线峰值) | U100-DiC-768 @ 50k (从头训练) | 增益 / 改善幅度 |
| :--- | :---: | :---: | :--- |
| **SSIM Mean** | 0.6017 | **0.6047** | **+0.0030**（显著超越基线收敛上限） |
| **SSIM Median** | 0.6057 | **0.6020** | 分布重心整体右移 |
| **LPIPS-VGG** | 0.3317 | **0.3294** | **-0.0023**（深度感知噪点与毛边大幅降低） |
| **Ink-IoU** | 0.3192 | **0.3234** | **+0.0042**（**历史最高墨迹交并比**） |
| **Frag (断笔连通域)** | 2.679 | **2.301** | **-0.378（-14.1%）**（**历史最低碎笔率**） |
| **SoftInk Ratio** | 1.062 | **1.046** | 更趋近理想 1.000 丰满度，无晕墨浮肿 |
| **MSE Loss** | 0.789 | **0.771** | 重建均方误差显著优化 |
| **训练 Diff 损失** | ~0.119 | **0.0942** | 深度优化突破，大批次+GELU势能显著 |

#### V92A 32×256 有效原生 EMA 评测指标（eval200fix，N=187，Heun 50 steps，CFG=1.0）

| 步数 (step) | 样本曝光量 | SSIM | LPIPS-VGG | Ink-IoU | Frag | 产物与核验来源 |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **10k** | 1.92M | 0.50550 | 0.42831 | 0.18342 | 1.9360 | `eval_auto_10000.json`、`0010000.pt.done` |
| **20k** | 3.84M | 0.51862 | 0.41829 | 0.18731 | 1.9541 | `eval_auto_20000.json`、`0020000.pt.done` |
| **30k** | 5.76M | 0.52187 | 0.41470 | 0.18821 | 1.9723 | `eval_auto_30000.json`、`0030000.pt.done` |
| **40k** | 7.68M | 0.52395 | 0.41245 | 0.18887 | 1.9907 | `eval_auto_40000.json`、`0040000.pt.done` |
| **50k** | 9.60M | 0.52331 | 0.41261 | 0.19033 | 1.9787 | `eval_auto_50000.json`、`0050000.pt.done` |
| **60k** | 11.52M | 0.52230 | 0.41336 | 0.18985 | 2.0269 | `eval_auto_60000.json`、`0060000.pt.done` |
| **70k** | 13.44M | **0.51804** | **0.41522** | **0.19235** | **2.1095** | `eval_auto_70000.json`、`0070000.pt.done`（最新核验） |

- **关联文档**：[V92A 32L实施配置说明](src/train/configs/v92_32l/README.md) · [实时快照数据](docs/experiments/ledger/live_snapshot_20261018.json)。

---

## 2. 骨架 / 两阶段 / Moyi / 字表实验（8月—10月初）

本阶段全部历史 run 见 [499行明细](docs/experiments/ledger/early_runs.md)，核心发展脉络如下：

| 系列代号 | 架构与核心设定 | 主要数据与产物路径 | 关键结果与结论边界 | 可视化图表链接 |
| :--- | :--- | :--- | :--- | :--- |
| **v1–v9** | DiT/DDPM，离散 char 与 writer 标签条件注入，后引入 REPA 表征辅助。 | `data/top10_style23`，旧 checkpoints | 验证了字形生成可行性，但在未见字符上发生模式崩塌（Seen 记忆不等于泛化）。 | [旧纪元报告](docs/04_experiments/00_exhaustive_experiment_chronicle.md) |
| **v10a/v10b** | 引入 STD 骨架空间条件，41 书家开集探索，长训至 200k。 | `assets/runs/v10a/` | Seen 与 Strict 协议不同，Strict 测试下 SSIM 约 0.5058。 | [历史全景图](docs/04_experiments/imgs/historical_strict_common_poster.png) |
| **v18/v21 (SkelNet)** | 两阶段：标准骨架→书家骨架（SkelNet）→最终图像。引入 latent/pixel 骨架回归。 | `assets/results/v21_skelnet_200k/` | Latent 骨架损失下降无法保证细笔连通与拓扑准确，两阶段级联误差放大，路线被封存。 | [v21实际Strict海报](assets/results/v21_skelnet_200k/posters/strict_poster.png) · [两阶段封存归档](docs/experiments/2026-10-03-twostage-archive.md) |
| **v46** | 彻底放弃中间骨架生成，转向单阶段直接生成图像。 | `docs/experiments/2026-10-03-v46-singlestage-experiment-summary.md` | 证实单阶段端到端训练在内容保真度上显著优于两阶段。 | [v46实验报告](docs/experiments/2026-10-03-v46-singlestage-experiment-summary.md) |
| **Recurrent Refine** | 递归骨架迭代细化（Recurrent Skeleton Refinement）。 | `docs/106moyi/recurrent_skel_refine/` | 迭代修正未带来稳定收益，反而增加了推理延时。 | [逐点指标CSV](docs/106moyi/recurrent_skel_refine/recurrent_refine_metrics.csv) · [原图海报](docs/106moyi/recurrent_skel_refine/recurrent_refine_poster.png) |
| **Moyi Top10 RF** | 12 通道联合条件（Image+Canny+Skel）vs 纯 4 通道 Image。 | `docs/106moyi/` | 12ch@50k SSIM 0.5935；4ch@80k SSIM 0.6085。证明增加输入通道不如精细化潜空间条件。 | [Moyi复现报告](docs/experiments/2026-10-03-moyi-reproduction-report.md) · [汇总README](docs/106moyi/README.md) |
| **v54–v68 (字表主线)** | 三因素字表、冻结/随机初始化、条件路由 `[2, 4, 5, 6]`、增强与 C2OT。 | `exp-std/csv/train_top10_aug_sym.csv`，`assets/runs/v68_family/` | **v68_aug_sp_c2ot** 达到历史最佳 SSIM **0.6149**@200k。奠定了现代主干训练基准。 | [17行主面原表](docs/experiments/mainline_20261009.csv) · [阶段回顾报告](docs/experiments/2026-10-13-vae-and-related-changes-report.md) |

---

## 3. VAE、OT、REPA、SDE 与 STD 回归（v70–v84）

| 实验/系列 | 核心设定与改动 | 产物与数据路径 | 实验指标与结论判定 | 关联报告与图表 |
| :--- | :--- | :--- | :--- | :--- |
| **v70** | STD 骨架经 VAE 编码后替代离散字表，改用空间潜特征注入，B288。 | `assets/results/v70_aug_sp_stdskel_c2ot/` | 最佳 SSIM 0.5535@65k，低于同时期 v68 (0.5833)，显露骨架连续注入的对齐困难。 | [SDE与骨架分析](docs/experiments/2026-10-12-v68-v70-sde-role-and-free-lunch.md) |
| **DINO Calli-VAE** | 自研 Calli-VAE：L1+KL+DINO_struct/style，移除判别器。 | `docs/experiments/2026-10-08-calli-vae-dino-repa-pretraining.md` | 高频带保持率从 SD 的 0.816 降至 0.783，重建存在平滑退化。 | [12500步重建图](docs/experiments/calli_vae_visuals/recon_step_12500.png) |
| **v74 (Posterior)** | 生成时从 sample 改为 posterior mode。 | `ANALYSIS/v68_vs_v80_sde_math_diagnosis_20261012.md` | 23 个测试点落后，ΔSSIM -0.008，ΔLPIPS +0.032，证伪单纯切换 mode 收益。 | [数学诊断报告](ANALYSIS/v68_vs_v80_sde_math_diagnosis_20261012.md) |
| **物理 VAE (v79)** | 基于 SD 底座重训：L1+LPIPS+Laplacian+Topband+PatchGAN。 | `docs/experiments/2026-10-13-vae-and-related-changes-report.md` | 187 字重建 PSNR 从 30.56 升至 34.04，高频保持率达 0.901。 | [物理VAE阶段总结](docs/experiments/2026-10-13-vae-and-related-changes-report.md) |
| **v80** | 物理 VAE 下关闭 OT，B288，200k 总步数。 | `docs/experiments/2026-10-12-v80-progress-200k.md` | 190k 原生 SSIM **0.6168**，LPIPS-VGG **0.3247**。成为物理 VAE 路线最高分。 | [训练曲线报告](docs/experiments/2026-10-12-v80-progress-200k.md) |
| **v80@65k 推理调参** | 相同 EMA 与噪声下，CFG 权重 (1.0~2.5) 与 SDE 噪声比调优。 | `ANALYSIS/v80_65k_inference_recommended.json` | 确认 CFG=1.5 带来最优视觉感知折中，纯 SDE 额外收益未确立。 | [调参报告](docs/experiments/2026-10-12-v80-65k-inference-tuning.md) |
| **v83A** | STD-soft zero-init 结合 hybrid 字表条件。 | `ANALYSIS/v83A_eval_contract_fix_20261012.md` | 修复采样评测合同后近白图从 33/187 降至 1/187，撤回早期“白化”结论。 | [合同修复分析](ANALYSIS/v83A_eval_contract_fix_20261012.md) · [v83A报告](docs/v83A_std_soft_report.md) |
| **v84A** | 标准 SD-VAE，纯 STD-soft 空间条件注入，200k 完整基准。 | `assets/results/v84A_std_eval200fix/` | 原生 SSIM 0.5846，LPIPS-VGG 0.3551，IoU 0.2840。成为后续风格后训练的标准成熟基模。 | [v84A海报](ANALYSIS/v84A_std_eval200fix_poster.png) · [结构图](ANALYSIS/v84A_std_eval200fix_struct.png) |

#### 增广实验（A1–A4，48 主机，B512×100k，OT 关闭，SD Latent）
| 增广分支 | 策略细节 | SSIM | LPIPS (alex) | Ink-IoU | Frag | 原始文件路径 |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **A1 morphE** | 形态学腐蚀/膨胀随机组合 | 0.6045 | 0.3304 | 0.3328 | 2.507 | `assets/results/A1_morphE/` |
| **A2 elastic** | 弹性变形 (Elastic Distortion) | 0.6044 | 0.3304 | 0.3381 | 2.412 | `assets/results/A2_elastic/` |
| **A3 grid** | 局部网格形变 (Grid Distortion) | 0.6031 | 0.3313 | 0.3338 | 2.242 | `assets/results/A3_grid/` |
| **A4 affine** | 经典仿射变换 (Affine) | 0.5970 | 0.3371 | 0.3187 | 2.430 | `assets/results/A4_affine/` |
- **对比图表**：[七臂增强全景图](ANALYSIS/aug_family_100k.png) · [数据配置清单](docs/experiments/ledger/state_history_20261018.md)。

---

## 4. 风格后训练、Writer 表与 v88 实验

以成熟冻结/热启基模 `v84A@200k` 为起点，探究风格调制的有效形式：

| 实验代号 | 架构与调制设定 | 训练配置与预算 | 实验指标与评测表现 | 核心产物与核验文档 |
| :--- | :--- | :--- | :--- | :--- |
| **v85A** | 在 Block 8/10/11 引入 Rank-32 Style AdaLN。 | 续训 30k 步，LR 5e-5 | SSIM 0.5862，LPIPS-VGG 0.3537。独立 Writer 判别器分类未见显著收益。 | [v85诊断纠正](docs/v85-diagnostics-2026-10-14.md) · [对照图](ANALYSIS/v85_poster_comparison.png) |
| **v85E** | 引入 Writer CE + AwaySTD 辅助损失。 | 续训 15k/30k 步，w_writer=0.02 | 15k 独立 Top-1 达 55.08%（基线 52.41%），但高步数下笔画边缘出现伪影。 | [核验图表](ANALYSIS/v85_verified_20261014/poster_01.png) |
| **v86 Full-Fixed (A100/B25/B50/C/D)** | 修复映射污染，进行 Full-Batch 真实反事实配对监督。 | B112~224，每 4 步双分支，预算 10k~20k | C 组成对反事实达到 SSIM 0.5855，LPIPS 0.35285，IoU 0.29592，初步显露反事实监督的稳定性。 | [后训练总结报告](docs/experiments/2026-10-15-posttraining-and-v88-review.md) · [36文件归档清单](archive/experiments/v86_20261015/README.md) |
| **v88** | Writer×Glyph 交叉调制（Rank-32，Block 3/7），从零预训练。 | B288，主 LR 5e-5，Writer LR 5e-6，从零 170k | 收尾 SSIM 0.5776，LPIPS-VGG 0.4189，IoU 0.2811。证实从零学习双线性交叉会损害主干几何收敛。 | [v88审查报告](docs/experiments/2026-10-15-posttraining-and-v88-review.md) · [内部探针数据](assets/v88_full_eval_20261016/internal_probes.json) |

---

## 5. 风格编码器、统计先验与梯度审计

独立表征训练（不包含生成器更新），用于确立书家风格评估标尺：

| 实验/模型 | 任务与结构设定 | 数据规模与划分 | 评测表现与关键指标 | 产物路径与可视化 |
| :--- | :--- | :--- | :--- | :--- |
| **v89 Style Encoder** | SD Latent mode×0.18215 + STD，ResNet-18 主干，CE+0.3SupCon+0.1STD。 | 26,002 图，23 真迹槽 + 3 STD 槽 | Best@3k：字符留出测试集 CE 30.07%，来源留出 13.64%，训练集 99.77%。 | [完整报告](docs/experiments/2026-10-16-v89-style-encoder-complete.md) · [产物目录](assets/style_encoder_v89/latest_20261016_01/) |
| **Pixel D0/D1** | 纯像素灰度输入（不含 STD），D0 纯 CE，D1 加跨字对比。参数 3.81M。 | B2048，10k/臂，LR 1e-4，seed 89 | 字符留出 2AFC 从 56.08% 提升至 61.04%，来源留出从 56.12% 提升至 60.71%。字符 mAP 0.5354。 | [对照文件](assets/writer_pixel_d0d1_20261016/comparison.json) · [Manifest](assets/writer_pixel_d0d1_20261016/data_manifest.json) |
| **Latent D0/D1** | 复用 v84A 潜特征缓存输入，参数 1.54M，跨字权 0.5，margin 0.1。 | B2048，10k/臂，LR 1e-4 | 字符留出 2AFC 从 57.50% 提升至 60.45%，来源留出达 57.14%。字符 mAP 0.5033。 | [对比报告](assets/writer_latent_d0d1_20261016/comparison.json) · [分片桥接](assets/writer_latent_d0d1_20261016/exact_shard_bridge.json) |
| **D2 (Dual Branch)** | 空间几何分支 + 局部笔法统计分支，CE + 跨字。参数 1.66M。 | B2048，完成 10k 步 | 终点验证集 mAP **0.55364**，匹配 2AFC **64.59%**。 | [D2配置](assets/writer_geometry_d2d3_20261016/D2_config.json) · [验证日志](assets/writer_geometry_d2d3_20261016/D2_validation.jsonl) |
| **D3 (SDF Aux)** | D2 主干基础上增加弱 SDF 几何辅助损失（权重 0.01）。 | B2048，训练至 8380 步停止 | 8k 验证点 mAP **0.55502**，匹配 2AFC **65.44%**。 | [D3配置](assets/writer_geometry_d2d3_20261016/D3_benchmark_config.json) |
| **Latent D1 梯度审计** | 探究判别器直接提供生成梯度的可行性，步长 0.1%~3%。 | v84A 固定 80 例，8 字符 | 0.1% 步长仅工程 pilot 通过，高步长破坏潜空间流场，明确未获主干训练验收。 | [梯度审计报告](assets/latent_d1_prototype_gradient_audit_20261016/results.json) · [梯度图表](assets/latent_d1_prototype_gradient_audit_20261016/gradient_summary.png) |

---

## 6. v90/v92 Scaling 与【20×256 vs 20×384 三项免训练对比实验】

### 6.1 主干宽度与深度 Scaling 实验集
- `v90A_20L_stdstrong`：深度 20 层，通道 384，4 heads，参数量 **53.98M**。Batch 192，完成 200k 步。
- `v90B_20L_w256_b288`：深度 20 层，收窄通道至 256，4 heads，参数量 **24.43M**。Batch 288，推进至 159.8k 步。
- `v90_24L_384`：深度加深至 24 层，通道 384，参数量 **65.67M**。在 Node 48 推进至 130k 步（已停）。
- `v92A_32L_w256`：深度跨越至 32 层，通道保持 256，参数量 **38.81M**。Batch 192，调度 200k（当前正在 Node 4090 执行）。

### 6.2 20×256 vs 20×384 三项免训练对比实验（严格样本曝光对齐）
为了裁定“在 20L 基础上是继续加宽还是加深”，并在决定是否直接启动 32L×256 之前，我们在 Node 4090 上执行了三项完全免训练的离线探针评估。
- **严格曝光对齐基准**：
  - **34.56M 曝光点（100% 精确对齐）**：`v90B_20L_w256` @ 120,000 步（$120\text{k}\times 288 = 34.56\text{M}$） vs `v90A_20L_stdstrong` @ 180,000 步（$180\text{k}\times 192 = 34.56\text{M}$）。
  - **~38.40M 曝光终点对齐**：`v90A` @ 200,000 步（$38.40\text{M}$） vs `v90B` @ 130,000 步（$37.44\text{M}$）与 140,000 步（$40.32\text{M}$）。

#### 实验 1：Train/Val FM 分桶（固定样本与噪声）
- **评测协议**：评估 187 个严格隔离字形及配对训练集真迹，固定 2 组全局噪声种子，计算不同时间步 $t \in [0.10, 0.25, 0.40, 0.60, 0.80, 0.95]$ 下的 Flow Matching 向量场均方误差（MSE）。
- **产物文件**：`assets/capacity_audit_20261017/exposure_matched_exact/report.json`。

| 模型与步点 | 样本曝光量 | 数据集划分 | $t=0.10$ | $t=0.25$ | $t=0.40$ | $t=0.60$ | $t=0.80$ | $t=0.95$ | 全局平均 MSE | 泛化 Gap (Val - Train) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **20×384 @ 180k** | **34.56M** | **Train**<br>**Val** | 0.31443<br>0.31383 | 0.24817<br>0.25106 | 0.25575<br>0.26383 | 0.27449<br>0.30011 | 0.37300<br>0.42282 | 0.55948<br>0.59557 | 0.33755<br>**0.35787** | **+0.02032** *(显著过拟合)* |
| **20×256 @ 120k** | **34.56M** | **Train**<br>**Val** | 0.33874<br>0.33593 | 0.26300<br>0.26329 | 0.26783<br>0.27048 | 0.29812<br>0.30562 | 0.43621<br>0.43943 | 0.62223<br>0.61044 | 0.37102<br>**0.37087** | **-0.00015** *(泛化稳健)* |
| **20×384 @ 200k** | 38.40M | Val | 0.31335 | 0.25086 | 0.26381 | 0.30050 | 0.42444 | 0.59769 | **0.35844** | +0.02289 |
| **20×256 @ 130k** | 37.44M | Val | 0.33452 | 0.26244 | 0.26991 | 0.30479 | 0.43703 | 0.60790 | **0.36943** | +0.00065 |
| **20×256 @ 140k** | 40.32M | Val | 0.33335 | 0.26180 | 0.26955 | 0.30438 | 0.43465 | 0.60693 | **0.36845** | +0.00120 |

- **结论**：在中低噪声几何骨架决定区（$t \le 0.40$），256 宽与 384 宽仅差 0.006~0.012；且 256 宽无过拟合现象，证明缩窄至 256 并未遭遇容量瓶颈。

#### 实验 2：Writer 容量测试（独立判别器分类与反事实置换）
- **评测协议**：
  1. 采用第三方独立风格判别器（23 书家原型 + 3 STD 槽）对模型在 `eval200fix`（187 张图）上的生成样本进行全量推断；
  2. 采用 Target-Aware 2×2 条件置换审计（AA/AB/BA/BB），测量置换目标书家后主干与输出头引起的 FM 误差增量。
- **产物文件**：`assets/capacity_audit_20261017/writer_disc_eval.json` 与 `assets/capacity_audit_20261017/writer_head_swap_eval/report.json`。

| 评估对象 | 曝光量 | 独立判别器 Top-1 | 独立判别器 Top-5 | std-margin | writer-margin | 主干置换 $\Delta L_\text{backbone}$ | 输出头置换 $\Delta L_\text{head}$ | 置换致误率 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GT 真迹 (天花板)** | — | **71.66%** | 93.05% | 0.6155 | +0.1737 | — | — | — |
| **20×384 @ 180k** | **34.56M** | **51.87%** | 85.03% | 0.5399 | **+0.0633** | **+0.01447** | +0.00033 | **78.88%** |
| **20×256 @ 120k** | **34.56M** | **41.71%** | 78.07% | 0.4664 | **-0.0180** | **+0.00809** | +0.00026 | **76.20%** |
| **20×384 @ 200k** | 38.40M | **55.08%** | 87.70% | 0.5489 | **+0.0684** | **+0.01460** | +0.00035 | **78.61%** |
| **20×256 @ 130k** | 37.44M | **44.39%** | 80.21% | 0.4813 | **-0.0273** | **+0.00881** | +0.00028 | **77.54%** |

- **结论**：两模型对书家置换均具备近 80% 的误差敏感性，排除了模式坍塌；但 256 宽独立判别器 Top-1 发生 ~10% 的跌落，主干置换误差增量减半，证实主干通道收窄导致书家 FiLM 调制存在“浅化”现象。

#### 实验 3：复杂度分层测试（笔画密度与连通块复杂度）
- **评测协议**：根据 187 个真迹字的墨迹像素密度（GT Mass）与连通块数（Connected Components）联合打分，严格等分为低复杂度（Simple, N=62）、中复杂度（Medium, N=63）、高复杂度（Complex, N=62）三档，计算细骨架拓扑召回率（Thin Skeleton Recall）、错误粘连数（Merged Components）与笔画断裂缺失数（Missing Components）。
- **产物文件**：`assets/capacity_audit_20261017/complexity_stratification.json`。

| 模型配置 | 复杂度分级 | 细骨架拓扑召回率 (内容保真) | 错误笔画粘连 (Merged 处) | 笔画断裂缺失 (Missing 处) | SSIM | Ink-IoU |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **20×384 @ 180k**<br>*(宽模型基线)* | **低复杂度 (Simple)**<br>**中复杂度 (Medium)**<br>**高复杂度 (Complex)**<br>*整体均值 (Overall)* | 37.09%<br>38.09%<br>**34.20%**<br>36.47% | 0.56<br>0.97<br>**1.37**<br>0.97 | 1.32<br>3.68<br>**4.55**<br>3.19 | 0.6272<br>0.5820<br>0.5164<br>0.5752 | 0.2338<br>0.2661<br>0.2921<br>0.2640 |
| **20×256 @ 120k**<br>*(窄模型等曝光)* | **低复杂度 (Simple)**<br>**中复杂度 (Medium)**<br>**高复杂度 (Complex)**<br>*整体均值 (Overall)* | **38.06%** *(+1.0%)*<br>35.71% *(-2.4%)*<br>**34.24%** *(+0.0%)*<br>36.00% | 0.63<br>0.98<br>**1.40** *(+0.03)*<br>1.01 | 1.44<br>3.76<br>**4.66** *(+0.11)*<br>3.29 | 0.5989<br>0.5657<br>0.5017<br>0.5555 | 0.2118<br>0.2433<br>0.2670<br>0.2407 |
| **20×384 @ 200k** | **高复杂度 (Complex)** | **35.23%** | **1.34** | **4.65** | 0.5175 | 0.2929 |
| **20×256 @ 130k** | **高复杂度 (Complex)** | **34.26%** | **1.48** | **4.56** | 0.5033 | 0.2685 |

- **结论与架构定型决策**：
  1. **证伪“窄模型在复杂字上崩溃”**：高复杂度下 256 宽的骨架召回率（34.24%）与 384 宽（34.20%）完全打平，粘连与断笔相差不足 0.1 处。这证明字形结构瓶颈不在通道宽度，而在感受野与多尺度拓扑传递。
  2. **决策执行**：触发规则 1 & 4，彻底跳过 24L×384 中间过渡方案，下一代架构直接定型为 **32L×256**（深度增加 60%，显存友好且保持 ~1,600 img/s 吞吐）；同时触发规则 3，将在 32L 结构中对书家风格条件注入通道进行专项参数扩容。

---

## 7. Infra、失效记录与归档缺口

| 记录项目 | 实际核验证据 | 影响与纠正边界 |
| :--- | :--- | :--- |
| **REPA 异步预取** | 20×384/B192 实测步速从 4.25 提升至 5.12 step/s（+20.5%）。 | [测试数据](docs/infra-1015-measurements.md)。受 CPU-GPU 拷贝带宽约束，不作全模型外推。 |
| **引导图整图编译** | TorchInductor 静态整图编译短测获得 +5.9% 吞吐提升。 | [运行时边界](docs/infra/runtime_20261017.md)。不改动模型收敛动力学。 |
| **评测管线加速** | 评测单轮耗时从 178~194s 优化至 73~87s。 | [重构报告](docs/refactor-report-2026-10-16.md)。保持全量采样精度不变。 |
| **U-Net 早期 EMA 缺陷** | `torch.compile` 前缀导致权重未能配对，旧 5k/10k/15k 自动 EMA 评测实为随机初始化。 | [审计报告](ANALYSIS/unet_20261018.md) · [EMA修复验收](ANALYSIS/unet_ema_fix_20261018.md)。旧 raw 指标可复核，作废旧 EMA。 |
| **V83 白化评测错误** | 早期评测因丢失 STD 软掩码合同将模型误判为“全白化”。 | [合同修复分析](ANALYSIS/v83A_eval_contract_fix_20261012.md)。纠正后证实模型结构完好。 |
| **v85 错用基线图** | v85 历史横评脚本曾误将 V80 图片作为 v84A 对照图读取。 | [纠正报告](docs/v85-diagnostics-2026-10-14.md)。作废基于错误图像的一切对比结论。 |
| **SDE 漂移实现缺陷** | 早期 SDE 采样器存在系数与扩散项不平衡。 | [数学诊断报告](ANALYSIS/v68_vs_v80_sde_math_diagnosis_20261012.md)。已修正漂移补偿项。 |
| **历史缺失快照目录** | 本地 `phase_review_20261013`、`recent_vae_20261013`、`posttraining_20261015` 已归档未留存。 | 严格不使用伪造文件链接；以对应归档总结报告与保留的全局 poster 为准。 |

---

## 8. 后续维护规则

1. **唯一事实源**：
   - 动态训练状态、实时 PID 与机器队列由 `state.md` 承载；
   - 所有实验的静态归档、超参设定、多维度指标与分析结论由本文件 `exp.md` 统一收录维护。
2. **新增实验准入条件**：
   - 必须登记：唯一 ID、执行主机、Checkpoints 与 Run 真实路径、模型架构、参数量、损失与优化超参、严格样本曝光量、原生评测数据、可视化图表路径及明确的结论边界。
3. **决策留痕**：
   - 严禁删除被替代的旧方案记录；任何架构调整（如放弃 24L、收窄宽度 256、扩展至 32L）必须保留完整的实测对比数据与推导逻辑。

## 9. Writer CFG审计（2026-10-18 19:03，只读）

原始产物：`48:/home/ds/Workspace/DiT/assets/results/cfg_sweep/`；
`full_cfg_eval.json`、`full_cfg_per_image.json`、`writer_swap_results.json`及逐CFG的`protocol.json/metrics.json/DONE.json`。
[已取回的数值、协议和路径快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。
本轮未重新采样、未更新模型、未启动Style Loss或改动训练队列。

### 同DiC-768@50k的Writer CFG：现有输出配对

Writer分数在有效181图上计算；Macro为按书家分组后等权平均Top1，Micro为全部有效图平均。
内容指标为187图，原生LPIPS-VGG。两CFG记录seed0、Heun50、SDE0，checkpoint SHA一致。

| 指标 | CFG1.0 | CFG1.6 |
| --- | ---: | ---: |
| Writer Macro Top1 | 55.0348% | 65.3177% |
| Writer Micro Top1 | 55.8011%（101/181） | 66.2983%（120/181） |
| Writer target margin | 0.062589 | 0.161343 |
| SSIM | 0.604340 | 0.605192 |
| LPIPS-VGG | 0.329501 | 0.327011 |
| Ink-IoU | 0.323462 | 0.337220 |
| Frag | 2.333244 | 2.059439 |
| SoftInk ratio | 1.049766 | 1.102819 |

GT的同评测器Macro约70.06%、Micro71.82%；这是评测器在该批数据上的表现，不是生成器风格的理论上限。
Gate@75k CFG1.4的Macro64.8524%、Micro66.2983%，与DiC CFG1.6差0.4653个Macro百分点，未做配对区间，不能宣布风格排名显著。
现有Writer指标基于编码器特征/训练原型余弦，不等同于直接读取CE分类头；也不能把识别提升直接等同真实笔法提升。

### 已发现的身份/可比性问题

- 当前`run_cfg_sweep_sampling.py`普通分支调用`load(..., use_ema=False)`，DiC专用分支加载`ck['ema']`；Base/Gate与DiC的raw/EMA读取路径不一致。
- 查到的Base与DiC协议均标`ema=true`，且`conditions_sha256`为空。这说明已有协议记录不足以排除身份误标；不能因为有DONE就断言权重/条件审计已完成。
- 旧Base@55k Micro59.12%与本轮54.70%差8/181，尚未定位完整原因。raw/EMA、batch/seed、代码版本、预处理和原型库都需要逐项核对，不能先归咎随机seed。
- `Baseline_55k_results.json`还留有初次失败扫描（margin NaN、内容指标0）；不进入有效比较，使用上方full_cfg文件也仍须完成身份复核。
- 同模型CFG扫描比跨架构排名更有证据价值，但仍只覆盖seed0；不能据此称三seed稳定free lunch。

### Swap与后续准入

DiC@50k CFG1.4的Swap目标Top1为47%，异书家翻转45.56%，OCR代理内容保持53%；
同模型CFG1.0分别40%、38.89%、51%。翻转改善约6/90，尚无多seed/分组区间。
未查到同协议GT/STD OCR基线，53%不解释为真实汉字正确率；本轮Swap没有CFG1.6，不能外推。

优先顺序：先修正并核验raw/EMA/条件身份及CFG1.0复现，再在同checkpoint/预处理/初始噪声协议下比较CFG1.2–2.0、至少三seed；
随后才考虑分时段Writer CFG、墨量匹配及独立表征/人工盲评。以上是研究建议，不是已排队任务。
Writer-only AdaGN可作为独立训练对照，Style Loss暂不因本次分数上涨自动启动。
CFG>1需要条件与Writer-null两个速度场，几乎翻倍网络评估工作量，不称推理计算免费。
