# state.md — 当前实验状态

## 2026-10-27：统一训练、权重与双机日计划入口

持续进度以[training.md](training.md)为准，权重登记见[ckpts.md](ckpts.md)，当天双机计划见[1027.md](1027.md)。00:49核查：48 J0正常计步；4090 S0日志停在1,350、GPU 0%，进程存在，已明确标记停滞待排查，不能仅凭status.json中的training视为正常。以下条目保留为历史执行记录。

## 2026-10-27：48启动Deep85 Decoder三组各5k

48串行J0（FM，Decoder冻结）→J1（FM+Image，Decoder冻结）→J2（FM+Image+GT重建，Decoder训练），三组同Deep85 EMA起点、原VAE和Writer语义映射，B512、compile开启、无梯度检查点。J2已先完成正式前50步并保存，区间约0.51step/s、峰值39.812GiB、冻结编码指纹通过，后续从50恢复至5k。队列已进入J0；从00:44起保守估计剩余6–7h。图像梯度权重重新校准为0.0212917，约FM的10%。[文档与路径](ANALYSIS/style_repr_injection_matrix_20261026/decoder_deep85_20261027.md)。代码提交da2ce77。

## 2026-10-27：4090恢复compile，启动S组随机初始化50k串行

新入口误将compile禁用，已恢复torch.compile(default)，保留BF16、channels_last、fused AdamW、缓存latent/STD及异步checkpoint，不开梯度检查点。4090队列S0→S1→S4→S6，各50k，B512/micro512，完全随机初始化，非P1/P2续训。S0已完成50步，首次编译约97秒，峰值allocated19.166GiB；第1至50步约1.6step/s。四组含开销暂估37–39h。48及Deep85起点的J组本次未启动。[阶段文档](ANALYSIS/style_repr_injection_matrix_20261026/compile_cold50k_20261027.md)。实现提交34e06e4。

## 2026-10-26 23:54：澄清从头训练，停止旧S/J续训队列

旧计划沿用S从P1、J从P2，和用户本轮要求的生成器从头训练不一致。4090 S0已SIGTERM，在569步原子保存checkpoint后退出，串行状态stopped/75；48未启动训练，相关传输已停止。保留全部产物，不将该段用于从零训练比较。后续需改随机初始化、重设对应校准/训练配方；无需P1/P2权重。48已有原图与VAE，先核查缓存路径/版本复用。旧续训ETA不作为从头训练的收敛预算。


## 2026-10-26 续实施：S/J短预检完成，串行已写，保持不长跑

[阶段文档第7节](ANALYSIS/style_repr_injection_matrix_20261026/README.md#7-续实施最快安全batch真实恢复串行方案)。首个阶段提交`ae14347`；之后按“安全且最快”复用候选，普通FM默认值未变。48 J四臂梯度关系通过，选图像有效16/micro8，J3 40.279GiB、0.4264step/s；FP32/B512累积、S2-large容量、真实P1保存恢复以及J3真实GT校准/恢复/配套Decoder导出逐张量核对均已完成。
两端`plan_final/{4090,48}/run_stage.sh`默认dry；S阶梯10k/29k/43.5k，J为4k/11.5k/19k，含阶段等待及8h余量预测63.95h，非长训保证。仍未封存两端实际资产ready；未启动S/J正式长训，未恢复旧队列。完整生成质量与独立Writer评价需要正式阶段产物，当前没有新质量结论。


## 2026-10-26 最新：S/J风格消融准备，尚未启动正式长训

[逐步记录及原始测量](ANALYSIS/style_repr_injection_matrix_20261026/README.md)。新范围为S0–S8+S2-large、独立J0–J3；旧20格矩阵/R4预训练暂缓。两端有效B512、禁止梯度检查点；最新batch准则为安全且最快，不必填满20/40GiB。
普通FM四类batch搜索两端已完成，4090旧12格复核及真实P1@10k EMA/Top10短测通过，约1.033step/s、20.38GiB reserved。J3真实Decoder+LPIPS梯度链通过，但micro16占46.14GiB，待micro8短测。B256 BF16累积有约0.48%梯度/6%更新差，不能声称严格等价；FP32/B512核查未完。
本次现场两端无style_matrix运行进程；48旧`testing R3I2`是中止时遗留状态。当前仅提交实现与证据，J四臂完整预检、校准、配套Decoder评测、串行脚本和64h预算仍待补。以下旧队列快照不代表当前仍在执行。


## 2026-10-25 23:00：Qwen P2-A 1,000 张定向合成与两级漏斗验收完成（触发停点，封存路线）

**执行总账**：Top10 书家 × 50 训练缺失汉字 × 2 seeds = 1,000 张定向合成，在 48（4090 48G）用 P2-A 4k 权重 8.2 分钟全量生成完成（2.01 图/秒，32.3 GiB）。
- **两级筛选漏斗实测数据**：
  - **总合成样本**：1,000 张（100% 结构正确无错字，STD 锁死骨架生效）；
  - **第一关（物理墨墨与连通质检）**：390 张合格（39.0%），淘汰 610 张（387 张中空虚墨 `hollow>0.08`、354 张飞白碎裂 `frag>2.5`、32 张微小对齐偏差）；
  - **第二关（独立判别器真实性核验）**：**书家标签保留 75 张（7.5%）**。何绍基 24、赵孟頫 14、褚遂良 10、欧阳询 10、柳公权 9、颜真卿 3、米芾 3、苏轼 2、王羲之 0、文徵明 0；
  - **执行降级（Degraded to Null-Writer）**：**315 张（31.5%）**。字形正确墨色饱满，但书家特异笔致无法通过 26 类留出判别器，按计划剥离书家标签，重标为通用书法素材；
- **Gate 2 / Gate 3 战略判定（全面触发退出机制，关闭 Qwen 扩量）**：
  - **有效书家标注率过低（7.5%）**：1,000 张中仅得 75 张有效书家配对（仅占 U-Net 训练集的 0.3%），行书大家（王羲之、文徵明）为 0 张；
  - **画质倒挂**：U-Net 原生终态（P1@10k + P2@20k）LPIPS 达 **.3497**、Frag 仅 **2.18**；而 Qwen P2-A 破碎率达 35.4%、LPIPS 为 **.4485**。将 Qwen 合成样本注入 U-Net 反会引入噪声监督，稀释真实墨质；
  - **决策落地**：**正式停止 Qwen 路线的后续扩量训练与注入**，模型资产封存，算力全部归还给 U-Net 主干的蒸馏与 Writer CFG。
- **产物索引**：
  - 报告：`ANALYSIS/qwen_synth_1000_acceptance_20261025.md`；
  - 48 远端：`/home/ds/Workspace/xy/qwen_synth_1000/`（1000 图及 `funnel_summary.json`、`funnel_details.json`）；
  - 本地归档：`assets/qwen_synth_1000/`。

## 2026-10-25 21:39：P2 strong/mild收尾对照已在4090启动

A strong 5k → B mild 5k串行；**两者独立从P2@20k EMA初始化**，不是A后继续训B。
114M、B512、主干LR5e-6/条件LR2e-5、pair_fraction .3（152/512）、同seed90；2.5k/5k存评，STD eval sigma0、CFG1。
增强配比时钟延续P2 20k；strong为20/25/40/15%，mild压成20/80/0/0%。未改变Loss/网络/GT/书家表冻结开关。
最新A约750步，1.62 step/s、19.11GiB；B已排队，预计剩余约1小时40分钟（含评测）。
目录：4090 `assets/p2_std_consolidation_20261025/`，监督进程PID3547284。
GT核对完成：39/120组合可配对；墨量/笔宽P50方向一致87.3%/70.0%，但仅10个共享字符×书体组。
W5/8细、W6粗有少量GT支持且生成幅度偏大，不能概括为Writer总体方向错误；W9无配对GT。
文档及GT/GEN图：`ANALYSIS/p2_std_consolidation_20261025.md`；不蒸馏、不自动替换默认模型。

## 2026-10-25：4090 P2条件利用审计与CFG验收

**已完成，无训练在跑**：CFG三seed、分时间窗、120张swap、墨量控制及全部100图读图完成。
三seed CFG1→1.6：LPIPS .349015→.346177、IoU .280953→.289914、probe28.33%→39.67%。
但probe在GT仅24%/重建GT32%，墨量控制后一个seed的margin CI跨0；未通过可靠风格教师验收，不启动蒸馏。
CFG保留推理候选，不替换P2默认权重；最终记录及五页图：`ANALYSIS/p2_execution_20261025.md`。

已固定P2@20k EMA，完成原生val100 sigma=0基线：SSIM .592304 / LPIPS .349359 / IoU .280905 / Frag 2.3031。
FM正确/同书体错误/null、CFG多seed/分时间引导、Writer-swap及GT域校准均已完成。
产物（4090）：`assets/p2_audit_20261025/`；执行记录：`ANALYSIS/p2_execution_20261025.md`。
发现并修复：离线嵌套结果汇总为空、旧swap时间方向/尺度/PNG维度错误、旧Writer probe重复除255。
旧probe无效，新probe另存`writer_probe_fixed.pt`；不把旧分数或单一probe增益作为蒸馏依据。**尚未启动蒸馏**。

对下方历史结论勘误：P1/P2 B512 vs直FT B256，不能称等曝光预算；在线val100实际sigma=.65，不能与sigma0混排。
本批val100没有全局未见字符，仅68图为未见Writer×书体×字符组合；不得宣称严格字符留出泛化。

## 2026-10-25 19:40：P1/P2 writer-aware 验收完成；路线锁定简单 U-Net

**验收结论（假设成立）**：同预算下 P1 混合阶段明确优于直 FT——
20k：P1@10k+P2@10k **.5864/.3535/.2725/Frag 2.02** vs 直 FT .5800/.3603/.2632/2.27；
30k：P1@10k+P2@20k **.5908/.3497/.2814/2.18** vs 直 FT 延长 .5829/.3581/.2638/1.91；
**P2@20k（30k 预算）超过直 FT@50k 终态（.5844/.3558/.2660）**。test200 .5895/.3466/.2874 与 val100 一致。
离线 eval200fix σ.65 .5834/.3488/.2918。P1 曲线 .5616→.5664；P2 曲线 .5864→.5908（无回落）。
**路线锁定**：114M + STD + Script + Writer Embedding，普通 FM；P0（kxl-100k 75k EMA 复用）→ P1 混合 10k → P2 配对专精 20k；
不加 Reference/CrossAttention/Writer×STD 模块/风格码本。
依据：风格结构诊断系列（`ANALYSIS/style_structure_diagnosis_20261025.md`）——
跨字线性可读 43.6%、无字形条件调制（四方零增益）、M1/M2 交互 ΔR²≈0、d_eff95=7、Writer Embedding 容量足够。
**偏差记录**：P0 为 kxl-100k 替代（40 万池 ckpt 不存在）；STD 强增强保留（协议连续性）。
**待办**：writer swap 干预测试两度于 LPIPS 加载后无声死亡（无 OOM/traceback），脚本 `exp_writer_swap.py` 就绪待修；
40 万池 P0 补训为可选增强。P0 分解等纯分析期间 writer_slot 已按诊断结论关闭（15k ckpt 留存）。
产物：`assets/u100_p1p2_20261025/`（4090）全套；文档 `ANALYSIS/p1p2_writer_aware_20261025.md`、
`ANALYSIS/style_structure_diagnosis_20261025.md`、`tools/u100_pretrain/README.md`（工具总索引）。

## 2026-10-25 22:35：P2-A STD条件LoRA 4000步圆满收官（Gate 1 关键指标落地）

按用户决策树执行。**P0b**（不训练）：书家笔法描述Prompt 46对配对——SSIM .5708→.5678微降，
文字描述无法注入笔法；真实组合版writer-swap（23组合）：换seed .5696 < 换书家 .7441，
seed噪声主导结论在真实组合下成立。强度扫描0.75-1.0最优，部署1.0。
**P1 书家残差**（20k冻结双适配器+rank8残差，颜真卿/褚遂良各3k步，LR5e-5，B8@256无ckpt）：
未见字上残差 vs 共享——颜真卿 +.0051 SSIM/−.0043 LPIPS，褚遂良 −.0021/+.0020，
两位方向相反、量级在n=10噪声内→**书家专精LoRA无显著收益，瓶颈在条件信息而非梯度冲突**。
**P2-A STD-conditioned LoRA**（按用户路线A）：Turbo+20k冻结双适配器+新rank16，
条件=目标字TTF字形（字体级联新渲染256 RGBA σ.65）经Qwen3-VL原生图像条件+
prompt结构指令；全量23,485训练，B4×accum4，ckpt on，3.20s/步，峰值36.15GiB，**4,000步全部跑完并完成全自动评测**。
- **val100 演进与终值对比**：
  - **Prompt-only 20k 基线**：SSIM **.5853** / LPIPS **.4458** / Ink-IoU **.2855**
  - **P2-A Step 0**（刚挂 STD，未训新 Adapter）：SSIM **.5954** / LPIPS .4656 / Ink-IoU .2179（结构瞬时锁死，墨迹苍白薄弱）
  - **P2-A Step 2000**：SSIM **.5883** / LPIPS .4498 / Ink-IoU **.2635**（墨迹快速恢复，LPIPS 下降）
  - **P2-A Step 4000（终态）**：SSIM **.5858** / LPIPS **.4485** / Ink-IoU **.2687**（墨迹进一步修复，LPIPS 逼近基线，行书 Ink 达 .2952 反超基线）
- **产物与权重留档**：
  - 检查点权重 `pytorch_lora_weights.safetensors`（65MB）及 `state.pt`（257MB）在 48 侧 `/home/ds/Workspace/xy/qwen-lora-ref-20261025/checkpoints/0004000/` 完好封存。
  - 评测指标与全套 100 张生成图在 `evaluations/0004000/`，指标 JSON 已同步落地本地 `assets/qwen_lora_ref_20261025/`。
- **Gate 1 判定**：结构锚定完全成立（0%错字/多笔少画），墨迹与感知损失基本抹平初始退化；但整体指标亦证实单纯延长 Qwen LoRA 训练无法产生超越基线的质变增益，**判定停止 Qwen 主体训练，将 P2-A 4k 权重作为离线合成教师模型**。

## 2026-10-25 07:40：Qwen双通道A/B/C臂评测完成——增量微弱，不足以证立

同预算同seed同5k子集（85k→90k段），eval200fix σ.65 EMA，n=187：

| 臂 | step | SSIM | LPIPS | Ink-IoU | skel | frag |
|---|---|---|---|---|---|---|
| BASE（85k起点） | 85000 | .5812 | .35903 | .24548 | .01747 | 1.87 |
| A TTF-only | 90000 | .5850 | .35805 | .24628 | .01758 | 2.80 |
| **C TTF+Qwen双通道** | 90000 | **.5864** | .35809 | .23966 | .01764 | 3.04 |
| C@87500 | 87500 | .5851 | .35766 | .24495 | .01803 | 2.78 |

判定：**C−A ≈ +.0014 SSIM / +.0000 LPIPS / −.0066 Ink**——在n=187噪声量级内，无有效增量；
且C的Ink反而略降、frag上升（2.80→3.04）。A/B/C口径下Qwen字形通道对114M U-Net**近冗余**
（与D2"seed噪声>书家信号"一致：Qwen条件提供的主要是TTF已覆盖的结构+少量墨迹先验）。
B臂（Qwen替换）未训练（A/C已足以回答主问题）。按用户口径：不满足"C显著优于A"，不扩全量。
过程：训练/评测链共修12处新旧脱节（详见tools/qwen_condition/README.md）；
A@90000权威数字取自arm_a_0090000目录（首轮单跑），链式重跑的该目录被protocol跳过逻辑写空头表。

## 2026-10-25 03:20：Qwen双通道A/C臂训练已拉起（48，Deep-Sym4 85k底座）

按用户方案C执行：`train_top10_aug_sym` 5k子集+eval200fix，Qwen字形存储4,493键
（LoRA20k生成，与TTF std_soft逐位同协议）；三臂配置`tools/qwen_condition/arm_{a,b,c}.json`
（A=TTF对照/B=替换/C=双通道dropout0.15），同seed90同数据序同5k步，全局曝光时钟调度
（117200余弦，85k→90k段LR 2.0e-5→1.3e-5）。std_stem 1→2通道零初始化扩展，
warm-start严格验证334键/ch0逐位相等；30步冒烟通过；期间修复老unet入口与演进后
shared trainer的8处兼容问题（详见[tools/qwen_condition/README.md](tools/qwen_condition/README.md)）。
**arm A实测训练中**：step85900，Diff.0366，1.73步/s，20.1G；A→C串行，预计~2.2h全部完成。
**0085000.pt已下载到本地** `assets/u100_deepsym4_114m/0085000.pt`（1.74GB）；
推ModelScope需用户提供账号token，未做。

## 2026-10-25 01:30：Qwen-as-Condition三项诊断完成，48具备A/B/C全部资产

**资产核实（48）**：U100 Deep-Sym4 114M ckpt在`assets/results/unet_deep_sym4_114m_60m/.../checkpoints/{0050000,0075000,0085000}.pt`；
src/unet代码在；TTF字体在（tools/fonts + configs/std_soft_fonts.json + build_std_soft_glyphs.py），
std_soft存储已被磁盘瘦身清除但**已重建**：`data/std_soft_glyphs_top10_v1`共8,061字形、覆盖100%缺0
（楷simkai/行STXINGKA/隶SIMLI级联）。4090另有今日新U100 FT25k（1.8G）未用。

**P0 Writer-swap**（30字×10书家，seed90固定只换名，LPIPS质心分类，候选楷10/行10/隶3）：
底模5/30（≈机会率5.3）、LoRA 8/30（超机会率但弱）、**GT上限仅10/30**（分类器本身噪声大，结论方向性）。
**强度扫描**（val100×scale 0/.5/.75/1/1.25）：SSIM .6039/.5857/.5857/.5853/.5805，
LPIPS .5072/.4508/.4453/.4458/.4530，Ink .1945/.2796/.2854/.2855/.2764——**0.75–1.0最优，1.0保持**。

**D1**（test200，TTF=std_soft 64→256 σ0.65）：TTF vs GT SSIM .5827/LPIPS .6809/Ink .1972/skel .0136；
Qwen vs GT .5821/.4365/.2827/.0182——**Qwen条件携带的GT相关信息显著多于TTF**（Ink+43%、LPIPS−36%），
SSIM打平；Qwen vs TTF SSIM .6144（两者字形面差异适中）。[D1海报](assets/qwen_lora_top10_20261024/d1_poster.jpg)。
**D2**（6字×3书家×3seed）：同书家换seed SSIM .5626 < 固定seed换书家 .7221——
**seed噪声主导、书家控制信号弱**（与swap分类一致）。[D2网格](assets/qwen_lora_top10_20261024/d2_grid.jpg)。
**D3**（train200 vs test200同协议）：train .5866/.4344/.3016 vs test .5821/.4365/.2827，
**差距≤.019——无seen/test条件质量悬崖**，可安全作为训练条件。

**判定**：D1✓（Qwen有TTF没有的墨迹/笔画信息）+D3✓（条件质量均匀）+D2△（书家保真弱，
收益可能主要来自"更真实的毛笔先验"而非书家特异结体）→ **方案C（TTF+Qwen双通道）值得做**，
按用户设计：Conv2d(1→2)第二通道零初始化、10–20%Qwen条件drop、首轮2–5k子集5k步、A/B/C同预算同种子。
产物：`p0_writer_swap/`、`p1_diagnostics/`（48）；本地assets同步；脚本`p0_experiments.py`、`condition_diagnostics.py`。
P1B参考图训练（train_ref.py已写好、root已搭）按用户转向搁置未启动。
## 2026-10-24 22:50：非Turbo Top10 LoRA完赛，test200与Turbo打平

`qwen-lora-top10-base-20261024` 20k步完成（1.221s/步、44.41GiB、零重启）。验证趋势4k→20k：
SSIM .5605→.5677、LPIPS .4439→.4242（16k后平台）、Ink .2723→.2848，与Turbo同形，基本收敛。
test200同协议（256/seed90/20k）：非Turbo(40步) SSIM .57577/LPIPS .42988/Ink .27911
vs Turbo(8步) .58209/.43648/.28272——**基本打平各占半项**；非Turbo LPIPS优势.007需5倍推理步数，
实用首选仍Turbo。分书家/分书体排序两轮高度一致。
[海报](assets/qwen_lora_top10_base_20261024/test200_poster_20k.jpg)、[指标](assets/qwen_lora_top10_base_20261024/test200_summary_20k.json)、
[总结文档§9](ANALYSIS/qwen_lora_top10_20261024.md)。

## 2026-10-24 17:00：U100 通用预训练→Top10 后训练管线完赛（4090 空闲）

**结果（FT 25k EMA）**：val100(σ0) SSIM **.5817**/LPIPS .3579/IoU .2654/Frag 2.12/零空白；
test200(σ0) .5826/.3527/.2804；eval200fix(σ.65) .5771/.3536/.2790。
**val100 刷新全部课程/reference 尝试**（此前最优 reference 25k .5775/.3612/2.72；generic48 .5625）。
**未达 U100 从零线**（eval200fix .6132，差 .036；对方 60M 曝光/18.6h vs 本线 25.6M/10.7h，预算与数据构成均不同）。
FT 曲线：0步 .4916 → 5k .5712（+0.08 一步到位）→ 25k .5817，15k 后仍缓升，25k 内无过拟合回落。
预训练 75k val100（行0-9未训练，σ.65）.4291→.4974@55k→.495@75k 平台；Frag 20.6→5.3。
目录 `assets/u100_pretrain_20261024`：15+5 检查点、status.json 全程、离线评测三口径。
修复记录：transfer_shared 相对导入层级（损失1.5h）、eval_offline 缺 inference_mode 致 OOM、eval200fix char_id 扩展映射。
候选下一步（未启动）：FT 延长 25k→50k（2.2h，val100 未平台）；预训练延长 75k→150k；seen 探针补测拟合差距。

## 2026-10-24 12:00：48参考图推理实验（负结果）+非Turbo Top10 LoRA启动

**参考图实验**：test200每图加2张同书家同书体异字训练集参考（该组合最高频2字，`pipe(prompt,image=[...])`
经Qwen3-VL进条件），其余协议与基线一致（256/8步/seed90）。20k LoRA配对结果：
SSIM .5821→.5705（71/200改善）、LPIPS .4365→.4453（95/200改善）、Ink .2827→.2576（64/200）——**全降**。
底模诊断（无LoRA同协议）：base无参考 .6032/.5027/.1948 vs base+参考 .55946/.5035/.21356，
配对Δ SSIM −.0437/LPIPS +.0008/Ink +.0188——底模同样净负。
**结论：参考图路线在此协议下无效，且非LoRA分布偏移所致**；要救此线需"带参考训练"而非推理侧注入。
产物：`ref_eval_20k/`、`base_ref_diag/`（各含200图+metrics+summary）；脚本
[ref_eval.py](tools/qwen_lora/ref_eval.py)、[base_ref_diag.py](tools/qwen_lora/base_ref_diag.py)。

**非Turbo Top10 LoRA**：48新目录`/home/ds/Workspace/xy/qwen-lora-top10-base-20261024`，runner3050857。
底模=已校验非Turbo`Qwen/Qwen-Image-2.1`（qwen-lora-20261023/base）；同top10数据/同配方
（256/B12/无ckpt/LR1e-4余弦/20k步/每4k保存评测），唯一差异：验证用原生40步（eval_steps=40），
VAE后验+文本17,453键按非Turbo组件全部重编码（不复用Turbo缓存）。
实测step210已开训：1.212s/步、峰值43.9GiB、loss0.096；预计~7.5h完成（含5次40步评测）。

## 2026-10-24 06:05：4090 转入 U100 纯通用预训练→Top10 后训练管线（当前主任务）

按用户决策：reference 线停止，改为 **100k 纯通用池预训练 75k 步 → Top10 后训练 25k 步**（ckpt/评测每5k，drop_writer 统一 0.25）。
管线入口 `python -m tools.u100_pretrain.run --steps 75000`，目录 `assets/u100_pretrain_20261024`，runner PID 1834751。
阶段全部自检通过：14,999 字 256 raster 与 std_bbox64 缓存**逐位一致**；strong 形变 bank 49s 生成（guard 回退 3.8%）；
书家分组 259 行（0-9 Top10 预留 SupCon 初始化、10-257 通用 248 组、258 OTHER 3,604 行）；benchmark 3.28 step/s / 9.69GiB。
**114,011,012 参数 Deep-Sym4（192/384/576×4+4、silu、无gate）、B256、LR6e-5 余弦、warmup1500、writer_mode=shared 端到端、
strong STD 增强、clean σ.65、EMA.999**。预训练只用通用 100,263 行（Top10 行 0-9 不参与，纯 ImageNet 范式）。
内联评测 val100（Top10 留出，σ.65，行 0-9 未训练——兼测读取器对 SupCon 方向的互操作性）。
实测 3.21 step/s：预训练 75k ≈ 6.5h+评测 ≈ **13:20 CST 完赛**；自动接 FT 25k（init 75k EMA、行 0-9 可训练、
20% 通用回放锚定、LR1.5e-5、eval σ0、0步基线）≈ 2.5h；随后离线评 eval200fix(σ.65)/test200(σ0)/val100(σ0)。
设计文档与论证：[ANALYSIS/u100_generic_pretrain_design_20261024.md](ANALYSIS/u100_generic_pretrain_design_20261024.md)、
[管线说明](tools/u100_pretrain/README.md)。48 机器未动（Qwen LoRA 在跑）。

## 2026-10-24 05:55：Reference Top10 30k 续训按指令停止

runner/trainer SIGTERM 于 25,375/30,000 步，日志与 15k/20k/25k 检查点+评测保留。
续训段指标：15k SSIM .5661/LPIPS .3702/IoU .2511 → 20k .5713/.3662/.2542 → 25k .5775/.3612/.2631（斜率 ≈ +0.001/k，弱但未完全平台）。
不再续训；reference 单向量注入路线存档。同前 0步 .5586 / 10k .5651。

## 2026-10-24 20:50更新：Top10纯Prompt LoRA切256+关检查点，快5.2倍

用户指出512对256数据是浪费。res_probe.py底模9字×256/512对照GT：SSIM .588/.597、LPIPS .507/.501、
Ink .184/.192，VLM确认256九字全部结构完整无崩坏→正式训练切256（GT原生分辨率）。
关梯度检查点（省1/3重算）+256（token 336→144，文本token不随分辨率缩）后OOM两次：
512/B8无ckpt 46.5GiB、256/B16无ckpt 46.9GiB；最终**256/B12/无ckpt+expandable_segments，43.9GiB稳定**。
实测1.215s/步=0.101s/样本（原512/B8/ckpt 0.524，加速5.2倍）。用户加码：**20,000步**（240k样本≈10.2epoch），
**每4000步保存+评测**（4k/8k/12k/16k/20k共5个检查点），预计总~7h。256 latents 23,785张4分钟重编码
（512版存latents512.pt），文本缓存复用。日志train_256_b12_20k.log，trainer1870908。
**10:19 CST完成**：1.216s/步、峰值44.07GiB、全程零重启；检查点共962MB。
23张验证（seed90、原生8步）趋势：SSIM .5693→.5750（20k最优）、LPIPS .4520→.4327单调改善、
Ink-IoU .2653→.2891；对比底模256探针（.588/.507/.184，样本集不同仅方向性）：LPIPS/Ink明显更好、
SSIM略低。隶书槽位收益最大（何绍基隶Ink .27→.42、赵孟頫隶 .28→.42）；褚遂良Ink偏弱(.14-.16)、
赵孟頫楷LPIPS恶化(.547→.571)。每槽n=1仅方向性；test200未动；尚未做视觉审查与test评测。
**test200全量评测完成**（20k LoRA、256、8步、seed90、逐图固定）：总体SSIM .58209/LPIPS .43648/Ink-IoU .28272；
分书体：楷.5949/.4071/.2846、行.5751/.4620/.2724、隶.5603/.4344/.3305。
书家SSIM排序：褚遂良.629>欧阳询.626>王羲之.594>苏轼.584>柳公权.578>赵孟頫.576>米芾.570>
何绍基.559>文徵明.559>颜真卿.546；验证槽里偏弱的褚遂良在test上反而最好（验证n=1的槽位信号不可靠）。
[海报](assets/qwen_lora_top10_20261024/test200_poster_20k.jpg)、[全量对照拼图](assets/qwen_lora_top10_20261024/test200_contact_sheet_20k.jpg)（200对GT｜生成红框，分书家）、
[指标](assets/qwen_lora_top10_20261024/test200_summary_20k.json)；
远端test_eval_20k/{images 200张,metrics.json,summary.json,poster.jpg}；脚本tools/qwen_lora/test_eval.py
（--poster-only可从metrics重建海报；评测加载LoRA需fp32转换与训练一致，已修复严格校验）。
**训练总结文档**：[ANALYSIS/qwen_lora_top10_20261024.md](ANALYSIS/qwen_lora_top10_20261024.md)
（方法/配方实测链/收敛判定：基本收敛合理停点，LPIPS尾部仍有~.003-.01边际；含test200全部分解表）。

## 2026-10-24：Top10全量（10书家×3书体）纯Prompt条件LoRA在48启动

48目录`/home/ds/Workspace/xy/qwen-lora-top10-20261024`，直接trainer1809041（无runner，缓存完成后绕过run.py）。
数据根`qwen-lora-top10-data-20261024`：top10划分CSV经本地中转同步48（与4090逐字节SHA一致），GT图48本有38,583张。
train23485（楷9363/行12285/隶1837）/validation100/test200，10书家真实组合，不做阈值清洗、不补造标签。
按用户要求删掉书家表32D→4token与书体embedding特殊条件链路：书家/书体/字全部写入文字prompt
（`书法家：{名}。书体：{书}书。请用…书写汉字「{字}」…`），冻结Qwen文本编码器；
writer/script标签仅用于书家均衡采样与评测选图。只训q/k/v/out rank16 LoRA 16.78M，无条件适配器/dropout。
文本缓存按（字,书家,书体）17,453键、VAE后验23,785张（6.2GB）全部新编码；Turbo底模SHA校验通过。
batch实测：B8 4.20s/步=0.524s/样本，B32 16.58s/步=0.518s/样本——4090算力打满，加大batch不提速；
最终B32×2500步（80k样本≈3.4epoch，与B8×10k同曝光同墙钟~12h），LR1e-4单段余弦至1e-5、warmup100。
每500步保存+评测（10书家×3书体各1张，Turbo原生8步、seed90）。实测峰值33.9GiB、GPU100%、loss有限、
LoRA初始梯度0.236非零。首检查点step500约2.3h后；此前无质量结论，可用resume延长步数。

## 2026-10-24：Top10 Reference 后训练由10k完整续至总30k

用户要求继续。4090新目录`assets/reference_top10_30k_20261024`，runner1832859。
从`reference_top10_20261024/runs/20261024-010707-Unet-flow/checkpoints/0010000.pt`
完整恢复raw/EMA/Adam/scheduler/RNG，不重置EMA，不重编码参考特征。
总步数30k（额外20k），每5k保存评测；原23,485训练/100验证/200测试、参考协议、B256与所有loss不变。
余弦总调度10k→30k，baseLR2e-5不变，恢复点LR从旧终点2e-6回到约1.56e-5；调度变化明确留档。
旧0/5k/10k产物保留，新checkpoint为15/20/25/30k。按既有1.92step/s估计额外约3小时。
trainer1832923，run `runs/20261024-025517-Unet-flow`，已实际跑到10025。
日志确认完整RNG/Adam恢复、EMA未重置，LR1.560911e-5；FM.1541、SupCon约3.91，有效anchor256。

## 2026-10-24：非Turbo已停，显式书家/书体Prompt重评完成

按用户要求SIGTERM停止48的runner1642156与trainer1645256，末次270/6000步；未到500步首次保存，无非Turbo LoRA检查点。
只改文本Prompt，前缀为`书法家：{calligrapher}。书体：{script}。请用该书法家的{script}风格书写。`；原字符/白底/黑墨要求保留。
同6字验证、seed90：非Turbo未微调基模40步，Turbo LoRA@6000原生8步且保留已训练条件token。
非Turbo旧→新：SSIM .50776→.51621，LPIPS .54429→.53641，Ink-IoU .19006→.22447。
Turbo@6000旧→新：SSIM .49015→.49239，LPIPS .45414→.45130，Ink-IoU .29938→.31781。
48结果`/home/ds/Workspace/xy/qwen-prompt-explicit-20261024`，18张新图完成；Turbo旧组复用原6张，不声称加载路径逐位等价。
图片显示收益不均匀，部分字主要变粗，智永等仍不贴近GT；不是完整测试或书家风格成功证明。未重新启动训练。
[海报](assets/qwen_prompt_explicit_20261024/poster.jpg)；[指标](assets/qwen_prompt_explicit_20261024/summary.json)；[实现](tools/qwen_lora/prompt_eval.py)。

## 2026-10-24：非Turbo基模同预算LoRA启动记录（已停止，见上）

48独立目录`/home/ds/Workspace/xy/qwen-lora-base-20261024`，runner1642156，trainer1645256。
非Turbo全部权重SHA校验通过，重新缓存2786张原生VAE posterior和1694字符条件；不加载Turbo LoRA/投影/缓存。
正式首步已完成，LoRA/条件梯度非零，书家表冻结，B8、512、rank16，目标6000步，每500步保存评估。
base6000.json逐步复现Turbo实际2k+4k LR历史：原1e-4余弦2k，100步回升5e-5，再余弦至1e-5。
相同2730/18/38划分、相同6字验证与seed；非Turbo40步采样对比Turbo8步，采样预算差异不能忽略。
初步峰值18.4GiB，长稳吞吐/质量待后续检查点；没有自动扩训练预算或启动其他GPU实验。

## 2026-10-24：Reference FM 接 Top10 全量后训练

4090 `assets/reference_top10_20261024`，runner1831644、trainer1831960。
从Reference预训练25k EMA严格加载全部权重，重启optimizer/EMA/局部步数；保持114.30M架构。
Top10 train23485/validation100/test200，原保护划分与映射ID/既有VAE latents复用。
训练/参考库仅Top10 train；1–2个同书家同书体异字参考，无Writer-ID输入、无新Writer分支。
U-Net+风格头全量微调，冻结DINOv2；B256、LR2e-5、warmup200、10k步，每5k保存评测。
保留FM+.05 SupCon与混合困难采样；无通用数据replay。先做0步100图验证作为同口径基线。
活动run `runs/20261024-010707-Unet-flow`；340个源张量严格加载、逐张量相等验证通过。
0步验证100图：SSIM.5586、LPIPS.39050、Ink-IoU.23333、Frag3.0614、SoftInk1.1580。
已完成25步：FM.3373、SupCon4.56（valid256），显存22.47GiB；预计约1.5小时，实测稳态速度后修正。
[实现与后训练配方](tools/reference_style/README.md)。

## 2026-10-23：4090 Reference-conditioned FM 正式启动

用户要求立即开跑。目录`/root/Workspace/xy/DiT/assets/reference_kxl_20261023`，
runner1830611、trainer1830675；活动run `runs/20261023-195603-Unet-flow`。
仅原保护过的楷/行/隶池：100095训练（168条无跨字参考排除）；1044验证（原1046中2条无合格训练参考排除）。
冻结DINOv2-B离线CLS全部缓存完成，768→256→128可训练头、平均跨字参考；生成器不吃Writer ID。
114298852参数，B256，LR6e-5/warmup500，30k余弦调度，每5k保存和固定跨字参考评测。
FM+.05全batch SupCon；50%硬负四元组+50%writer-balanced跨字对。
首次B512前向OOM，失败run保留；B256实际进入训练，step0 SupCon4.432238、valid256、hard170，GPU约22.6GiB。
2026-10-24 00:38已完成30k并保存6个checkpoint。25k候选SSIM.4558/LPIPS.46147/IoU.23664/Frag5.039；
30k为.4551/.46107/.23702/6.665，后段验证平台；尚无独立Writer或reference-swap结论。
按用户要求已转入上述Top10后训练，旧run不续训、不删除。
[实现与启动说明](tools/reference_style/README.md)。48的其他任务未动。

## 2026-10-23：Turbo LoRA从2000完整续训至6000

用户要求继续。trainer927137已实际恢复至2001步，LoRA逐张量与2000检查点完全相等校验通过，
条件投影、两组优化器状态、CPU/CUDA/Python RNG完整加载；冻结书家表与数据SHA核对不变。
配置turbo_extend6000.json：额外4000步，B8/rank16/512不变，每500步保存与同6字8步验证。
学习率从原末端1e-5用100步平滑升到5e-5，之后余弦降回1e-5；这不是继续沿用已结束的原2k调度。
日志train_extend6000.log，原检查点/评测不覆盖；run_status.json链接活动train_status.json，原首轮状态保存在first_run_status.json。
目录仍为48 `/home/ds/Workspace/xy/qwen-lora-turbo-20261023`，预计额外约4.5–5小时，实际看吞吐与评测。

## 2026-10-23：48已切换Turbo LoRA正式训练

用户指定先训Turbo。独立目录`/home/ds/Workspace/xy/qwen-lora-turbo-20261023`，runner612123、trainer614522。
完整校验现有Turbo权重后，已缓存2786张GT后验及1694个字符条件；不复用1张probe缓存。
正式60/2000步实测：B8/不累积、约4.10秒/步、峰值18.58GiB；LoRA16.78M+条件适配4.75M，书家表冻结。
512、rank16、LR1e-4、warmup100，每500步保存与6字原生8步采样验证。当前尚无正式采样质量结论。
非Turbo旧编码573308/573309及旧runner529811已停止；新下载器618420仅下载，不会自动训练或抢GPU。
Turbo与非Turbo目录、缓存、检查点隔离；[配置](tools/qwen_lora/turbo.json)、[实现](tools/qwen_lora/README.md)。

## 2026-10-23：BC FM 114M Brush条件预训练与Top10迁移准备

用户新方案已改为Top10。4090目录`assets/brush114_top10_20261023`，P0 PID1827823已运行，完成约25,600/124,894条固定VAE重建属性测量。
5维：骨架EDT宽P20/P50/P80（按bbox高度归一化）、内部黑度、灰度过渡面积/轮廓长度近似边缘软化。低置信度只屏蔽Brush，不删除FM图像。
114M主干增加5→128→256 Brush MLP及Middle/Decoder独立FiLM，训练10%null、噪声0.075std。Writer在预训练完全绕过。
串行`tools.brush_fm.run`：P0宽度可靠性通过后预检B512→短训5k（60k LR调度）→固定噪声控制测量与图像留档，停在review gate。
通过统计和人工内容审查后才允许resume至60k→Top10全解冻30k，不盲跑。独立书家表冻结，微调属性必须来自同书家同书体的其他训练字符。
首轮沿用约10万楷/行/隶通用缓存+Top10训练，显式75/25采样；非20万/40万。进一步精确去重、严格字符留出，所有标准化与bank只取最终训练子集。
13项轻量检查通过，补充Brush架构的严格Writer转移测试通过；当前尚无Brush生成结果。[实施说明](tools/brush_fm/README.md)。

## 2026-10-23：48上Qwen条件LoRA已部署，下载与编码串行依赖已启动

目录`/home/ds/Workspace/xy/qwen-lora-20261023`，bootstrap PID529811。
正式底模为非Turbo Qwen-Image-2.1，33.13GB独立下载并逐组件SHA256校验；不拿Turbo缓存替代。
VAE完成即编码GT后验mean/std，文本编码器完成即编码字符条件，缓存全部完成后自动接正式2000步训练。
train2730/validation18/test38，颜真卿/智永/赵佶楷书；冻结独立对比学习书家表3×32，训练4-token投影、书体token和rank16 LoRA。
512分辨率、B8、不累积、均衡抽书家、LR1e-4、warmup100、每500步保存及6字验证。
Turbo两步真实反向预检通过（PID577557，已结束）：LoRA16.78M参数、条件适配4.75M；
梯度非零、LoRA权重确实改变、书家表不变，B8峰值18.56GiB。不是8步采样质量或收敛证据。
当前正式优化尚未开始；真实阶段查bootstrap_status.json、cache_*_status.json、train_status.json，不把runner PID当成训练步数。
[实现与限制](tools/qwen_lora/README.md)。

## 2026-10-23 13:40：停止部分解冻续训，启动全解冻P1

部分解冻续训进程1825880已SIGTERM停止，末次确认18,450步，保存历史检查点不删除。
从原通用预训练50k EMA重新初始化，不继承部分解冻微调权重/优化器；直接联合微调10k，保存/评估5k、10k。
`top10_full_unfreeze_config.json`：adapter_only_steps=0，主干patterns=[*]，分支LR1e-4，主干2e-5，STD/Script组1e-5，warmup200。
仅独立对比学习书家表及不使用的共享writer_proj冻结，其余Encoder/STD/Script/时间条件/Middle/Decoder/Output全部可训练。
保留B512、10%通用回放、同验证100字/噪声/VAE/采样；输出top10_full_unfreeze_runs，日志top10_full_unfreeze.log，状态full_unfreeze_status.json。
元设备检查确认三组LR、参数无重复且完整覆盖可训练参数；不同时启动其他建议实验。当前数据是约10万楷行隶，不是附件泛称的40万。

## 2026-10-23 13:00：停止114M，延长48M Top10微调至50k

用户要求切换。114M runner1821400及训练1825018均SIGTERM退出，日志最后34,950步，保留全部已落盘检查点；不会自动接其微调。
48M从Top10微调10k检查点完整续到微调50k（额外40k），新配置`assets/generic48_kxl_top10_20261022/top10_extend_50k_config.json`。
新日志top10_extend_50k.log，状态extension_status.json，新run输出top10_extended_runs；原10k结果不覆盖。
模型/优化器/EMA/RNG恢复，保持联合微调选层、冻结书家表与10%回放；每5k保存评估。
LR调度由10k延至50k，步数仍从10k开始，分支LR约4.6e-5、主干组约9.1e-6；不是原最低LR持续训练。

## 2026-10-23：114M同口径两阶段排在48M之后

新增持久队列 `tools.generic_pretrain.queue_wide`，目录 `assets/generic114_kxl_top10_20261023`。
等待48M预训练50k与Top10微调10k全部完成，再执行114M显存短检→从零预训练50k→Top10微调10k。
结构channels192/384/576、每尺度4块，其余沿用48M最新两阶段配置：B512、同LR/调度、同表/数据/STD/VAE，每5k保存评估。
缓存直接复用，不重新筛选/编码；不加载48M权重到114M。GPU容量预检延后，不打断当前48M；OOM明确失败，不静默改batch。
当前114M只排队，不能宣称已训练或确认B512容量。[入口说明](tools/generic_pretrain/README.md)。

## 2026-10-22：48M楷行隶预训练→Top10微调串行已启动

用户最新预算修订：通用预训练50k，后接Top10微调10k，两阶段均每5k保存/评估。
活动trainer保留运行至其已读入的30k停止点，然后由接管runner从30k完整状态续到50k，LR原本就是50k调度、不改变。
实际生效后续配置 `pretrain_50k_config.json` / `top10_50k_config.json`；原始30k启动清单保留，不能误当最新预算。
微调起点为50k EMA而非30k。下方原30k预算及runner PID为接管前历史。

4090持久runner `tools.generic_pretrain.full`，目录 `assets/generic48_kxl_top10_20261022`。
101,678候选（楷36,591/行35,622/隶29,465），仅必要可读/STD覆盖及留出防泄漏，不额外审美筛选。
最终train100,263/validation1,046，排除369张留出近重复；STD覆盖已通过。
runner PID1819230，当前GPU编码124,894行（含Top10训练/验证缓存），约166图/s；FM尚未开始，真实阶段见远端status.json。
48M channels96/192/256、8块、B512 compile、LR7e-5；预训练30k、50k调度，自动接Top10微调10k。
Top10独立对比表按raw-ID选取10行并冻结，前1k仅Writer-only分支，后9k低LR联合Decoder等，10%通用回放。
新bbox STD、sigma0、增强off；VAE/目标不改；[链路文档](tools/generic_pretrain/README.md)。
配置与清单已拉回[本地](assets/generic48_kxl_top10_20261022/manifest.json)，数据/权重继续保存在4090。


## 2026-10-22：Qwen合成32张小样部署，复用现有环境

按用户要求改为复用48主机 `/home/ds/miniconda3/envs/pytorch/bin/python`，不再创建新环境。
现有torch2.14.0+cu126、transformers5.19.0、accelerate1.15.0可用；diffusers0.40缺接口，已在本任务vendor目录补0.41并通过真实pipeline导入检查，原安装不覆盖。
工作目录 `/home/ds/Workspace/xy/qwen-calli-20261022`；官方Turbo权重约32.45GB，ModelScope下载中，下载前权重SHA与HF指定revision核对一致，完成后全文件校验。
32张固定设计：颜真卿/智永/赵佶楷书，目标泉/梅/岳/霞，2seed；24参考迁移+8STD-only。
最终输入本地 `assets/synthetic/qwen32_20261022_final`，已实际检查参考图并剔除扫描边框；训练集只作异字参考，目标字符排除三书家训练和现有留出字符。
推理runner等待下载就绪后自动生成32张与匿名poster；bootstrap_status.json/inference_status.json记录进度。当前尚无生成结果，不宣称合格。
生成仅为研究候选，不赋予已验证Writer标签，不加入训练；[实现说明](tools/synthetic32/README.md)。

## 2026-10-22：暂停新训练/无模糊评测，完成STD bbox修复

用户改为优先修正STD缩放。固定σ0.65的新训练 `extreme48m_top20_x1_no_std_aug/20261022-205633-Unet-flow` 已停止（最后确认300步，未到首个5k保存点）。
旧几何sigma0推理runner及GPU评测已停止，未将其结果作为修复后结论。
根因：旧renderer固定em居中、未做墨迹bbox缩放。已修复为真实墨迹bbox→等比最长边填满256/短边居中→BOX64，无额外边距。
4090新版本 `assets/data_cohorts/top20_20261021_v1/std_bbox_v1/{display256,condition64}` 全部8945字已生成并通过跨度256/64全量检查。
保持旧index字体/face选择不变；旧缓存、历史权重配置不覆盖。旧最长边覆盖min/median/max=35.9%/70.3%/81.25%。
3项轻量几何回归通过；[对照图](assets/audits/std_bbox_20261022/preview.png)、[修复说明](tools/std_bbox/README.md)。
评测支持--std-root显式选择新64缓存与--std-sigma选择模糊；当前未自动恢复训练或启动新bbox推理。

## 2026-10-22：30k完成，seen/test补评4/4完成

X1于20:37:41正常结束，6个5k间隔权重保存完成，raw+EMA书家表不变核验通过；X2未启动。
25k/30k各Top20 train_half200张、test400张补评完成，同EMA/CFG1/Heun50/固定逐图噪声，与旧114M同集合比较。
30k新/旧：seen SSIM0.7753/0.8302、LPIPS0.16194/0.12474、IoU0.70894/0.77715；
test SSIM0.5646/0.5703、LPIPS0.35732/0.35396、IoU0.29258/0.29073。
SSIM seen-test gap新0.2107、旧0.25985（缩小18.9%），LPIPS gap0.19538/0.22922（缩小14.8%）。
结论：拟合差距缩小主要因seen拟合变弱，留出SSIM/LPIPS仍稍差，不能宣布泛化改善或过拟合被解决。
新模型25k→30k seen改善幅度大于test，gap仍继续扩大；当前没有证据单凭小模型就消除后期泛化平台。
产物 `assets/audits/extreme48m_seen_20261022/{comparison.csv,gaps.csv,status.json}`；阶段报告已补记。
下方排队/在训状态为历史记录。

## 2026-10-22 20:35：补测48.33M seen/test拟合差距

用户要求评估过拟合是否缓解。持久runner PID1815685，等待X1正常完成30k及末次验证后再用GPU。
队列：25k/30k各测Top20 train_half200张与test400张，共4项，EMA/CFG1/Heun50，逐图固定噪声。
沿用冻结训练探针，与旧114M@相同步数的完整评测对比，不混用validation/test。
产物 `assets/audits/extreme48m_seen_20261022`：status.json、逐项日志/PNG/身份、comparison.csv和gaps.csv。
拟合差距缩小需结合留出质量判断，不能把训练拟合更差直接称作泛化改善。

## 2026-10-22 17:22：正式恢复原定B512，X1已启动

编译路径B512容量预检通过：13.61GiB allocated、14.94GiB reserved、短测2.69step/s。
此前eager B512的OOM不能代表compile主路径容量；B256短跑已停止，保留其日志，不混入正式结果。
正式运行 `assets/results/extreme48m_top20_x1/20261022-172221-Unet-flow`，runner PID1812862、训练PID1814666。
记录 `assets/results/extreme48m_top20_x1_launch_20261022_compiled`；实际B512/LR7e-5/warmup1500/50k调度/停止30k。
恢复原定15.36M样本曝光；X2未排队。独立表冻结与EMA574/574再次确认。
17:24实测已到100步，Diff2.0921，编译后首个完整区间2.67step/s，GPU100%，训练日志显存14.67GiB。
初估剩余约3.2–3.5小时（含评测，不作为长期稳定吞吐承诺）。
本条替代下方B256当前状态；首次编译短跑保留为infra试运行，不参与模型比较。

## 2026-10-22 17:16：按用户指令启动48.326M X1

4090持久runner PID1810203、训练PID1810350；当前首次compile，尚无稳定训练吞吐。
运行 `assets/results/extreme48m_top20_x1/20261022-171554-Unet-flow`。
启动记录/日志 `assets/results/extreme48m_top20_x1_launch_20261022_b256/{status.json,train_config.json,capacity.log,train.log}`。
B512容量预检OOM，保留失败目录 `extreme48m_top20_x1_launch_20261022`；改B256预检通过：17.90GiB allocated、18.15GiB reserved。
短测eager3.05step/s只作容量证据，不能当正式compile长训吞吐。
LR7e-5、warmup1500、50k调度、停止30k；每步曝光256而非原512，不宣称等步数即等样本量。
X1不加DropPath/分组WD，WD0.02；X2尚未启动或串行排队。
日志确认训练缓存30386/30386覆盖，独立表冻结与ID映射通过，SHA49128f50d719127dcfadd087a3c270011fececf65925a3ae89061be79bb4860a，EMA配对574/574。
本条取代下方“未启动新生成器”的准备阶段状态。

## 2026-10-22：独立书家表完成；48.33M下一代实现就绪

此前“书家表在FM中可训练”是实现错误，不符合独立对比学习、主训练冻结的约定。
现已独立训练Top20的全部20×32行：train30386、validation200，test400未参与；5000步完成，选中2250步，验证Top-1 52%。
正式原表与审计：`assets/data_cohorts/top20_20261021_v1/writer_tables_supcon_v1`，不采用旧生成器坐标对齐版本。
主训练修复冻结、resume后换表、清除旧Adam槽位及EMA精确拷贝；历史checkpoint保留原可训练身份。
下一代96/192/256×8-block：实际48,325,988参数、50 ResBlock、5 Attention；X1/X2配置与说明见
[extreme48m](src/unet/configs/extreme48m/README.md)。X2为DropPath0.08+分组WD0.05，X1不加此正则包。
两组B512/LR7e-5/warmup1500/50k调度，先停30k；STD形变保持off，不额外引入附件strong造成混杂。
6项轻量测试通过。B512显存、吞吐及收敛尚未实测；未启动新生成器，也未启动旧30k→33k换表续训。
下方“等待是否仅训练表”等记录已被本条修正替代。

## 2026-10-22 16:07：完整评测结束，报告已落盘

20个checkpoint、112/112项test/half-seen评测完成，训练保持停止。
报告：[四个集合的拟合/泛化对照](ANALYSIS/cohort_generalization_20261022/README.md)，含四张完整曲线和数据链接。
主要结论：拟合/泛化差距大；Top5后期LPIPS回退，Top10到30k仍改善，Top20后段平台。
在同一Top10 test上M20与M10基本持平，不比较不同难度集合的总均值来排名。
用户要求尝试Top20书家表训练；已核对30k checkpoint：writer参数在AdamW中更新30000步，动量非零，相对初始表L2变化2.18253。
书家表原本可训练，“解冻”是no-op。未启动无差异续训，等待确认是否改为冻结主干仅训练表的不同消融。

## 2026-10-22 15:36 CST：停止Top2，GPU转全部checkpoint完整评测

按用户要求已停止训练runner/Top2及旧10组测试runner，保留所有日志、已完成测试和权重。
Top2停止在约13k，但仅5k/10k已完整保存；不宣称存在13k权重。
Top20/10/5各5k/10k/15k/20k/25k/30k，共18个；加Top2两个，共20个checkpoint。
每个权重测所有支持档位的test与train_half（half-seen），合计112项。
新队列 `python -m tools.cohort_eval.batch`，输出 `assets/audits/cohort_all_ckpt_20261022`，
runner日志同路径加`.runner.log`；计划/进度/结果分别为plan.json、status.json、results.csv/json。
15:42实测已完成20/112项，最新可用权重test/train_half均完成，正在Top20@25k。
首轮最大档位SSIM（train_half/test）：Top20@30k 0.8302/0.5703、Top10@30k 0.8509/0.6022、
Top5@30k 0.8461/0.6009、Top2@10k 0.7825/0.6227。不同测试列不可据此跨模型排名。
显示较大的训练/测试差距，但探针内容难度未逐字匹配，暂不全部归因于过拟合。
拟合差距表fit_gaps.csv；按逐图结果汇总嵌套子集，并校验身份后复用旧结果，避免重复GPU生成。
batch32、VAE8、EMA/CFG1/Heun50；旧训练不再自动续训。下方运行记录均为历史快照。

## 2026-10-22：独立checkpoint/集合评测与训练拟合探针

新增工具 `python -m tools.cohort_eval.run --checkpoint ... --collection topN/test|train_equal|train_half --out ...`，
支持任意CSV，EMA/raw、步数和CFG显式可配。严格核对checkpoint的条件映射及训练成员身份。
训练探针固定在 `assets/data_cohorts/top20_20261021_v1/train_probes_v1`，等量/减半数量：
Top20=400/200、Top10=200/100、Top5=100/50、Top2=40/20。
半量嵌套于等量，跨层级嵌套，逐图固定种子；3项轻量测试通过。
仅构建探针和工具，不将额外20组train评测插入正在运行的10组test队列。
用法与结果解释见 [独立评测工具](tools/cohort_eval/README.md)。

## 2026-10-22 15:23 CST：前三组30k完成，Top2在训；启动10组测试矩阵

Top20完成02:54:56，Top10完成08:06:32，Top5完成13:16:23；均已保存30k并正常结束。
Top2于13:16:25启动，15:23到12,150步；独立测试并行前稳定1.63step/s，预计晚间完成。
正式测试队列 `assets/audits/cohort_matrix_30k_20261022_v2`：Top20测20/10/5/2、
Top10测10/5/2、Top5测5/2、Top2测2，共10组。统一30k EMA/CFG1/Heun50/逐图固定噪声。
前三模型立即测试，Top2测试等待其30k训练任务正常结束；不混入中途权重。
结果按完成顺序写入 `results.csv` / `results.json`，实时状态在 `status.json`。
首次独立测试因远端评测API不接受diffusion_override而失败，尚未产生结果；
已在独立脚本改用公共接口，失败目录保留为`cohort_matrix_30k_20261022`，不与v2合并。
训练未重启、未改动。下方为历史启动记录。

## 2026-10-21 21:41 CST：用户修订，四组先各30k（替代70k队列）

旧70k队列已停止，Top20约600步warmup，无checkpoint；旧日志保留，未清除数据或产物。
新队列：`assets/data_cohorts/top20_20261021_v1/launch_30k_lr7e5`。
Top20 → Top10 → Top5 → Top2；每组从零训练，**stop_at_step=30000，max_steps=lr_decay_steps=50000，lr=7e-5**。
其他配置和固定数据不变；每5k保存及验证。先30k切换后项，不自动继续至50k。
输出：`assets/results/clean_cohorts_20261021/launch_30k_lr7e5/topN_u114/时间戳-Unet-flow`。
初始ETA约每组5.3–5.7小时，总21–23小时；以新启动后实测速率为准。
下方70k状态仅作被替代的历史记录。

## 2026-10-21 21:32 CST：清洗后书家梯度串行（覆盖下方旧实时状态）

4090 已启动独立持久队列，顺序 **Top20 → Top10 → Top5 → Top2**。
21:35:48 实测：Top20 已到150步，Diff1.8185；编译后连续两段1.63step/s，显存19.10GiB，GPU利用率100%。runner PID `1793996`，训练 PID `1793997`。
每组从零训练70k，B512，普通114M共享条件U-Net；不启用REPA/Critic、硬分离或DiC。
数据固定为 `assets/data_cohorts/top20_20261021_v1/datasets_final`，不采用细线边框额外清洗版。
队列、逐组配置与输入哈希：远端 `assets/data_cohorts/top20_20261021_v1/launch_70k/queue.json`；
实时状态 `status.json`，日志 `top20.log` / `top10.log` / `top5.log` / `top2.log`。
输出根：`assets/results/clean_cohorts_20261021/topN_u114_70k/时间戳-Unet-flow`。
后项仅在前项70k保存、评测完成且正常退出后启动；任何失败即停，不跳过。
估算每组12–13h、全队列49–52h（包含评测/存盘估算，非实测整轮）；预计10月23日深夜至24日凌晨结束。
详细数据数量与协议见 [训练说明](tools/data_cohorts/README.md)。下方2026-10-18状态不是当前实时数据。

运行状态复核：2026-10-18 19:09 CST（Asia/Shanghai）；三臂历史结果沿用15:25登记，CFG审计见下。
全部实验、详细设定、路径、历史对比与图表统一在 [exp.md](exp.md)。
更早前原始状态请查阅 [历史state归档](docs/experiments/ledger/state_history_20261018.md)。

**19:03只读审计补充**：Writer CFG原始文件与用户给出的DiC数值相符；但采样脚本Base/Gate走raw、DiC专用分支走EMA，协议却均标ema=true，条件哈希为空。旧/新CFG1.0基线差异尚未消除，不把这些跨模型行当作统一EMA横评。详见[实验总账§9](exp.md#9-writer-cfg审计2026-10-18-1903只读)及[原始数值快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。本轮未启停训练、未重新采样、未排Style Loss。

## 1. 当前运行任务表

| 机器 | 当前执行 | 当前训练日志 | 最新评测指标 | 队列后项 |
| --- | --- | --- | --- | --- |
| 48（4090，48GB） | **U100-Deep-Sym4 从头预训练**（纯U100实现，关gate，大幅对称加深至 114M 参数，预算 60M 样本曝光） | **已按指令平稳停机封存**（完成至 85,300 步，落盘 0085000.pt），GPU 显存 0 MiB 完全释放；**已完成深度磁盘瘦身与压缩**，工作空间由 150G 压缩至 65G（瘦身 56.7%），根分区可用空间由 172G 扩展至 256G（净释放 84GB） | **确认全面收敛，刷新全线 SOTA**：85k 节点 SSIM Mean **0.6129** (75k 峰值 **0.6132**) / LPIPS **0.3207** / Ink-IoU **0.3381** / Frag **2.14** / 零空白！连续 1.5 万步指标高度横盘，训练损失 0.0592 | 核心成果（0075000.pt、0085000.pt、Base 55k、DiC 三臂等关键权重与评测数据）全量无损归档 |
| 4090（24GB） | **U100-DiC-768**，75.83M CNN U-Net | **79,250 / 100,000**，约 1.71 step/s，显存 22.24 GiB，Diff 0.0742；PID 1738543 稳健存活 | 50k 历史峰值（SSIM 0.6047 / IoU 0.3234 / Frag 2.301 / Hit 64.6%） | 冲刺 100k 最终封版 |

凌晨历史配置/日志：[03:47快照](docs/experiments/ledger/live_snapshot_20261018.json)；晚间CFG及进程核验：[19:03–19:09快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。两份快照不能互充实时状态。
本地只拉取文档和关键小图像，不滥停训练，非指示不修改任何训练代码。

## 2. 48卡：CalliFlow-v2 从头预训练（Deep DiC + 双速度场显式建模）

| 核心项 | 详细设置与实时状态 |
| --- | --- |
| **架构代际** | **CalliFlow-v2**（下一代 Hourglass CNN U-Net，参数量 63.23M） |
| **核心创新** | **双速度场显式分解**：$v_\theta = v_\text{base}(z_t, t, g, s) + \Delta v_\text{writer}(z_t, t, g, s, w)$ |
| **Base Field** | 负责汉字内容、基本笔画连通与结构合理性，输入严格物理屏蔽 Writer ID（纯学 $\mathbb{E}[v^*\|z_t,t,g,s]$） |
| **Writer Field** | 负责书家结体变化、笔势转折与墨迹特征，多尺度提取（$8^2 \to 16^2 \to 32^2$），DiC通道条件门控调制，零初始化头 |
| **推理加速** | **单次前向极速 CFG**：$v_\text{CFG} = v_\text{base} + \alpha \Delta v_\text{writer}$，彻底免除双倍网络前向调用 |
| **损失函数** | **双 FM 闭式监督**：$\mathcal{L} = \|(v_\text{base} + \Delta v_\text{writer}) - v^*\|^2 + 0.5 \|v_\text{base} - v^*\|^2$，无需外部风格教师或配对真迹 |
| **骨干规格** | Channels `[128, 256, 384]`，Encoder 4 ResBlocks/尺度，Decoder 4 ResBlocks/尺度，Middle 3 ResBlocks + SDPA，GELU 激活 |
| **STD 先验** | $32^2, 16^2, 8^2$ 多尺度空间注入，mild 增强为主 |
| **训练动力学** | Global Batch 512，LR 6e-5（Cosine 衰减至 0.1），Warmup 1500 步，EMA 0.999，Channels-last，Torch Compile |
| **执行监控** | PID 714015，吞吐 **2.83 step/s（1,449 samples/s）**，显存稳健驻留 **16.42 GiB** / 48 GiB，日志位于 `logs/train_calliflow_v2_120k.log` |
| **阶段性观测点** | 25k（12.8M 曝光：内容收敛与残差非零）、75k（38.4M 曝光：等曝光量对标 DiC-768）、120k（61.44M 曝光：最终封版） |

## 3. 4090卡：U100-DiC-768（从头训练，全功能 DiC 架构 U-Net）

| 核心项 | 登记 |
| --- | --- |
| 测试的是什么 | 全功能 DiC 架构组合：全开 `residual_cond_gate` + `stage_specific_cond`（adapter 模式）+ `act_fn='gelu'`。 |
| 规模与吞吐 | Batch=768，75,831,524 参数，步速 ~1.71 step/s（1,313 samples/s），显存占用 22.24 GiB / 24.56 GiB。 |
| 50k历史记录 | B768×50k=38.4M曝光，Base B512×55k=28.16M；SSIM0.6047/IoU0.3234/Frag2.301不构成等曝光单变量收益。Frag不是越低越好。 |
| 当前进程 | 19:09核验PID1738543存活，日志76.6k，Diff0.0767，目标100k。 |

## 4. 已落盘核心资产

- **4090从头训练50k产物**：`4090:/root/Workspace/xy/DiT/assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow/checkpoints/0050000.pt`（及 5k~50k 完整权重队列）。
- **4090 50k评测海报与指标**：`assets/results/eval_4090_b768_50k/posters/eval200fix_poster.png`，`eval_stdskel_summary.csv`。
- **48卡三臂完赛 Checkpoints 与评测**：`assets/results/unet_dic/` 下三臂 55k~75k 完整 checkpoints 与 summary CSV。
