# 2026年9—11月书法生成全周期报告：仓库外图像与数据清单

**审计依据：** KiritoFD/DiT，main 分支截至 2026-11-02 的完整 Git tree（提交 155081ea16ff）。本清单只把仓库**确实没有的独立原图/必要配对数据**列为待取项；“有文字报告但没有真实PNG”与“已经有原始CSV、只需画图”严格区分。此处的“缺失”仅指当前 Git 跟踪文件缺失，不意味着服务器上也不存在。

[返回全周期报告](./2026-09_to_11_full_cycle_review.md)

## A. 需要从服务器或历史归档回收的原始素材

| 优先级 | 图号 | 当前仓库缺失的精确内容 | 现有材料（不要重复索取） | 建议归档位置 |
|---|---|---|---|---|
| **高** | **FIG04：v21/v23 SkelNet** | **v23 SplitLN 对应的独立生成海报**，及同一字符的 STD骨架、SkelNet预测骨架、GT骨架、最终生成图，各6–10例；需附 step/Writer/Script/idx。 | 已有 v21 的 seen/strict_poster.png 与 seen/strict_struct.png（assets/results/v21_skelnet_200k/posters/）；也有历史跨版本总海报。缺的是专门可对齐的 v23 输出与中间骨架。 | docs/reports/assets/sep_skelnet/ |
| **高** | **FIG05：两阶段为何失败** | 同一批 img_id 下的**GT骨架、纯预测骨架、pred+std混合、纯STD**四种条件输入，以及同一 Render checkpoint 的四套实际输出PNG；每组必须记录 eval_blend_alpha（尤其0.5和1.0）及条件来源。 | 文字审计与指标已在 docs/experiments/2026-09-30-stage1-skel.md、2026-10-03-v46-singlestage-experiment-summary.md；历史外部目录曾记录 exp-std/runs_5050/posters/eval200_poster.png 和 exp/purestd_posters/eval200fix_poster.png，但**这两份路径下的原始PNG不在当前Git树**。 | docs/reports/assets/two_stage/ |
| **高** | **FIG13：K/GAR Decoder公平读图** | ~~对同一批**冻结K最终ODE latent**，分别以原始Decoder、D1@2500 Decoder、GAR Decoder解码的逐字PNG及GT~~ **2026-11-02 已补齐** ✅ | 见下行「已回收」 | `assets/d_20261101/reports_assets/gar_crossdecode/` |
| **中** | **FIG03：9月中旬风格/少样本** | v14 87联合类别与 v15 少样本实验的**GT配对、书家/书体/字符索引和同字结果**；至少3个书家、每家3–5字，包含Seen/Strict身份。不是要求重新训练。 | 仓库已有 v15a/b/c 的部分 seen/strict g*.png（docs/1031/eval_imgs/assets/ink_eval/）和大里程碑海报；**v14/v15/GT的共同样本manifest和可直接使用的同字合成图**未见。 | docs/reports/assets/style87_fewshot/ |
| **中** | **FIG08：DiT与U-Net同图读图** | 72M U100、114M Deep-Sym4，以及所选DiT对照在**相同样本、GT、seed和采样条件**下的最终输出PNG（建议20例，覆盖细笔与复杂偏旁）；并补每个run的准确GPU墙钟/FLOPs记录，以便不把step节省误报为计算节省。 | 两条U-Net的训练曲线数字和部分poster已在10月18/19日报告与 docs/experiments/ledger/unet_35000_poster.png 中；仅需**等面板同条件原图**。 | docs/reports/assets/unet_dit_matched/ |

### A1. 已回收：FIG13 同 latent 交叉解码（2026-11-02）

**位置：** `assets/d_20261101/reports_assets/gar_crossdecode/`（约 20MB，900+ PNG）
**内容：** 4 臂 × 100 字（K / K+D1dec / D1b_2500 / GAR@672）的 `g*.png` 与 `gt*.png`，
外加 `manifest.csv`（400 行，含 checkpoint sha256、seed、sampler、CFG、GT 与输出哈希）与 README。
**可比性：** 已逐字段核验四臂共享生成器 `60a21973…`、seed 0、heun/50/CFG1.0、同 VAE 与同数据集，
100 张 GT 合并哈希四者相同；作图脚本强制检查，不通过拒绝出图。
**图：** `assets/metrics_index/fig13_crossdecode.png`（左）、`fig_decoder_fig13.png`（右）。

**回收时发现的评测事故（重要）：** `D1b_2500` 的 `decoder_override` 为 `null`，
其 200 张生成图与 K **逐字节完全相同**，指标也与 K 完全相同 ——
即 D1b 评测从未换装它训好的 decoder，是**空操作**。
此前报告/文档把 D1b 当「decoder 起作用的干净对照」不成立；
真正换装的是 `K_plus_D1dec`（指向 `runs/D1/checkpoints/0002500.pt`，SSIM 0.598173）。
**该类实验今后必须核对 `eval.json` 的 `decoder_override` 并抽查输出 PNG 是否与基线相同**，
只看指标表格发现不了此故障。详见 `assets/d_20261101/reports_assets/gar_crossdecode/README.md`。

## B. 已经在本次对话提供，但尚未作为Git文件保存

| 对应图号 | Git当前缺失项 | 当前状态 | 还需要什么 |
|---|---|---|---|
| **FIG15a/b** | **21字 × 多历史模型的完整原始海报PNG**（及便于展示的上下半部裁切） | 用户已在对话提供；上一版PDF已内嵌。**不必再找或重新生成这张图**；目前Git树没有这个完整21字拼图，仓库Markdown版保留空位。 | 把现有PNG复制进 docs/reports/assets/21_chars/；更重要的是补一份21行元数据CSV，以免海报只有肉眼证据。 |

**21字CSV的最小字段：** row_idx、img_id、char/unicode、writer_id、script_id、GT路径及hash、原始评测面板、每列model_id/ckpt_sha、seed、CFG、solver/steps、STD渲染版本。至少保证行与GT的1:1对应；没有的字段留NA而不是猜。

## C. 当前Git里没有的“同口径新测结果”（不是从旧硬盘搬图）

这些属于严谨的最终总结所需的补充评测，不应被写成已完成结果。

| 优先级 | 图号/用途 | 所需新数据 | 原因 |
|---|---|---|---|
| **高** | 结题核心表／FIG14补充 | S0@50k、S4@50k、K、K+D1dec 的**同面板、同seed** Writer-swap：逐字符/书家方向、幅度、OCR、E-G，附 bootstrap CI。 | 仓库已有各自部分 Writer 指标，但不是所有模型都完成严格相同的交换条件；当前不能将不同 T3 配对数或不同面板拼成统一风格冠军。 |
| **高** | FIG06／历史强模型对比 | **187共同样本**上，K、S4、S0与v68/v80的评测清单：同 img_id、GT hash、采样协议和seed；需要实际生成PNG及逐字 SSIM、LPIPS(Alex/VGG标明)、Ink-IoU、OCR、E-G。 | 仓库有部分历史187图和不同原生指标，但没有能支持全部模型严格并列的新统一原始产物。不是要求重新训练模型，只需有权重时统一推理。 |
| **高** | 严格开集结论 | 按**全部阶段训练字/字符联合集合**构造的 seen组合 / unseen Unicode / source-holdout manifest（至少 img_id、writer、script、char、source_id、所有训练阶段seen标记）。 | 旧文档对val100是seen还是unseen有冲突；缺统一名单前不能称“严格未见字泛化已验证”。 |

## D. 仓库已经有的素材：**不需要你再取**

- **FIG01、FIG07、FIG12、FIG16：** 全部有可直接绘制的文字/结构化源数据；分别见 All.md、assets/ledger/points_all.csv、assets/metrics_index/lineage_summary.csv、docs/d_results.md；只需CPU作图/文案排版。
- **FIG02和FIG06：** docs/04_experiments/imgs/historical_strict_common_poster.png、milestone_eval200_evolution_poster.png，assets/milestones_posters/01_v10b_390k.png、05_v66_150k.png、06_v68_200k.png，均已在仓库。
- **FIG09：** docs/infra-1015-measurements.md 和 ANALYSIS/style_repr_injection_matrix_20261026/measurements/ 含4090/48GB计时及预算JSON，足够先绘出效率图。
- **FIG10、FIG11：** docs/1028/posters/S_comparison.png、J_comparison.png，以及 docs/1028/raw/S*/J*、ANALYSIS/.../full_eval_20261028/ 结构化数据均在Git内。
- **FIG13已有部分：** assets/d_20261101/review/ 含K+D1dec best/worst 20图；assets/metrics_index/fig_decoder_sweep.png 有Decoder步数曲线，无须再次索取。
  **2026-11-02 更新：FIG13 右侧「ΔSSIM/ΔLPIPS/容差骨架F1 曲线」已补齐并升级为六轴图**
  `assets/metrics_index/fig_decoder_fig13.png`（生成脚本 `plot_fig13.py`，纯本地 CSV 作图，
  无新推理/无跨机传输），其中还额外画出 E-G 身份轴——它是**非单调**的（1000 步见顶后转负），
  此前 `fig_decoder_sweep.png` 缺这一轴，容易让人误读成「越久越好」。FIG13 仍缺的只有
  左侧「K+GAR 同 latent 独立解码 PNG 组」。
- **FIG14：** assets/metrics_index/fig_pooled_eg_ocr_labelled.png 已存在，可直接嵌入本报告的GitHub Markdown版。用户提供的134面板图也已经纳入PDF版。

## E. 建议的最小提交格式

按优先级先补 **FIG04 + FIG05 + FIG13** 三组真实图；FIG15原图已在本次对话提供，只差归档。每组遵循：

- images/: 真正从模型运行目录导出的原始 PNG（不作AI仿造）。
- manifest.csv: img_id、char、writer、script、split、GT hash、model checkpoint、seed、solver、CFG，必要时加 latent hash。
- README.md: 原始运行目录、生成脚本命令、权重文件及采样版本；缺失字段显式写NA。
- 保留旧版报告里的图位，直到原始图片与身份对应性核对完毕。不要把“只有PNG”视为可直接计算同协议统计指标。

**结论：当前不需要重新下载全仓库数据，更不需要补14张外部原图。** 缺口主要是9月SkelNet/两阶段的**配对中间结果**、11月GAR的**同latent实际生成图**和补齐真正同口径评价所需的**逐字manifest**。其余图可以从已经入库的数据直接制作。
