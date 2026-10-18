# 实验总账

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

## 1. 当前执行：仅两条架构线

### 线 1：Node 48 —— DiC 三臂对照串行队列（50k Checkpoint 热启至 75k 步，已全量完赛）

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
