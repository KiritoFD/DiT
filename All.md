# All — 书法生成全部实验索引与研究结论

> **定位**：长期实验账本和“为什么做、看到什么、能推断什么”的入口；不复制已有大量表格。代表性的可展示结果进 [Milestones.md](Milestones.md)；当前实验及未决问题进 [Current.md](Current.md)。新增 run **必须在这里建索引**，然后链接原始配置/日志/逐图表。数据和结论之间需要显式保留协议差异。

## 1. 已有结构化真相源（先复用）

| 数据/文档 | 范围 | 使用原则 |
|---|---|---|
| [exp.md](exp.md) | 训练/测试详细总账，多阶段实验 | 每条 run 的起点、参数、历史版本 |
| [docs/04_experiments/00_exhaustive_experiment_chronicle.md](docs/04_experiments/00_exhaustive_experiment_chronicle.md) | 13 纪元长篇编年史 | **历史叙事**；与后来的勘误对读 |
| [assets/ledger/points_all.csv](assets/ledger/points_all.csv) / [runs_all.csv](assets/ledger/runs_all.csv) | 2026-10-09 快照，616 评测点/36 runs | 有效的旧实验局部账本，**不是 10-31 全库** |
| [tools/all_experiments_eval.json](tools/all_experiments_eval.json) / [assets/all_runs_deep_parsed.json](assets/all_runs_deep_parsed.json) | 更多旧 run 汇总/解析 | 需检查去重、错误协议和终点选择 |
| [docs/106moyi](docs/106moyi/README.md) / [docs/107](docs/107/README.md) | MoYi 基线，数据/指标与证据勘误 | 多协议数字不能直接拼 |
| [docs/108](docs/108/README.md) | 本地/4090 库存、代码漂移、评测协议差异 | 解决旧 checkpoint 来源与相互矛盾的值 |
| [docs/1028](docs/1028/CONCLUSION_AND_NEXT_PHASE_20261028.md) | S/J 对照，多轴风格与图像训练 | 最严格的近代 matched 对照 |
| [docs/1030/eval_style_v2](docs/1030/eval_style_v2/FINAL_REPORT.md) | E-G 真迹可靠性及生成书家指标 | E-G 来自同一特征网络，不能重复算“独立验证” |
| [1030p125.md](1030p125.md) / [1031.md](1031.md) | P75 训练、E-G、OCR/环境 | 以**实际** P0 50k、P1 5k、P2 15k、P3 5k 记；旧 120k/125k 是历史标签 |
| [ckpts.md](ckpts.md) / [Data.md](Data.md) / [Disk.md](Disk.md) | 资产、VAE、数据与归档 | 文档记录非现场文件校验 |

## 2. 全历史谱系（不是线性“每版都进步”）

| 时期/路线 | 代表 run | 尝试变量 | 核心观察 | 判断/证据 |
|---|---|---|---|---|
| 初代 DiT+DDPM | v1–v9 | char-ID、writer-ID、REPA | seen 字有效，OOV 无字表保证；采样慢 | 闭集记忆与结构泛化必须区分（A/B） |
| STD 骨架开集 | v10b | no-char-ID + 骨架 xattn | 开集内容可行，但 long-run seen–strict gap 大 | 保留核心想法，放弃无必要的全层重 xattn（B） |
| Flow+4ch vs 12ch | v11/v12，MoYi 12ch/4ch | 动力学、辅助图像/骨架共同去噪 | 4ch 同预算与 12ch 持平或更好；项目 12ch aux 梯度占比约51% | 12ch **在现有条件/权重配置下**弱，不是理论不可能（A） |
| 数据清洗+条件减法 | v13 | 清洗 50k、FM、轻量 AdaLN | 基础画质/稳定性提升 | 多变量合并变化，不能单独将全部增益归因 FM（B） |
| 书家×书体细分类 | v14 | 87 class 细分 | 稀疏 Writer/Script 槽变难 | 长尾限制（B） |
| SkelNet/形变 | v21/v23 | learnable 结构预形变、Joint→Split Norm | 形变级联失败与归一化振幅冲突 | SplitNorm 诊断有意义，但两阶段整体未晋级（A/B） |
| 字表/预训练表示 | v54/v60/v61/v66/v68 | learned vs random table、路由、C2OT、增强 | 语义表比随机表强，v68 SSIM 最强；闭集代价明显 | 纯表开集受限（A）；C2OT 与 route 的独立增益**未证明** |
| STD-skel 迁回 | v70 | pure STD skeleton + C2OT | 65k 平台约0.5535 | 此一实现负面，不否定 STD 图像条件（A） |
| VAE 与 latent | v71/v72/v74/Calli-VAE | 新 VAE、noaug、posterior mode、频域表征 | v72 收敛快，但 VAE 与 data 混杂；mode 切换负面；早期 v81 长训“越训越淡” | 编解码/latent contract 是一阶变量（A/B） |
| STD-soft+条件注入 | v83/v84 | 移除 char-table，标准字 64×64 → 256 tokens + pooled | 兼顾任意字输入和可控结构；v83 arm4 无增强综合胜出 | 当前开集生成的主基线（A） |
| Writer 风格监督 | v85/v86/v89 | writer CE、pair、critic、表征监督 | 分类器可被模型迎合；v89 训练 CE99.77% vs 留字符30.07%、留来源13.64% | 不能拿训练风格教师输出当独立风格证明（A） |
| 复杂交互 vs 简单基线 | v88、S0–S11 | Writer×Glyph、局部 attention、spatial AdaGN、DynamicConv | S0 内容强，S4 结构/墨迹略好，S5 E-G AUC 高，其他多数不稳；v88 长训较弱 | 注入结构必须与表征形式匹配；复杂度不保证收益（A/B） |
| 解码器训练 | Deep85→J0–J7 | Decoder 可训、像素 LPIPS/Sobel、Writer 条件 | J2/J3 的 SSIM/LPIPS 确实改善；骨架/Writer 没同步；J5–J7 成绩近似 | Decoder 可改善成像，不自动改善字形拓扑或身份（A） |
| 通用预训练→Top10 | P75（原 P125 命名） | P0 其他书家；P1 混合；P2 Top10；P3 E-G 分叉 | 50k P0→5k mix→15k Top10→5k；P3B 像素指标微优；writer 条件曝光偏少 | **通用预训练正迁移与总算力优势仍未证明** |
| 通用模型参考图/LoRA | Qwen-Image、Qwen3-VL refs | 两张同书家异字参考、LoRA、合成数据 | 当前参考图注入多数负面；数据协议不匹配 | 不将“见参考图”当作天然收益（A/B） |
| 独立内容/风格评价 | DINO、E-G、OCR | real-AUC、OOV 字符、OCR 准确率 | 真实 E-G AUC95.98%；低阶特征 AUC49.35%；E-G 字符检索13.14%；OCR 已在 J/S/P 面板中发现错字差异 | 两轴必要，GT sanity/面板配对缺一不可 |

## 3. 哪些改动确有实测价值？

- **标准字形作为连续空间条件**：使模型具备输入新字符的接口；优于“只能查到已训练 ID 的字表”这一**能力约束**，不代表每个 STD 实验的 SSIM 都更高。
- **真实数据清洗与标签/极性修复**：清除黑底、裁剪、字映射混乱等伪信号；这类正确性收益不能包装成架构新颖性。
- **4ch 主生成目标 + 独立几何条件**：MoYi 50k 的 4ch/12ch 指标接近，项目内等权 12ch 梯度竞争实测明显；建议作为经济、稳健的基线，而非绝对理论禁令。
- **已学习的表内容/合适的条件结构**：旧20k同频随机表 ablation 比 learned table 少约0.0106 SSIM；但该数值限于**同一实验协议**。
- **长训练与成熟 U-Net**：S0@85k 的 OCR 比 50k 好（**两个面板须先重算同口径**）；J0/J2 需记完整 Deep85 先验。
- **局部 Writer 条件**：S5@50k E-G AUC87.98% vs S0 82.13%；在 v2 独立评测协议 S5 89.99%，但内容损失与跨书家异质仍存在。
- **Decoder 解冻+弱图像域监督**：J2/J3 相比 J0 在**同起点** SSIM +0.0045–0.0049，LPIPS 改善；OCR/Writer/结构不一定同步。
- **OCR 与 E-G 两轴评价**：有了“字写对没”和“像哪个书家”的不同量表；新 OCR 曾出现字符串排序及 PIL 宽高 bug，必须保存修正后的权威样本键。

## 4. 哪些改动阴性/误导，哪些尚无因果证据？

| 类别 | 具体判断 |
|---|---|
| 已见反例 | 在现有 g 条件下等权 12ch；多余结构损失；1px 线经连续 SD VAE；SkelNet 两阶段级联；v88 Writer×Glyph；单纯 Spatial AdaGN(S1)；DynamicConv 未稳晋级；冻结 Decoder 图像损失(J1≈J0)；多个 style tokens 堆叠；Qwen 2-ref 注入。 |
| 混杂无法归因 | C2OT；Sym3x 增广；所谓“黄金”层路由；v72 中 VAE 与 noaug；历史 SDE +0.02（在不同 latent/ODE 修正后未通用复现）；模型变大是否总会改善未见字。 |
| 需要重新检查 | v10b/v13 标注“错骨架反而证明完美零样本正交”：错位 Bug 只证明流程/条件依赖，需要同协议反事实；v70“无限泛化已成功”与后来的平台失败矛盾；历史报告中“12ch 已彻底证伪”的泛化措辞过强。 |
| 尚未回答 | P0 50k 预训练净正迁移；P75 扩 Top10 至额外25k能否追上 scratch；E-G 监督能否提升独立风格评价；Writer 条件应控制 U-Net 高/低频还是 Decoder；OCR 能否可靠发现局部缺笔。 |

## 5. 一个可检验的统一数学框架

把书法图像写为潜在生成过程：
\[
x = D_\psi(z),\qquad
z \sim p_\theta(z\mid g,c,s,w,r), 
\]
其中 `g` 是标准字形的连续结构，`c` 目标 Unicode，`s` 书体，`w` 书家，`r` 是同条件下自然书写差异。`D_\psi` 为 VAE decoder。

核心不是强迫 `z` 唯一，而是学习条件分布 `p_\theta`：**“字是什么”与“怎么写”可以相关，但不应把 writer 变成唯一的内容来源**。以信息论语言提出可验证目标：
\[
I(X;C\mid G)\approx 0\quad\text{并非要求！}
\]
**注意：上述零互信息不是训练目标。** `g` 在有限字体/异体映射下不一定唯一决定字符，且“同字不同书家”的合法形变会影响局部拓扑。实际约束是 **OCR 正确性不随 writer 改变而系统恶化**，同时 `I(X;W\mid C,S,G)>0` 且这种依赖沿着真实书家差异方向。

把 FM 输出拆为一种**诊断性**分解（并不要求工程中真做两个网络）：
\[
v_\theta(z_t,t;g,s,w)
= v_{\mathrm{base}}(z_t,t;g,s)
+\Delta v_w(z_t,t;g,s,w).
\]
- `v_base`：公共构字、墨迹和几何结构；
- `Δv_w`：条件差异，可能是**结字、笔宽、墨色、收笔与飞白的跨尺度变化**，不必是字形刚体形变；
- `D_\psi`：latent→RGB 的高频笔触映射，影响像素/LPIPS，但不保证 char/Writer 条件忠实。

可检验的三条预测：

1. **主输出目标优于重复监督**：若 g 已提供足够几何，12ch 额外重建目标不会提高可预测的 `I(X;G)`，却可占用优化梯度。通过四通道/12通道等计算与图像梯度冲突对照检验。
2. **条件控制强于无意义自由度**：S5 的 `Δv_w` 应有更大独立 writer 检索 AUC/正确 Writer 比错误 Writer 的 FM loss 差；不能只看 E-G 自身分数。分层 ablating 可以找出贡献尺度。
3. **Decoder 主要改高频的假设**：固定 `z` 交换 `D_\psi`，如果 LPIPS/笔宽有变化而 OCR/Writer-swap 基本不变，支持图像域改善机制；若字符类别变了则否定“只改纹理”。

更严格地分解训练与泛化误差：
\[
\mathcal R_{\rm target}=
\underbrace{\mathcal R_{\rm representation}}_{\text{STD/VAE/条件接口}}
+\underbrace{\mathcal R_{\rm estimation}}_{\text{数据覆盖与模型容量}}
+\underbrace{\mathcal R_{\rm adaptation}}_{\text{writer条件曝光/优化}}
+\underbrace{\mathcal R_{\rm evaluation}}_{\text{OCR、E-G与面板误差}}
\]
这只是**研究预算的启发式分项，不是数学上自动相加成立的恒等式**。真正的训练主目标仍是条件 FM，例如 `L_FM = E‖vθ(z_t,t,cond)−(ε−z_0)‖²`（该仓库使用的时间方向；必须对照实际 sampler 代码）。

**现阶段最有用的理论表述**：我们在优化一组相互冲突但相关的量：字符内容 `\mathcal C`、条件风格 `\mathcal W`、墨迹画质 `\mathcal Q`、跨字泛化 `\mathcal G`；一个 checkpoint 只有在**统一协议下的 Pareto 改善**才可称为全面晋级。SSIM 平台不等于条件模型学好了，E-G 高分也不保证字写对。

## 6. 今后每条实验必须添加的索引（模板）

| id | family/parent | train_steps/Top10曝光 | ckpt身份 | data_hash / STD / VAE | eval_id / seed / N | OCR/EG/SSIM/LPIPS/IoU | verdict | source |
|---|---|---|---|---|---|---|---|---|
| 待填 | — | — | — | — | — | — | pending | — |

不允许只有“best.ckpt”“100k”和一张裁剪过的 poster。最小可追溯身份是 **run_config+optimizer/EMA+checkpoint+data/char/remap+VAE/decoder+eval panel+seed+指标实现版本**。训练结束从 [Current.md](Current.md) 按这一格式迁入本文件，再挑少数晋级节点放到 [Milestones.md](Milestones.md)。

## 7. 建议把全部实验做成机器可读 manifest

无需把过去数百条实验手工抄第二遍。首选将仓库现有 `assets/ledger/runs_all.csv`、`points_all.csv`、`tools/all_experiments_eval.json` 和最近的 `docs/1028/raw`、`docs/1030`、`1030p125.md` 提取到同一 `assets/experiment_registry.jsonl`（一行一身份），再自动生成本文件的“完整 run 索引”表。

生成器**只合并有 provenance 的字段**；冲突结果保留多行并标注协议，不做数值覆盖。现阶段尚未生成 registry——先保留现有结构化来源，避免“汇总”反而制造错误身份。

## 8. 2026-10-31 实测补充（本机执行的数字与路径，与会话同步）

以下为本机（4090/本地 4070）当日实测，供 §2/§3 的判断引用；来源口径见各条。

### 8.1 OCR 新评测修正后全量数字（读光判别器，配对以 eval_stdskel_batch.csv 为权威）

- 修正内容：PNG 字典序配对 bug + PIL (w,h) 反置 bug；此前 0%/7.8% 全部作废。
- 4090 侧（`docs/1031/ocr_panel_scores.csv`，val100 全量）：S0@50k 71.11/80.00、S1 77.78/86.67、S4 77.78/88.89、S6 73.33/81.11、S8 75.56/78.89；P0 67.78/81.11、P1 67.78/82.22、P2 75.56/88.89、P3A 74.44/91.11。
- heldout187（协议 CSV 直配）：S0@85k **81.29**/89.47（seed1 76.02/90.06）。
- 48 侧（`docs/1031/ocr_panel_scores_48.csv`）：S5@85k h187 三 seed 75.44–78.95；S11@50k 78.95/73.33；J5/J6/J7@5k h187 82.56/81.98/81.40、val100 79.07–79.65；机制 decoder_swap 与 original 完全相同（96.15，train panel）——换解码器只动风格不动内容的直接证据。
- 各面板 GT sanity 89.4–100%。

### 8.2 P125 链运行时勘误（影响历史解读）

- `optimizer()` 的 `rate()` 读 `lr_style_representation`/`lr_style_injections`，而 Config 字段为 `lr_style_repr`/`lr_style_inj` ⇒ **P125 全链 style 两组实际 LR 为 legacy 1e-4**（P3A ckpt `initial_lr=1e-4` 实证），spec 写的 1e-5 从未生效。2026-10-31 已修（tools/p125/patch_rate_fields.py）。
- 关联修复：restart scheduler 的 base_lrs 现显式取自本 spec config（resume 会把 param_groups 覆盖成旧衰减 LR）；P0(3 组)→B/C1(4 组) 跨拓扑 resume 用全新 moments；固定 LR 视界校验不再作用于 schedule_restart 的 arm。见 tools/style_matrix/runtime/{train,common}.py。

### 8.3 权重与判别器部署

- S 矩阵：4090 `assets/style_matrix_20261027/matrix64/4090/execution/runs/S*/checkpoints/0050000.pt`；48 同构 S2/S3/S5/S7。
- 85k：4090 `assets/style_focus_85k_20261029/4090/runs/S0/checkpoints/`（50k/55k/85k）；S4@85k 停于 51k（0051024.pt 干净恢复点）；48 S5@85k 同前缀。
- P125：4090 `assets/p125_20261030/execution/runs/{P0,P1,P2,P3A,P3B}/checkpoints/`。
- 读光判别器 `duguang_ft_ext.pt`（77MB，md5 6e12a8e0…）已部署 4090/本地(`assets/p125_20261030/ocr/`)/48(`assets/p125_20261031_ocr/`)；Real-Test Top-1 80%（微调前 0%）。
