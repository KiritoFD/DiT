# U100 实验工具集（tools/u100_pretrain）

围绕"通用预训练 → Top10 后训练"主线的全部实验工具。逐件说明见各脚本 docstring；
设计与结论引用：`ANALYSIS/style_structure_diagnosis_20261025.md`、`ANALYSIS/p1p2_writer_aware_20261025.md`、
`ANALYSIS/u100_generic_pretrain_design_20261024.md`。

## 主线训练

| 脚本 | 用途 |
| --- | --- |
| `prepare.py` | 书家分组数据（Top10=行0-9 预留 + 通用 248 组 + OTHER；两臂 CSV + tables） |
| `run.py` | 通用预训练 → FT 管线（单臂；B256；raster/bank/prepare/benchmark 分阶段断点续跑 + 容量门禁） |
| `extend_ft.py` | FT 余弦延长（完整状态 resume，显式 horizon 变更） |
| `run_p1p2.py` | **当前主线**：P1 writer-aware 混合 10k（70% 通用 null 行 +30% Top10）→ P2 配对专精 20k（PairMixtureSampler 30%）→ 离线评测 |
| `run48.py` / `run_style48.py` | 48M 对照臂 / writer_slot 连续风格臂（诊断用，已按结论关闭） |

## 特征与评测

| 脚本 | 用途 |
| --- | --- |
| `build_rasters.py` | 从 STD store 索引确定性重渲 256px raster 并逐位校验 |
| `precompute_bank.py` / `build_style_bank.py` | strong 形变 bank / DINOv2-B CLS 特征库（管线组号） |
| `merge_top10_bank.py` | Top10 特征库书家序重映射（名字序→raw-ID 序） |
| `eval_offline.py` | 检查点离线三口径评测（eval200fix σ.65 / test200 σ0 / val100 σ0；--raw-ids、--identity-callig、--allow-legacy-ema、--sets） |
| `eval_round1.py` | 笔宽 EDT 脊线 P10/P50/P90+W1 + DINO 探针 Writer Top1/Margin |
| `build_probe.py` | 独立 writer 探针（VAE 解码域训练图 → DINO → logistic，留出验证） |
| `exp_writer_swap.py` | 固定 STD/噪声 × 书家行切换反事实（探针判定 + 散度 + 海报；执行有挂因待修） |

## 风格结构诊断（2026-10-25 轮，全部纯分析）

| 脚本 | 测量 | 结论要点 |
| --- | --- | --- |
| `exp_a_diagnose.py` | 初版跨字差分相关（被修正） | 0.58 系共享项伪影 |
| `exp_a2_variance.py` | 平衡面板方差分解 + 双置换检验 | 主效应 0.5%/交互 5.6%/无低秩结构/相关=基线 |
| `exp_a3_patch.py` | patch 级谱 + B0/B1/B2 留字预测 | patch 交互无结构；ridge 增量零 |
| `exp_a4_pixel.py` | 像素验证面板 | 交互=词汇性草化选择（王羲之 海/立/令） |
| `exp_multiscale.py` | G/C/S/L 四尺度分离表（MMD/Top1/AUC/CI） | G .436；C/S 零增益；L +1.9pp |
| `exp_decompose.py` | LDA 谱/分类器阶梯/脚本与聚类控制 | 质心.284<LDA.328<QDA.361<logistic.426；无结构条件化 |
| `exp_structure.py` | K-shot 原型曲线/交互增量/判别谱/log-loss | K90≥32；交互负增益；d_eff95=7；QDA log-loss 更差 |
| `exp_m1m2.py` | M1 全局偏置 vs M2 Writer×STD 双线性（留字） | ΔR²_test ≈ 0 → 交互路线关闭 |
| `exp_m1_predict.py` | M0/M1/M2 多空间预测（被 exp_structure 取代） | — |
| `prepare_ft.py` / `calibrate_lambda.py` / `build_blind.py` | FT 混合 CSV / λ 梯度校准 / 盲评拼表 | — |

## 约定

- 特征库/STD store/tables 路径均相对 `assets/u100_pretrain_20261024/`；管线一律 `status.json` 断点续跑。
- 所有诊断使用严格留字 + 按字符分组 bootstrap；噪声下限来自同 (书家,字) 重复真迹。
- 结论纪律：能量占比≠世界占比；相关须减共享项基线；准确率增益须过 log-loss/配对 CI。
