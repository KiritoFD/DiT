# Milestones — 书法生成阶段里程碑（2026-10-31）

> **只记录可解释历史转折的代表 checkpoint 与同协议指标。** 这是“策展目录”，不是把所有实验按 SSIM 排名。全实验索引见 [All.md](All.md)，活跃实验见 [Current.md](Current.md)。
>
> 资料截面：GitHub `main`（2026-10-31）；本轮只做仓库静态考古，**没有 SSH/本地 GPU 复验或下载大权重**。下表路径是记录的来源路径，**不是本次验证文件存在的承诺**。最后两天聊天中的新 OCR 数值注明“待 CSV 同步”。

## 0. 先说明评价纪律

- **不同协议不能混榜**：`docs/106moyi` 旧 **Strict** 与 `docs/107` 冻结的 **187 字 / Heun50 / CFG1 / ODE** 不是一回事；`docs/108` in-memory 口径、S/J 面板（100/187）、LPIPS-Alex/VGG、SDE/ODE 也不可混用。海报上的指标必须带 `protocol_id`、`n` 和采样配置。
- **同样的 187 张不等于 187 个严格未见汉字**：按 Seen/未见组合/未见字分层；有条件的话另建 All-Stages-Unseen。
- **比墨迹质量必须看字是否写对**：OCR Top-1/5/Margin、E-G AUC/EER/mAP、LPIPS、Ink-IoU、笔宽/异常率分开报；绝不合成一个“总分”。
- **先保留再归档**：以下“保留”是**不可删除的逻辑身份**（ckpt 或能恢复该状态的归档），不是复制 2 GB 权重多份。使用硬链接/符号链接+文件清单，先校验原件再处理；不自动清理。
- **证据分级**：A = 同协议/单变量或明确物理测试；B = 经验结果但存在混杂；C = 归档事实或候选机制、待复核；X = 失败/阴性对照，有历史研究价值。

## 1. 核心 checkpoint 白名单

| 保留级别 | 阶段/建议显示名 | 最有用的结果和协议 | 为什么必须留 | 优先的图/ckpt 来源 |
|---|---|---|---|---|
| **A** | **MoYi 12ch @50k**（252M） | 旧 Strict：SSIM **0.5935**、LPIPS **0.3724**；后续统一 187 协议 SSIM **0.5764** | 官方/复现多通道目标锚点；12ch 并非普遍无用 | `docs/106moyi/moyi/results/moyi_top10_rf/`；`docs/107/02_model_benchmarks.md` |
| **A** | **MoYi 4ch @50k/@80k**（252M） | 旧 Strict：50k **0.5949 / 0.3678**；80k **0.6085 / 0.3489**；统一 187 的 80k SSIM **0.5986** | 与 12ch 同步数对照及长训收益；**50k、80k 都留** | `docs/106moyi/moyi/results/moyi_top10_rf_4ch/eval_step{50000,80000}/` |
| A | **v10b @390k** | 早期 strict SSIM 约 **0.5058**（旧协议） | 纯 STD 骨架、无 char ID 的开集可行性概念证明 | `tools/setup_milestones_dir.py`：`01_v10b`；`docs/04_experiments/imgs/milestone_eval200_evolution_poster.png` |
| A | **v13 @125k** | 旧 strict SSIM 约 **0.5713**（版本/评测待回填） | 清洗数据、FM+轻量 AdaLN、开集路线的中间基线 | `tools/setup_milestones_dir.py`：`02_v13` |
| **X** | **v21 SkelNet @155k** | 失败/结构不稳定；需按原协议补指标 | 保留显式骨架形变失败原型，解释级联误差 | `tools/setup_milestones_dir.py`：`03_v21`（唯一旧归档需核验） |
| **A/X** | **v23 SplitLN @85k** | 骨架与风格两路联合归一化的修复；统一分数待补 | 对照 v21 归一化竞争，已测骨架方差约为风格 38× | `tools/setup_milestones_dir.py`：`04_v23` |
| B | **v54 纯字表 @150k** | `docs/108` in-mem SSIM **0.5960**；`docs/107` 统一约 **0.5890** | 建立 char-table 闭集表现和泛化成本的锚点 | `assets/results/v54_minimal_tables_noskel/`、`tools/gather_eval_for_download.py` |
| B | **v66 三表+route @150k** | 统一187 SSIM **0.5976**；in-mem **0.5955** | 小模型字表+分层条件路由；“路由单变量有效”**未证明** | `assets/results/v66_tables_condroute2456/`、`docs/107` |
| **A** | **v68 Sp/2 @200k**（59.1M） | 统一187 SSIM **0.6169**；in-mem **0.6149**；曾记录 SDE **0.6289**（另一采样协议） | 字表闭集的高质量标杆；**不能把收益独归因 C2OT/增强/SDE** | `assets/results/v68_aug_sp_c2ot/.../eval_samples_ctrl/step0200000/eval200fix/` |
| **X** | **v70 std-skel @65k** | in-mem SSIM **0.5535**，65k 后平台，约78.75k 停止 | 直接骨架通路弱表现的反例；禁止宣称所有 STD 控制上限低 | `assets/results/v70_aug_sp_stdskel_c2ot/` |
| B | **v72 Calli-VAE @50k/@65k** | in-mem **0.5999 / 0.6034**，同阶段 v68 **0.5701 / 0.5833** | 显著提升学习速度的路线，但与 noaug 等改变混杂 | `docs/108/03_metrics_ledger.md`；现存 ckpt 待盘点 |
| A | **v83-A arm4 @100k** | SSIM **0.6098** / LPIPS-VGG **0.3252** / Ink-IoU **0.3467**（该臂旧独立协议） | 无增强在四臂综合领先；否定“增强必有收益” | `docs/v83A_std_soft_report.md`，原 ckpt 待盘点 |
| A | **v84A STD-soft @200k** | SSIM **0.5846** / LPIPS-VGG **0.3551** / Ink-IoU **0.2840**（原生协议） | 取消 char-table、输入标准字图像，真正面向开集的独立成熟锚点 | `assets/results/v84A_std_eval200fix/`；`ANALYSIS/v84A_std_eval200fix_poster.png` |
| **X** | **v88 Writer×Glyph @170k** | SSIM **0.5776** / LPIPS-VGG **0.4189** | 复杂双线性交叉调制未提高泛化的长训阴性结果 | `README.md` / `docs/infra-1015-measurements.md` |
| **A** | **Deep-Sym4 @85k**（114M） | J0 的同起点参考：SSIM **0.6149**（J 面板，额外5k后） | 成熟 U-Net 生成器；**不能把 J@5k 误读成冷启动 5k** | 48 本机 `assets/results/unet_deep_sym4_114m_60m/.../checkpoints/0085000.pt`，本地曾备份，见 `ckpts.md` |
| B | **U100 Base @55k** | 已确认视觉回滚基线；需统一补 OCR/E-G | 同架构其他容量/数据配方参照 | `ckpts.md`：48 本机及 2026-10-27 本地备份清单 |
| **A** | **S0 @50k/@85k** | 50k val100：SSIM **0.60275**、LPIPS **0.33950**，E-G AUC **82.13%**；85k OCR 新配对 **81.29%**（待 CSV 入库） | 简单 Writer Vector+Global AdaGN；内容强基线，**两个步数都留** | `ckpts.md`，`docs/1028/raw/S0/`，`p125_metrics_full_4090.csv` |
| A | **S4 @50k** | val100：SSIM **0.59875**、Ink-IoU **0.29513**、E-G AUC **84.91%** | Writer×LocalCNN + Spatial AdaGN，局部条件对照 | `ckpts.md`；@85k **未完成（约51k），不得标成结果** |
| **A** | **S5 @50k/@85k** | 50k val100：SSIM **0.59750**、E-G AUC **87.98%**；85k heldout187 E-G AUC 三 seed **87.68/84.17/86.17%** | 局部 Attention×CrossAttn 风格强候选；内容并未同步领先 | `ckpts.md`；`docs/1031.md`；48 本机 S5@85k |
| A | **J0 @5k / J2 @5k** | J0：SSIM **0.6149** / LPIPS **0.3220**；J2：**0.6198 / 0.3150**（187字） | 同 Deep85 起点下测试可训练 Decoder+图像损失；两个状态都留 | `docs/1028/raw/J{0,2}/`、`docs/1028/posters/J_comparison.png` |
| B/X | **J5/J6/J7 @5k** | OCR heldout187 **81.4%–82.6%**（新实测待 CSV 入库）；三臂差异不明显 | Decoder Writer/STD 条件与是否回传梯度的阴性对照；至少留 J5、J7 完整原件 | `docs/1029/eval_unpaired/REPORT.md`，48 `assets/style_matrix_20261028/generation2full/48/runs/J*/` |
| **A** | **P75：P0@50k/P1@55k/P2@70k/P3A,P3B@75k** | val100 SSIM **.5573→.5636→.5857→.5857/.5881**；P2 E-G AUC **77.77%** | 同一预训练链各阶段逻辑不可合并；保留 P0、P2、P3A、P3B，P1 至少保留可恢复状态 | `1030p125.md`、`1031.md`；4090 `assets/p125_20261030/` |

**未识别的历史名称 `13ecosx`**：当前 GitHub 默认分支的代码搜索没有找到该精确字符串。**保留待定位项，不擅自等同于 v13 或 C2OT**；找到 run 原目录、config、checkpoint identity 后再加入白名单和 poster。

### 2026-10-31 OCR 新评测（已从对话获取，**不是本次仓库/GPU复核**）

- 修正了 PNG 字典序与 PIL 宽高错误；权威逐样本映射来自 `eval_stdskel_batch.csv`，不能直接按外部 CSV 行序对齐。
- S0 50k 71.11% → 85k 81.29%（**先核对两端面板是否同源，再声明配对差**）；S4 50k 77.78%；S5@85k 75.4–79.0%（三 seed）；J5/J6/J7 heldout187 81.4–82.6%；P3A 新 OCR 值待原始 CSV 一并归档。
- 正式 OCR 结果必须保存：`eval_id`、`panel_id`、目标字来源、GT Top-1、生成图 SHA、OCR 权重 hash、Top1/Top5/Margin；**GT sanity 高不保证跨历史模型协议完全一致**。

## 2. 真正应当保存到本地的东西

首选保存每个里程碑的 **20 张统一样本生成 PNG + 20 张对应 STD/GT + 逐图 CSV + 配置/identity + 汇总 JSON + poster**。主 checkpoint **不批量搬运**；已经存在的 48→本地核心备份（参见 `ckpts.md`）登记为备份，不再重复传。

推荐本地布局（仅示意，目前未创建实际图片）：

```text
assets/milestone_gallery/
  panel_eval200fix_v1/
    manifest.json                 # 每个 sample_id 与每一行模型的精确映射
    {stage_id}/                  # 20 张 PNG，小文件而非整个 eval
    posters/
      historical_20.png
      recent_style_20.png
```

已有轮子（**优先复用**）：`tools/setup_milestones_dir.py`（旧阶段 ckpt meta）、`tools/gather_eval_for_download.py`（历史图收集）、`tools/make_poster.py`（Pillow 拼图）、`tools/infer_milestones_eval200.py`（旧模型跨接口推理）、`docs/1028/posters/*.indices.json`（已冻结选择）。

**注意：旧 `infer_milestones_eval200.py` 使用 `strict=False` 加载并对若干未识别 Writer 回退到 0，且取样/解码参数需要跟原 checkpoint 对齐。它生成的“统一图”不能未经 strict 加载与条件映射审计就当定量证据。** 可以复用它的阶段清单，但正式指标必须以完整评测契约重跑。

## 3. 补指标队列（不在本次文档任务里占 GPU）

| 批次 | 输入与工作 | 输出及验收 |
|---|---|---|
| Q0 CPU | 枚举白名单 ckpt：路径、真实文件/符号链接、`.done`、配置哈希、EMA、Decoder、VAE、字典、源代码快照 | `checkpoint_inventory.csv`，状态区分 **verified / archived-only / missing** |
| Q1 CPU | 将现成评测 `assets/ledger/points_all.csv`、`docs/106moyi`、`docs/1028/raw`、`docs/1030`、P75 OCR CSV 按 `protocol_id` 归档 | `metrics_long.csv`；不同协议不覆盖同一个槽位 |
| Q2 GPU | 对**能严格加载**的旧阶段权重，在当前统一 187（再单列真未见字）下重跑 3 seed、OCR、E-G、SSIM/LPIPS/Ink-IoU | 带图/GT 字符 sanity、输出 hash、逐图 CSV 的 `backfill` 目录 |
| Q3 CPU | 每阶段采相同 20 个样本，产出带 **显著行标题 + 方法 + 同协议指标**的 poster；抽查 g/gt/std、繁简/Unicode 及索引映射 | 本地 `assets/milestone_gallery/`，不拿缺图/错位样本填黑充数 |

Q2 各模型 VAE、预处理、Sampler 和 EMA 可兼容性不同；如果不能完全保持生成定义，应明确标注为 **replication**，不要悄悄用今天的 VAE 替换历史 VAE。

## 4. 指向原始证据

- [MoYi 两个通道基准与归档](docs/106moyi/README.md)；[跨协议更严格对照](docs/107/02_model_benchmarks.md)；[关键历史勘误](docs/107/03_evidence_grade_and_errata.md)。
- [13 纪元编年史](docs/04_experiments/00_exhaustive_experiment_chronicle.md)；[早期七里程碑](docs/04_experiments/08_milestone_evolution_summary.md)。
- [全量实验总账](exp.md)；[全部指标曲线与来源审计](docs/108/03_metrics_ledger.md)；[阶段权重清单](ckpts.md)。
- [S/J 控制实验](docs/1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)；[风格独立性审计](docs/1030/eval_style_v2/FINAL_REPORT.md)；[P75 实际训练边界与口径](1030p125.md)。
