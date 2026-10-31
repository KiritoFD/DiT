# Current — 当前实验簇暂存区（2026-10-31）

> 此文件只写**现阶段正在推进/等待验收**的实验，不把计划写成完成。长期记录转入 [All.md](All.md)，正式候选入 [Milestones.md](Milestones.md)。机器统一使用 **4090**；48 的已有文件只读、无新训练或大文件传输。

## 当前结论
- 最近的 OCR 识别结果使用修复后的 **样本 idx → 字符**映射；不要再用 `sorted(glob("g*.png"))` 的字典序或外部 CSV 行顺序。旧 0%/7.8% 是配对/方向错误，非模型能力。
- 现有 P75 是 **P0 50k + P1 5k + P2 15k + P3 5k = 75k**；原日志 `@120k/@125k` 是旧计划命名，非真实累计步数。
- P75 预训练是否有净收益尚未通过相同 Top10 曝光/总算力与 All-Stages-Unseen 严格验收。
- S0@85k 的内容 OCR 可识别性较强；S5 更倾向 Writer 身份，但两机数据/Seed 尚不能给跨模型定论。
- J2 解冻 Decoder 的图像质量正效应有证据；是否有无参考字符正确性/Writer 收益，尚未证实。

## 实验簇 C31：4090 后训练（**计划，未启动**）

| Arm | 起点 | 实际新 Steps | 唯一主变量 | 需保存 |
|---|---|---:|---|---|
| A 延长旧流程 | P75-P3A@75k | 25k | 新 LR horizon 继续 Top10-only FM | +5k/+15k/+25k |
| B 直接 Top10 | P0@50k | 25k | 跳过 5k 70/30 混合阶段 | +5k/+15k/+25k |
| C Writer-first | P0@50k | 5k+20k | 前段 writer 条件高 LR、主干低 LR | +5k/+15k/+25k |
| D0 常规图像域对照 | A/B/C 统一最佳起点 | 5k | FM，不动 Decoder | 起点/终点 |
| D1 Decoder 适配 | **与 D0 同一个起点** | 5k | FM+可训练 Decoder+弱图像损失 | 起点/终点；配套 decoder |
| D2 E-G pilot（可选） | 与 D0 同一个起点 | 5k | 弱 E-G style loss | 仅独立 Writer 验收通过后开启 |

**前置审计**：从原 checkpoint 完整读取模型/Optimizer/EMA/RNG；冻结数据顺序、真实 LR group、Writer map、STD/VAE、noise seed；A/B/C 面板/时间窗一致。A vs B/C 的 Top10 曝光不同；不可说成完全单变量。

## 评测契约（所有新 checkpoint）

- 统一 heldout187 **×3 seeds**，GT/字符从 **该 eval 的 `eval_stdskel_batch.csv` 或严格 manifest** 读取，逐图校验对应样本 ID、Unicode/异体、书家、书体；同时保留原 val100 的历史指标，禁止跨集合混比。
- **OCR**：Top-1、Top-5、Target Margin、GT sanity 和 OCR 模型 hash。**E-G**：AUC/EER/mAP、GT ceiling、同面板 normalization；**图像**：SSIM/LPIPS（标注 Alex 或 VGG）、Ink-IoU、笔宽、墨量、Near-blank；**Writer-swap**：同 STD+noise 仅换 writer。
- 每条 run 导出：`run_identity.json`、`eval_manifest.csv`、`metrics_per_image.csv`、`metrics_summary.json`、`20_selected.json`、`poster_20.png`。
- 统一对照使用按字符聚类的 bootstrap；不同 seed 重复字符不当作独立 3N。

## 当前归档/补评任务（**pending**）

1. **先盘点** Milestones.md 白名单 checkpoint 的真实目录、大小、损坏/缺失、软链接与 .done；不要自动删除/移动。
2. **补历史 OCR/E-G**：先不重新采样，扫描已有 PNG；严格验证 OCR 预处理与样本序号，在有效字段齐全时回填，否则标记 `needs_reinfer`。
3. **统一旧 eval**：MoYi 12/4、v10b/v13/v21/v23、v66/v68/v70/v84/Deep/S/J 等不同架构必须按原 VAE/条件映射载入；不能用 `strict=False` 吞键后报正式新成绩。
4. **本地 gallery**：只拉 20 样本 PNG、STD/GT、每图 CSV 和 poster；大权重遵从 [AGENTS.md](AGENTS.md) 默认不搬运。已有 Deep85 等本地备份避免重复。
5. **poster 脚本**：`tools/milestone_poster.py` 从 JSON manifest 生成 20 列、每模型**醒目行标题+方法+同协议指标**；检查 row/column 键、图路径，缺失直接失败，输出 PNG 和冻结选择 JSON。

## 交接记录格式

```text
日期 / 机器 / run_id / 真实累积步数 / parent checkpoint
数据合同+LR+代码快照 / 是否 running-complete / 是否 checkpoint .done
评测面板、seed、OCR/Writer GT ceiling / 关键结果与逐图索引
与哪一臂是 matched 变量 / 结论分级 / 下一动作
```

**禁止**：对仍在运行的任务误判完成、重复传输 GB 权重、自动删除唯一副本、给缺失评测补估算值、把不同 SSIM 实现聚合成一个字段。

## 簇 C31 实际启动状态（2026-10-31 晚，4090 已运行）

前置审计与计划的差异：为落实 A/B/C 的 restart 语义，修了三处运行时（§8.2 of All.md：rate() 字段错位、restart base_lrs、跨拓扑 resume、LR 视界校验）。**style LR 因此与原 P125 实测口径不同（本轮真正按 spec 生效）**。

| Arm | 起点（fork 实现方式） | 实际 counter | LR (backbone/style_repr/style_inj/writer_table) | 状态 |
|---|---|---|---|---|
| A | P3A@20000（serial fork symlink + resume，schedule_restart origin=20000 horizon=25k） | 20000→45000（25k/35k/45k 三评测点） | 1e-5 / 5e-5 / 5e-5 / 2.5e-5 | **训练中** |
| B | P0@50000（fork + resume，restart origin=50000 horizon=25k） | 50000→75000 | 同 A | 排队 |
| C1 | P0@50000（restart origin=50000 horizon=5k） | 50000→55000 | 2e-6 / 5e-5 / 5e-5 / 1e-4 | 排队 |
| C2 | C1@55000（restart origin=55000 horizon=20k） | 55000→75000 | 同 A | 排队 |

- 已完成：**A@25000 评测点**（val100 SSIM 0.5914 / LPIPS 0.3478 / Ink-IoU 0.2829 vs P3A@20000 的 0.5857/0.3528/0.2725，四项全改善，旧配方延长仍在收敛）；heldout187×3 seeds 面板已出。本地 20 样本补 OCR Top-1 80%、SSIM(g,gt) 0.668；poster `assets/milestones_posters/A_25k.png`。
- 串行：`assets/abc_20261031/plan.json`（--resident-pid 761xxx 轮换预留），1.46s/步，总 ETA ~34h。评测后自动落 `evaluations/{stop:07d}/{protocol}`。
- 累计口径换算：A@45k=用户口径 100k；B/C2@75000=75k（与上文 P75=75k 口径一致）。

### poster/补指标管线（与 Milestones.md §2/§3 对接）

- 快速路：`tools/make_ckpt_poster.py`（目录内 g/gt 数值序采样 20 张 + 大字横幅）+ `tools/milestones/supplement_metrics.py`（本地 GPU OCR + SSIM）；产出已入 [Milestones.md](Milestones.md)。
- 严格路：`tools/milestone_poster.py`（manifest 驱动，协议 ID 校验）用于正式同协议对比海报；两者并存，正式结论以严格路为准。
