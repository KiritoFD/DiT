# 评价器、数据合同与重大勘误

> 核心论断：**实验结论的最大非模型风险是样本/条件映射错误和评价协议漂移**。本页列明已证实的事故，以及代码审阅中仍存在、但没有被证明已影响当前分数的风险。依据 [1031原始记录](../../1031.md)、[107证据勘误](../107/03_evidence_grade_and_errata.md)、[E-G最终报告](../1030/eval_style_v2/FINAL_REPORT.md)、[OCR代码4090](../../tools/p125/ocr_panel_eval.py) / [48](../../tools/p125/ocr_panel_eval48.py)。

## 1. OCR 不再是随机初始化ResNet，而是中文预训练识别器后训练

现有读光识别器为 ModelScope 中文OCR **ConvNextViT/ViTSTR**，真迹上微调5k，使用 **peak-timestep CE** 的类别输出；与 CTC greedy 解码不是同一指标。ModelScope底模 vocab 原始覆盖 **4,908/7,010** D0清理字符，扩展头后必须从checkpoint及配套 vocab 确认覆盖和类别映射。

D0已核查：**151,657** 条去重标签、**7,010** 独立Unicode；冻结的 Real-Test 候选约 **6,975** 图（6个整书家留出+Top10分层）。但目前可靠公布的微调验收只是 **200张** Real-Test：Top1 **80.0%**，楷79.2%(101)、行77.9%(68)、隶87.1%(31)。**不能表述为6975张全量80%**。

OCR三类用途严格分开：
- **可识别**：生成Top1/Top5、target margin、GT sanity、有效 `n`、词表覆盖；
- **部件/断笔正确**：真实误字、缺横、错偏旁、连笔/断笔 Corruption-Test 与人工复核；目前**没有完成**可靠的召回/误报曲线；
- **跨字符与跨来源泛化**：writer/script/source/字复杂度分层；不能因为GT图识别率89%–100%就假设所有书体都已经校准。

## 2. 三个已经造成错误结论的输入事故

1. `sorted(glob('g*.png'))` 字典序 `g0,g1,g10,...,g2` 与CSV顺序配错；原0%/7.8% OCR**完全作废**。
2. PIL `resize((32,192))` 实参为 (width,height)，把字符拉成横条；已修为高32、动态宽。
3. 某些J/S面板生成图顺序与外部CSV行序不同；`samples.jsonl.sample_id` 与CSV `img_id` 也可能不同域。**权威配对应来自该eval本身的 `eval_stdskel_batch.csv` idx↔`g{idx}.png`↔目标字符**。修后S0@85k seed0 h187 Top1 **81.29%**，旧外部CSV配错得到79.65%。

更重要的是，**GT sanity 89%–100%只是必要条件**。少量繁简异体、缺图、重复索引仍可能掺在高GT正确率中；只有精确JOIN和每图哈希/Unicode检查才能排除。

## 3. 目前OCR脚本尚存的审计风险（**建议修改，非宣称已触发**）

检查 [`tools/p125/ocr_panel_eval.py`](../../tools/p125/ocr_panel_eval.py) 和 [48版本](../../tools/p125/ocr_panel_eval48.py)：

- `net.load_state_dict(ck['model'], strict=False)`：可能吞掉缺失参数。正式验收改 `strict=True` 或断言 missing/unexpected keys为空，记录checkpoint、vocab哈希。
- `score(net, pngs, rows)` 使用 `zip` 位置对齐；`min(len(gens),len(gts),len(rows))` 会**静默截断**，缺失 `g{i}.png` 时即使数值排序也可能重新错位。改为 `by_idx` 对 `g{i}`/`gt{i}`/`eval_stdskel_batch.csv` **集合相等的强制JOIN**，缺任一失败。
- 旧文件没有batch CSV时回退外部CSV行序，会重开已踩过的坑。对“新正式评测”应**fail-closed**，旧历史只走经人工核验的兼容清单。
- vocab扩展之后必须校验 index从1开始/blank id / Unicode映射双向 round-trip，标签不在 vocab 的样本显式记 exclusion。
- 当前以时间步最大类别logit选择识别时间步，然后对该logit做softmax，不是标准序列概率校准；可以报Top1和Margin，但**未经独立校准不把max prob称为真实置信度**。

这些风险**没有在本轮重新运行或修复**。重测指标时须指定新的 `evaluator_version`，历史数据应保留、不能悄悄覆盖。

## 4. E-G 强在哪里，弱在哪里

[冻结E-G真实验收](../1030/eval_style_v2/FINAL_REPORT.md) 同书体、异字符、1008真迹查询、K64参考：AUC **95.98%**，EER **7.44%**，mAP **93.09%**；低阶20维墨量/宽度/布局线性探针 AUC **49.35%**，残差化后 E-G仍 **93.11%**。

这可以支持：E-G 提取了跨字 Writer 身份，而不是简单依赖上述20项低阶统计。**不可以支持**：完全排除扫描纸张/作品来源共线、证明书法审美完美、把二分类AUC当Writer十分类准确率。两机同样数据同样权重逐位一致属于**管线复现**。

HWD官方特征在真实中文书家核验 AUC约 **51–54%**，不具备足够独立Writer裁判能力；HWD-Ridge基于CHAED标签的审美回归 MAE **17.9686** vs均值 **18.3218**、PCC **.0713**，未达到可靠评分，不能据此选最美模型。

E-G对字符检索：913个跨书家同字query、120字符，Top1 **13.14%**。远高于1/120随机水平，**不能说完全丢弃字符信息**；但不够作为OCR。

**注意**：E-G AUC/EER/mAP、Style-FID/KID使用同一网络特征，属于多种统计而不是四个独立“裁判”。P3B也用E-G作训练损失，必须将训练教师自身分数列为**诊断性**，以来源留出、未见字/独立Writer指标与swap作实质验收。

## 5. 严格划分四个“未见”，审计当前冲突

- `Unseen-combination`：目标 `(char,writer,script)` 未出现在生成器训练；
- `Unseen-in-Top10`：Unicode未在任何Top10训练图中出现；
- `Unseen-in-All-Stages`：Unicode在P0通用+P1+P2+P3所有图像标签中都未出现；
- `Source-held-out`：同一作品/扫描源未在任何训练阶段出现（必要时“同书家不同作品”也必须单列）。

旧187评测有**75/187训练见过的组合**，不能叫187全严格未见。[107](../107/03_evidence_grade_and_errata.md)。

**一个未解决的真实文档冲突**：`1030p125.md` 声称 val100“全部字符在Top10训练集内”，而 `docs/1030/eval_style_v2/FINAL_REPORT.md` 记为 **48个seen/50个unseen（有效98）**。可能是训练阶段/Writer组合定义/CSV版本不同，也可能是记录错误；**在重建训练字符并集、对照CSV sha前不选择一边**。新评测必须输出 `seen_top10`、`seen_P0`、`seen_all_stages`、`same_writer_script`、`same_work_id` 五列，再修正旧报告。

仅“路径无重叠”不能确认同一张扫描的不同裁剪/重编码或同作品没有泄漏。OCR D0已消除三语料的相同路径，并不证明视觉指纹、作品来源级去重已完整。

## 6. 还存在的“数字相同名”事故

| 事故 | 正确解释 |
|---|---|
| S0@85k SSIM **.5963** vs **.6081** | 同批图，但 serial in-memory vs eval_p125/skimage实现不同；不合并均值，原比较用serial |
| LPIPS-Alex与VGG | 两个不同特征网络，不能填同列 |
| MoYi 12ch `i.png` vs `t{i}.png` | `t{i}` 是骨架图，误用会虚高评价；取真实生成的 `i.png` |
| MoYi 4ch80k vs12ch50k | 非等训练预算；4ch同50k只领先SSIM约.0014 |
| v68 Heun50 vs MoYi Euler50 | 前者约100NFE，后者50NFE，非等推理算力 |
| 标记 `eval200fix` | 有时是val100、h187或与外部CSV不一致；目录名不构成数据集ID |
| `S0@50k OCR 71.11` vs `S0@85k OCR 81.29` | 原始面板分别 val100 与 h187：**不能计算+10.18pp作为85k的训练收益** |
| `P2@120k` vs实际P2@70k | 保留旧命名作别名，实际更新计数必须从checkpoint和训练阶段链求出 |

更完整的边界见 [证据表](01_EVIDENCE_MATRIX.md)。

## 7. 推荐的“严合同”输出

每个评测点强制关联下列身份：

| 类别 | 最低字段 |
|---|---|
| 运行 | run_id、parent_ckpt SHA、实际累计更新、EMA/raw、model/config/source hash |
| 数据 | train/eval CSV SHA、所有阶段char集合hash、Writer remap、Script、作品来源 |
| 条件 | STD renderer+TTF hash、VAE/Decoder hash、g/gt图片hash |
| 采样 | sampler、NFE、CFG、SDE、noise seed、显式 `protocol_id` |
| 逐图 | `idx,char,writer,script,gt_path,g_path,source_id,seen_flags` |
| OCR | 权重/vocab/解码版本、有效n、GT Top1/5、生成Top1/5/Margin、排除原因 |
| E-G | 特征权重、K、参考池、GT AUC ceiling、pooled/macro AUC/EER/mAP |
| 画质 | SSIM窗口、LPIPS-Alex/VGG、InkIoU阈值、宽度/Frag/连通度 |
| 统计 | 主要终点、paired effect、按字和Writer分层CI、seed、预设失败阈值 |

严格的命名与目录契约应让“同一模型、同一批图、同一评估器”复算逐位一致。20字poster是可视化，**不代替187×3或冻结test**。必要时先CPU侧重建manifest，再进行GPU重评；不靠肉眼选择方便的正确GT。

数学统计入口：[math/04](../math/04_TRANSFER_EXPOSURE_AND_STATISTICS.md)；运行执行入口：[05下一阶段](05_NEXT_EXPERIMENTS.md)。
