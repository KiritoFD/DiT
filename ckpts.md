# Checkpoint登记

2026-10-29 20:28：本次仅归档明文报告；SSH入口关闭，未核验新增checkpoint，不宣称60k或更晚权重已存在。55k完整性沿用19:22已确认记录。

2026-10-29 19:22：新增4090 `assets/style_focus_85k_20261029/4090/runs/S0/checkpoints/0055000.pt`（1,827,105,304 bytes）及48 `assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/0055000.pt`（1,847,756,091 bytes），原子.done记录step55000，未复制或覆盖原50k。官方HWD权重在两端各自下载至`~/.cache/torch/hub/checkpoints/VGG16_class_10400.pth.part`，707,488,473 bytes，严格加载10400类完整state成功；扩展名.part不代表未完成。它仅用于冻结评测，非生成模型或新风格教师。CHAED Ridge仅为失败的探索回归器，不进入生成训练。

10-29本轮64h状态核查只读：S9约45.65k、S11约15.05k；未来匹配LR spec尚由边界交接生成，未宣称其权重已存在。未迁移或覆盖checkpoint。

2026-10-29后续阶段修正：新增S10、S0/S4、S11的`*_60_matched`/`*_75_matched`未来身份链，对齐S9现有两段LR。当前24h checkpoint及当前S11冷启动权重保持原身份；检测到另一成熟轨迹已开始时不回滚，交接器登记deferred。

2026-10-29阶段归档：S0–S7@50k及J0–J3@5k的指标副本进入`docs/1028/raw/`，报告未声称其逐run完整代码commit已确定。四个历史实现短SHA已在本地Git对象库解析为完整SHA；checkpoint权重未迁移、未覆盖。新J4–J7@5k已完成，当前48推进S11。

2026-10-28 23:20：新后台队列确认推进；4090 S9从完整403步维护存盘恢复后越过500，48 J4从完整1,675步恢复后越过1,700。后续S9-FM/Pair均由同一个S9@50k本机只读软链接分叉，保留Optimizer/EMA/RNG；J4–J7维持各自Deep85独立起点。未跨机搬运权重。

2026-10-28 22:28：旧J0延长已安全停止，48保存`assets/style_matrix_20261027/matrix64/48/execution/runs/J0/checkpoints/0011654.pt`，SHA `9920864ca489ce98633cf13483d1df4f1b9c13c4ef298a0616400dcb3f7b13ab`。J3@5k与全部历史权重保留。新轮smoke权重仅为链路验证，不作科学训练结果或正式起点；S9随机初始化，J4–J7均独立加载Deep85 EMA和原VAE。

更新时间：2026-10-28 21:30（Asia/Shanghai）。延长轮结果聚合见[docs/1028](docs/1028/README.md)，实时状态见[training.md](training.md)，完整历史索引见[exp.md](exp.md)。

## 路径与检查口径

下文路径均相对对应机器仓库：4090为`/root/Workspace/xy/DiT`，48为`/home/ds/Workspace/DiT`。新计划根目录为`assets/style_matrix_20261027/matrix64/{4090,48}`。

封存口径变更（2026-10-27）：>1MB大资产在ready.json只记path+size不哈希；小文件（spec/代码/remap/CSV头/calibration）保留SHA256并在每次启动及每任务前校验。旧计划`cold50k/4090`、`deep85_decoder/48_v2`的ready为全量哈希口径，均已停止、保留存档。

## 当前实验已保存（2026-10-28 16:30，matrix64计划）

| 机器 | 实验 | 进度与已落盘权重 | 用途 |
|---|---|---|---|
| 4090 | S0 | 已完成50k；1k及每5k checkpoint保留 | 50k EMA已评测100字 |
| 4090 | S1 / S4 | 均完成50k且`.done`、100字EMA评测齐全 | S1空间AdaGN；S4局部表征+空间AdaGN |
| 4090 | S6 | 已完成50k、`.done`及100字EMA评测 | Vector+DynamicConv |
| 4090 | S8 | 训练至40,400+；最新完整checkpoint 40k | Local CNN+DynamicConv进行中 |
| 48 | J0/J1/J2/J3 | 各完成5k并有`.done`与187字EMA评测 | J2/J3评测使用各自checkpoint的Decoder EMA |
| 48 | S2 / S3 | 均完成50k且`.done`、100字EMA评测齐全 | Vector/Multi-Token的Cross-Attention对照 |
| 48 | S5 | 已完成50k、`.done`及100字EMA评测 | 局部表征+Cross-Attention |
| 48 | S7 | 已完成50k、`.done`及100字EMA评测 | Multi-Token+DynamicConv；结果已入docs/1028 |
| 48 | J0延长 | 从5k完整checkpoint恢复，已越过5,150 | scheduler键名兼容修复后继续到12k |

精确路径均相对对应机器仓库：

```text
# 4090
assets/style_matrix_20261027/matrix64/4090/execution/runs/S0/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/4090/execution/runs/S1/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/4090/execution/runs/S4/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/4090/execution/runs/S6/checkpoints/0050000.pt
# 48（将J0替换为J1/J2/J3可得到对应路径）
assets/style_matrix_20261027/matrix64/48/execution/runs/J0/checkpoints/0005000.pt
assets/style_matrix_20261027/matrix64/48/execution/runs/S2/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/48/execution/runs/S3/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/48/execution/runs/S5/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/48/execution/runs/S7/checkpoints/0050000.pt
```

S0@50k评测清单记录checkpoint SHA256 `50d41ee21091f865f3b7315297adeafbba113d934d3094e7203c1e0017a2cccc`；J2@5k为 `30487452a1a790d9e7083dc3e28e1aa7f9242e087f723add516188b2fa11b65c`。本次核查`.done`与评测完成清单，未对GB级文件重新哈希。

外部J2映射内部目标J3；外部J3映射内部目标J4（图像监督+可训练Decoder、无GT重建）。J2/J3的raw/EMA Decoder保存在对应checkpoint中；后续延长必须恢复模型、Optimizer、EMA、RNG并保留身份映射。

2026-10-28 20:50左右，J0延长首次恢复在读取scheduler LR时触发`KeyError: last_lr`。checkpoint本身完整，实际字段为PyTorch标准`_last_lr`。代码改为兼容`_last_lr/last_lr`并校验LR组数，更新ready中该源码SHA后重新启动；模型、Optimizer、EMA、RNG及实验identity仍从原5k checkpoint恢复，没有产生替代checkpoint或绕过完整性校验。

J0@5k的评测随旧计划迁移：`complete.json`内仍保留旧`48_v2`目录的Decoder路径，该旧路径现已失效。配套Decoder文件已位于当前J0评测目录的`paired_vae/c7e005c1…/decoder_ema`，其`identity.json`中的checkpoint SHA与当前J0 `.done`一致；后续使用时按当前checkpoint重新导出或解析现有目录，不要直接打开旧路径。本次未重新哈希Decoder大文件。

## 成熟模型与参考权重

| 机器 | 模型 | 现存步数 | 本轮用途 |
|---|---|---|---|
| 48 | Deep-Sym4 114M | 50k、75k、85k | **J组统一起点85k EMA**；三个checkpoint全部保留（未移动） |
| 48 | U100 Base 72M | 55k | 历史视觉参考（未移动） |
| 48 | Qwen condition A / C | 90k | 历史条件消融（未移动） |
| 4090 | P1 | 10k | 保留；S组从零训练不加载。**2026-10-27归档至`assets/_archive_20261027/u100_p1p2_20261025/p1_runs/`，永不删除** |
| 4090 | P2 | 20k | 保留；J组从Deep85开始不加载。**同上位于`_archive_20261027/u100_p1p2_20261025/p2_runs/`** |

归档明细与清理确认门见[Disk.md](Disk.md)；数据资产与重建配方见[Data.md](Data.md)。

## 48→本地关键权重备份（2026-10-27）

本地目标为`G:\DiT-48-artifacts-20261027`，完整白名单及远端绝对路径见[下载清单](ANALYSIS/remote48_backup_20261027/manifest.json)。Deep-Sym4 50k/75k/85k、U100-Base 55k、J0/J1/J2/J3各5k、Qwen-condition A/C各90k及69MB元数据包已于10-28 12:42全部落盘；总计20,020,117,802 bytes，逐项尺寸核验通过并生成`completed.json`。未纳入可重建数据、cache、逐样本PNG和普通中间权重。S组新50k和J组@12k属于后续追加项。

## 数据资产与配对

- `p2_train.csv`两机SHA一致（`976b573d…`）；`generic48_kxl_top10_20261022/std_bbox64/index.csv`两机SHA一致（`c02cef37…`）——S组协议同源。
- 48的`u100_pretrain_20261024/`此前缺val100.csv、char/font remap（KB级补齐）与std_shape bank。bank按[Data.md](Data.md)配方在48**本地重建**（build_rasters逐位校验+precompute_bank，seed90），**重建结果与4090的bank SHA完全一致**（`bc3db05f…`）——零GB级传输达成同协议。曾尝试的2.8GB跨机中转已中止并登记为反面案例（见[AGENTS.md](AGENTS.md)数据规则）。std_shape_bank.pt为>1MB资产，按新口径只登记path+size。
- J组沿用旧校准文件`deep85_decoder/48_v2/calibration.json`（未改动，路径不变）。

## 排除项（沿用）

旧计划目录曾按[Disk.md](Disk.md)归档。4090的`cold50k/4090`及P1/P2关键归档仍保留；48旧归档已于15:27按用户确认删除，但`deep85_decoder/48_v2/calibration.json`与`train_unaugmented.csv`始终留在原位，当前`matrix64`中的J checkpoint副本也完整保留。48不完整P1传输残片已删除，4090完整P1@10k/P2@20k不受影响。`matrix64`计划的probe日志继续留作吞吐证据。
## 2026-10-29 15:46 暂停扩展并汇总

按用户要求暂停H0/H1实现与启动；双机controller和later观察器SIGSTOP，当前S11训练、S9评测子任务保留，之后不再自动接下一实验。4090 S9已完成50k：val100 SSIM0.59814/LPIPS0.34543/Ink-IoU0.27672，未超过S0；48 J4–J7各5k高度重合，Decoder晋级为空，S11约44.3k/50k，纯训练剩约1.2h。共享训练面板方向cosine约0.886不是留出风格泛化证据；187字差分n_cells=0。完整数据、实际读图、边界及暂停PID见[10-29汇总](docs/1029/RESULTS_AND_PLAN_REVIEW.md)。旧自动队列ETA暂不适用，恢复前重新定计划。

## 2026-10-29 新评价与S11终点

48 `assets/style_matrix_20261028/generation2full/48/runs/S11/checkpoints/0050000.pt`已完整落盘，step=50,000，size=2,044,369,601 bytes。读取既有`.done`的登记SHA256：`47221eccda23b2a1e97469bf7594707933ad8ddefdaea5600d0a84e3acc3e691`，未重新哈希大文件。

24个权重终点的现成PNG已重评；历史P1/P2/Deep75/85/U10055通过逐张GT RGB像素匹配确认语义，保留原protocol.json的EMA、采样和STD身份。[完整报告](docs/1029/eval_unpaired/REPORT.md)。未移动或修改旧权重，无两训练机之间的权重互传。S10无权重，H0/H1未启动。
## 2026-10-29 书写领域指标首轮

已完成缓存特征上的Style-FID/KID、Writer Verification池化AUC/EER、书家宏平均检索mAP，以及k=5 Density/Coverage；两端complete均存在。按用户要求先交付已算指标，暂停扩展实现。官方HWD、CHAED/TabNet尚无分数，不混入结果表。[已完成指标报告](docs/1029/domain_metrics/REPORT.md)，原始统计在同目录及远端`assets/eval_domain_20261029/`。S5的E-G身份核验点估计最好；S7池化Style-FID更低但身份核验不是最好；J2分布有小幅改善，J4–J7仍接近。未进行新生成器训练、未清理权重，无GB级跨机传输。
## 2026-10-29 S0/S4/S5@85k计划

各自本机50k checkpoint通过软链接作为新run起点，完整恢复raw/Optimizer/EMA/RNG；原权重不覆盖、不互传。新权重将写入4090 `assets/style_focus_85k_20261029/4090/runs/{S0,S4}/checkpoints/`及48 `assets/style_focus_85k_20261029/48_v3/runs/S5/checkpoints/`，55k起每5k保存至85k。目前只确认已越过50k，未把未落盘步数当作checkpoint。

48旧`48/`、`48_v2/`为保存/评测interval启动校验失败记录，均无optimizer更新；v3统一eval_every=ckpt_every=5000后已正常推进。[身份与计划](docs/1029/style_focus_85k/README.md)。
