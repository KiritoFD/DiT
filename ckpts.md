# Checkpoint登记

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
