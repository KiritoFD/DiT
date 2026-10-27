# Checkpoint登记

更新时间：2026-10-27 15:11（Asia/Shanghai）。延长轮已实现并重新拉起：实施记录见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)，实时状态见[training.md](training.md)，完整历史索引见[exp.md](exp.md)。

## 路径与检查口径

下文路径均相对对应机器仓库：4090为`/root/Workspace/xy/DiT`，48为`/home/ds/Workspace/DiT`。新计划根目录为`assets/style_matrix_20261027/matrix64/{4090,48}`。

封存口径变更（2026-10-27）：>1MB大资产在ready.json只记path+size不哈希；小文件（spec/代码/remap/CSV头/calibration）保留SHA256并在每次启动及每任务前校验。旧计划`cold50k/4090`、`deep85_decoder/48_v2`的ready为全量哈希口径，均已停止、保留存档。

## 当前实验已保存（2026-10-27 15:11，matrix64计划）

| 机器 | 实验 | 进度与已落盘权重 | 用途 |
|---|---|---|---|
| 4090 | S0 | 已完成50k；1k及每5k checkpoint保留 | 50k EMA已评测100字 |
| 4090 | S1 | 训练至23,200；最新完整checkpoint 20k | 继续至50k，日志步数不等于已持久化步数 |
| 48 | J0/J1/J2/J3 | 各完成5k并有`.done`与187字EMA评测 | J2/J3评测使用各自checkpoint的Decoder EMA |
| 48 | S2 | 训练至36,150；最新完整checkpoint 25k | 继续至50k；保存节奏为5k/10k/25k/50k |

精确路径均相对对应机器仓库：

```text
# 4090
assets/style_matrix_20261027/matrix64/4090/execution/runs/S0/checkpoints/0050000.pt
assets/style_matrix_20261027/matrix64/4090/execution/runs/S1/checkpoints/0020000.pt
# 48（将J0替换为J1/J2/J3可得到对应路径）
assets/style_matrix_20261027/matrix64/48/execution/runs/J0/checkpoints/0005000.pt
assets/style_matrix_20261027/matrix64/48/execution/runs/S2/checkpoints/0025000.pt
```

S0@50k评测清单记录checkpoint SHA256 `50d41ee21091f865f3b7315297adeafbba113d934d3094e7203c1e0017a2cccc`；J2@5k为 `30487452a1a790d9e7083dc3e28e1aa7f9242e087f723add516188b2fa11b65c`。本次核查`.done`与评测完成清单，未对GB级文件重新哈希。

外部J2映射内部目标J3；外部J3映射内部目标J4（图像监督+可训练Decoder、无GT重建）。J2/J3的raw/EMA Decoder保存在对应checkpoint中；后续延长必须恢复模型、Optimizer、EMA、RNG并保留身份映射。

## 成熟模型与参考权重

| 机器 | 模型 | 现存步数 | 本轮用途 |
|---|---|---|---|
| 48 | Deep-Sym4 114M | 50k、75k、85k | **J组统一起点85k EMA**；三个checkpoint全部保留（未移动） |
| 48 | U100 Base 72M | 55k | 历史视觉参考（未移动） |
| 48 | Qwen condition A / C | 90k | 历史条件消融（未移动） |
| 4090 | P1 | 10k | 保留；S组从零训练不加载。**2026-10-27归档至`assets/_archive_20261027/u100_p1p2_20261025/p1_runs/`，永不删除** |
| 4090 | P2 | 20k | 保留；J组从Deep85开始不加载。**同上位于`_archive_20261027/u100_p1p2_20261025/p2_runs/`** |

归档明细与清理确认门见[Disk.md](Disk.md)；数据资产与重建配方见[Data.md](Data.md)。

## 数据资产与配对

- `p2_train.csv`两机SHA一致（`976b573d…`）；`generic48_kxl_top10_20261022/std_bbox64/index.csv`两机SHA一致（`c02cef37…`）——S组协议同源。
- 48的`u100_pretrain_20261024/`此前缺val100.csv、char/font remap（KB级补齐）与std_shape bank。bank按[Data.md](Data.md)配方在48**本地重建**（build_rasters逐位校验+precompute_bank，seed90），**重建结果与4090的bank SHA完全一致**（`bc3db05f…`）——零GB级传输达成同协议。曾尝试的2.8GB跨机中转已中止并登记为反面案例（见[AGENTS.md](AGENTS.md)数据规则）。std_shape_bank.pt为>1MB资产，按新口径只登记path+size。
- J组沿用旧校准文件`deep85_decoder/48_v2/calibration.json`（未改动，路径不变）。

## 排除项（沿用）

旧计划目录已按[Disk.md](Disk.md)归档：`cold50k/4090`、`deep85_decoder/48_v2`（calibration与CSV留原位）、`48/`失败目录、`style_matrix_20261026/plan_final` warm-start产物，全部位于两机`assets/_archive_20261027/`下，保留不删除。`matrix64`计划的probe探测目录（日志与train.jsonl）留作吞吐证据，临时checkpoint已清理。
