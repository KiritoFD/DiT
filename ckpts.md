# Checkpoint登记

更新时间：2026-10-27 02:45（Asia/Shanghai）。延长轮已实现并重新拉起：实施记录见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)，实时状态见[training.md](training.md)，完整历史索引见[exp.md](exp.md)。

## 路径与检查口径

下文路径均相对对应机器仓库：4090为`/root/Workspace/xy/DiT`，48为`/home/ds/Workspace/DiT`。新计划根目录为`assets/style_matrix_20261027/matrix64/{4090,48}`。

封存口径变更（2026-10-27）：>1MB大资产在ready.json只记path+size不哈希；小文件（spec/代码/remap/CSV头/calibration）保留SHA256并在每次启动及每任务前校验。旧计划`cold50k/4090`、`deep85_decoder/48_v2`的ready为全量哈希口径，均已停止、保留存档。

## 当前实验已保存（matrix64计划）

| 机器 | 实验 / 步数 | 来源 | 用途 |
|---|---|---|---|
| 4090 | S0@1,000 | 从旧cold50k计划原样迁移（含`.done`清单） | 当前串行从1k续训，已越过旧停滞点 |
| 48 | J0@5,000 + heldout187评测 | 从旧48_v2计划原样迁移 | stage1首个任务，到点直接进入评测 |
| 48 | J1@575 | 旧48_v2计划J1优雅中断点 | stage1第二任务从575续训至5k |
| 48 | J2@50 | 旧48_v2计划预检点 | stage1第三任务从50续训至5k |

迁移由`matrix64.py`完成并登记在各计划目录`migrations.json`；旧spec字节级复用，identity严格匹配，续训断言可通过。

身份映射登记：外部J3对应**内部目标臂`J4`**（图像监督+可训练Decoder、无GT重建分支），见`matrix_identity.internal_objective`；外部J2仍对应内部J3。延长spec（`*_ext75.json`/`*_ext12.json`）携带`continues_identity`指向对应5k/50k spec的身份，其自身identity仅用于延长段新checkpoint。

关键权重（未变更）：

```text
# 48：Deep85 85k EMA，J组统一起点，1,824,546,895 bytes
assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt
SHA256 f80304d7de17f334ebf57dfda312988dbd5e994a18f68e6d36f8cb30d5696349
# 4090：S0@1k（迁移前原文件）
assets/style_matrix_20261027/cold50k/4090/execution/runs/S0/checkpoints/0001000.pt
SHA256 e93dc2142629749fb682c0137c5c4ed3e6b71c5bf4ba966c868119b62164281a
```

## 延长轮权重要求

- J0–J3的5k checkpoint必须完整携带Optimizer/EMA/RNG/scheduler状态（现有`build_checkpoint`口径已满足），供5k→12k两阶段restart恢复；延长spec经`continues_identity`校验身份。
- S组50k checkpoint同样作为75k延长起点，raw/EMA不得混用。
- S8从零随机初始化（seed90）；J3从Deep85 85k EMA与原始VAE独立初始化，禁止从J2任何checkpoint热启动。

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
