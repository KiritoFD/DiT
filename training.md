# 训练进度

更新时间：2026-10-27 15:11（Asia/Shanghai），以下为双机日志与评测快照。计划见 [1027.md](1027.md)，权重见 [ckpts.md](ckpts.md)，资产和空间见 [Data.md](Data.md)、[Disk.md](Disk.md)。

| 机器 | 当前任务 | 步数 / 目标 | 状态 | 速度 | GPU |
|---|---|---|---|---|---|
| 4090 24GB | S1，Vector / Spatial AdaGN | 23,200 / 50,000（46.4%） | 正常训练 | 约1.54 step/s | 总占用约22,367 MiB，利用率100%；峰值allocated 20.76 GiB |
| 48GB | S2，Vector / Cross-Attention | 36,150 / 50,000（72.3%） | 正常训练 | 约1.71 step/s | 总占用约21,223 MiB，利用率100%；峰值allocated 19.13 GiB |

速度取正式训练较长窗口，累计速度仍包含少量启动、保存开销。4090此前S0停在1,350，从1k恢复后已完成50k；当时futex等待的根因未定位，不应称为已明确修复。

## 队列与预计时间

| 机器 | 已完成 | 当前与后续 | 估算 |
|---|---|---|---|
| 4090 | S0@50k，100字EMA评测 | S1 → S4 → S6 → S8，各50k | S1约4.8–5h；四组余下50k训练及评测合计约32–34h |
| 48 | J0/J1/J2/J3各5k及187字EMA评测 | S2 → S3 → S5 → S7各50k → J0–J3分别延长至12k | S2约2.2–2.4h；完整后续约37–40h |

4090在五组S完成50k后，脚本等待`selection.json`选一对继续到75k；选定后还需约9–10h。该阶段不自动挑选模型，受64h wall上限影响。48的J延长已挂入串行；两机ETA均依赖后续实测吞吐、编译、评测和磁盘状况。

## 已完成的评测

| 模型 | 评测 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ |
|---|---|---:|---:|---:|---:|
| 4090 S0@50k | 100字，EMA | 0.60275 | 0.33950 | 0.28587 | 2.627 |
| 48 J0@5k | 187字，EMA | 0.61488 | 0.32198 | 0.34373 | 2.397 |
| 48 J1@5k | 187字，EMA | 0.61494 | 0.32220 | 0.34280 | 2.389 |
| 48 J2@5k | 187字，EMA＋配套Decoder EMA | 0.61977 | 0.31505 | 0.34468 | 2.236 |
| 48 J3@5k | 187字，EMA＋配套Decoder EMA | 0.61934 | 0.31474 | 0.34328 | 2.186 |

J2/J3相较J0/J1有初步SSIM和LPIPS收益，J2与J3差距很小；尚需同协议多seed、实际读图、交叉Decoder和12k评测，无法确认重建锚是否带来稳定收益。S0为不同测试集及训练起点，不能与J组直接排名。

## 路径与运行约定

远端仓库：4090 `/root/Workspace/xy/DiT`；48 `/home/ds/Workspace/DiT`。计划均在`assets/style_matrix_20261027/matrix64/{4090,48}/`，其中`execution/status.json`、`execution/runs/<arm>/train.jsonl`、`checkpoints/`与`evaluations/`为当前事实来源。两机compile、BF16、有效batch512，禁止梯度检查点。

48可用磁盘约101G（df快照），串行每任务启动前有25 GiB空闲空间门槛；之后checkpoint和评测仍会增长，需持续核对。4090约1.5T可用。禁止自动删除旧checkpoint。详细配置、LR restart、数据配对和历史事故见 [延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)。
