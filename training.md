# 训练进度

更新时间：2026-10-27 16:05（Asia/Shanghai），以下为双机日志与评测快照。计划见 [1027.md](1027.md)，权重见 [ckpts.md](ckpts.md)，资产和空间见 [Data.md](Data.md)、[Disk.md](Disk.md)。

| 机器 | 当前任务 | 步数 / 目标 | 状态 | 速度 | GPU |
|---|---|---|---|---|---|
| 4090 24GB | S1，Vector / Spatial AdaGN | 26,200 / 50,000（52.4%） | 正常训练 | 约1.54 step/s | 总占用约22,367 MiB，利用率100%；峰值allocated 20.76 GiB |
| 48GB | S2，Vector / Cross-Attention | 39,600 / 50,000（79.2%） | 正常训练 | 约1.71 step/s | 总占用约21,223 MiB，利用率100%；峰值allocated 19.13 GiB |

速度取正式训练较长窗口，累计速度仍包含少量启动、保存开销。4090此前S0停在1,350，从1k恢复后已完成50k；当时futex等待的根因未定位，不应称为已明确修复。

## 队列与预计时间

| 机器 | 已完成 | 当前与后续 | 估算 |
|---|---|---|---|
| 4090 | S0@50k，100字EMA评测 | S1 → S4 → S6 → S8，各50k | S1约4.4h；四组余下50k训练及评测合计约31–33h |
| 48 | J0/J1/J2/J3各5k及187字EMA评测 | S2 → S3 → S5 → S7各50k → J0–J3分别延长至12k | S2约1.8h；完整后续约36–39h |

4090在五组S完成50k后，脚本等待`selection.json`选一对继续到75k；选定后还需约9–10h。该阶段不自动挑选模型，受64h wall上限影响。48的J延长已挂入串行；两机ETA均依赖后续实测吞吐、编译、评测和磁盘状况。

## J组Loss进展

J0–J3训练FM末千步日志均值均约0.0402，首千步约0.0525；J2/J3 Image L1从约0.0157下降到0.0106，最后两个千步窗口趋平。图像日志是最后8张microbatch的分量，反向仍累计16张。仅凭训练Loss和唯一的5k固定集评测无法确定收敛步数；继续既定四组12k对照，建议在10k补评。详细窗口与条件见[J组Loss及磁盘审计](ANALYSIS/style_repr_injection_matrix_20261026/j_loss_disk_20261027.md)。

## 已完成的评测

| 模型 | 评测 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ |
|---|---|---:|---:|---:|---:|
| 4090 S0@50k | 100字，EMA | 0.60275 | 0.33950 | 0.28587 | 2.627 |
| 48 J0@5k | 187字，EMA | 0.61488 | 0.32198 | 0.34373 | 2.397 |
| 48 J1@5k | 187字，EMA | 0.61494 | 0.32220 | 0.34280 | 2.389 |
| 48 J2@5k | 187字，EMA＋配套Decoder EMA | 0.61977 | 0.31505 | 0.34468 | 2.236 |
| 48 J3@5k | 187字，EMA＋配套Decoder EMA | 0.61934 | 0.31474 | 0.34328 | 2.186 |

J2/J3相较J0/J1有初步SSIM和LPIPS收益，J2与J3差距很小；尚需同协议多seed、实际读图、交叉Decoder和12k评测，无法确认重建锚是否带来稳定收益。S0为不同测试集及训练起点，不能与J组直接排名。

## S组当前对照结论

S0是从零训练的`Vector + Global AdaGN`基线，也是目前唯一完成50k正式评测的S模型。S0@50k：SSIM 0.60275、LPIPS 0.33950、Ink-IoU 0.28587、Frag 2.627、SoftInk 1.0249、near-blank 0%。实际海报显示已学会清晰的书法化映射且整体墨量接近GT，但仍有局部粘连、缺/错笔、过粗墨块和结体漂移。

`Skel-IoU=0.0253`是1像素骨架的精确位置IoU，不能直接解释为“拓扑只有2.5%”。阈值0.3–0.7复算均约0.025，排除二值化阈值错误；允许1/2/3像素容差后的对称骨架F1为0.124/0.191/0.256，说明主要问题是精确中心线对位过敏。同一`validation_val100.csv`下，P1@10k/P2@20k/通用预训练FT@25k分别只有0.0156/0.0186/0.0179，S0反而更高。历史Deep-Sym4的0.039–0.045来自另一套187字CSV；0.3–0.9历史值多为显式骨架条件或VAE同图往返，均不可直接比较。当前输出的set标签仍写`eval200fix`但实际CSV是`validation_val100.csv`，属于命名/报告问题。复算证据见[S0骨架指标审计](ANALYSIS/style_repr_injection_matrix_20261026/measurements/verification/s0_skel_metric_audit.json)。

S1仅把注入从Global AdaGN改为空间AdaGN，当前26.2k尚无正式生成评测。同期训练FM与S0几乎重合但略高：S0在20–21k/25–26k窗口约0.11/0.09，S1约0.12/0.10。这只说明空间注入没有带来明显优化速度优势，不能据此判定生成质量或Writer控制更差。必须等S1@50k同协议评测后，才可得出S1−S0的单变量结论；S4/S6/S8尚未开始，更不能提前比较局部表征和DynamicConv。

## 路径与运行约定

远端仓库：4090 `/root/Workspace/xy/DiT`；48 `/home/ds/Workspace/DiT`。计划均在`assets/style_matrix_20261027/matrix64/{4090,48}/`，其中`execution/status.json`、`execution/runs/<arm>/train.jsonl`、`checkpoints/`与`evaluations/`为当前事实来源。两机compile、BF16、有效batch512，禁止梯度检查点。

48旧归档、损坏P1残片及普通版QwenImage21基础模型清理后可用磁盘约167G（df快照），串行每任务启动前有25 GiB空闲空间门槛；之后checkpoint和评测仍会增长，需持续核对。4090仓库盘约1.5T可用。禁止自动删除旧checkpoint。48整盘及各工作区归属见[Disk.md](Disk.md)；详细配置、LR restart、数据配对和历史事故见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)。

## 48关键结果本地备份

2026-10-27 16:01已启动48→本地后台串行下载，不占GPU。清单为[manifest.json](ANALYSIS/remote48_backup_20261027/manifest.json)，目标目录`G:\DiT-48-artifacts-20261027`，总量18.65GiB。实测短窗口约0.65MiB/s，对应约8.2h；这是网络瞬时估算，不是完成承诺。下载器支持`.part`断点续传、完成字节数校验和单实例锁。S2只在50k完成后补抓，J组12k完成后另补最终权重。
