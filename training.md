# 训练进度

更新时间：2026-10-28 09:52（Asia/Shanghai），以下为双机日志与评测快照。计划见 [1027.md](1027.md)，权重见 [ckpts.md](ckpts.md)，资产和空间见 [Data.md](Data.md)、[Disk.md](Disk.md)。

| 机器 | 当前任务 | 步数 / 目标 | 状态 | 速度 | GPU |
|---|---|---|---|---|---|
| 4090 24GB | S6，Vector / DynamicConv | 26,650 / 50,000（53.3%） | 正常训练 | 约1.57 step/s | 峰值allocated 20.04 GiB |
| 48GB | S7，Multi-Token / DynamicConv | 50 / 50,000 | 编译/预热后已开始训练 | 稳态待测 | 首50步峰值allocated 19.52 GiB |

速度取正式训练窗口，累计速度仍包含少量启动、保存开销。4090此前S0停在1,350，从1k恢复后已完成50k；当时futex等待的根因未定位，不应称为已明确修复。

## 队列与预计时间

| 机器 | 已完成 | 当前与后续 | 估算 |
|---|---|---|---|
| 4090 | S0/S1/S4各50k并完成100字EMA评测 | S6（26.7k）→ S8，各50k | S6余约4.1h；S8按当前量级约8.8h，stage-1预计10-28晚间结束 |
| 48 | J0/J1/J2/J3各5k；S2/S3/S5各50k并完成100字EMA评测 | S7 50k → J0–J3分别延长至12k | S7稳态速度待500步复核，暂按约8–9h；随后J延长约9–10h |

4090在五组S完成50k后，脚本等待`selection.json`选一对继续到75k；选定后还需约9–10h。该阶段不自动挑选模型，受64h wall上限影响。48的J延长已挂入串行；两机ETA均依赖后续实测吞吐、编译、评测和磁盘状况。维持ETA：4090（含stage-2）10-29 ~10:50，48 10-29 ~08:15。

## J组Loss进展

J0–J3训练FM末千步日志均值均约0.0402，首千步约0.0525；J2/J3 Image L1从约0.0157下降到0.0106，最后两个千步窗口趋平。图像日志是最后8张microbatch的分量，反向仍累计16张。仅凭训练Loss和唯一的5k固定集评测无法确定收敛步数；继续既定四组12k对照，建议在10k补评。详细窗口与条件见[J组Loss及磁盘审计](ANALYSIS/style_repr_injection_matrix_20261026/j_loss_disk_20261027.md)。

## 已完成的评测

| 模型 | 评测 | SSIM ↑ | LPIPS ↓ | Ink-IoU ↑ | Frag ↓ |
|---|---|---:|---:|---:|---:|
| 4090 S0@50k | 100字，EMA | 0.60275 | 0.33950 | 0.28587 | 2.627 |
| 4090 S1@50k | 100字，EMA | 0.58659 | 0.34940 | 0.28062 | 2.265 |
| 4090 S4@50k | 100字，EMA | 0.59875 | 0.34055 | 0.29513 | 2.299 |
| 48 S2@50k | 100字，EMA | 0.59494 | 0.34554 | 0.28350 | 2.489 |
| 48 S3@50k | 100字，EMA | 0.59756 | 0.34258 | 0.29036 | 2.601 |
| 48 S5@50k | 100字，EMA | 0.59750 | 0.34246 | 0.28509 | 2.385 |
| 48 J0@5k | 187字，EMA | 0.61488 | 0.32198 | 0.34373 | 2.397 |
| 48 J1@5k | 187字，EMA | 0.61494 | 0.32220 | 0.34280 | 2.389 |
| 48 J2@5k | 187字，EMA＋配套Decoder EMA | 0.61977 | 0.31505 | 0.34468 | 2.236 |
| 48 J3@5k | 187字，EMA＋配套Decoder EMA | 0.61934 | 0.31474 | 0.34328 | 2.186 |

J2/J3相较J0/J1有初步SSIM和LPIPS收益，J2与J3差距很小；尚需同协议多seed、实际读图、交叉Decoder和12k评测，无法确认重建锚是否带来稳定收益。S0为不同测试集及训练起点，不能与J组直接排名。

## S组当前对照结论

S0是从零训练的`Vector + Global AdaGN`基线，也是目前唯一完成50k正式评测的S模型。S0@50k：SSIM 0.60275、LPIPS 0.33950、Ink-IoU 0.28587、Frag 2.627、SoftInk 1.0249、near-blank 0%。实际海报显示已学会清晰的书法化映射且整体墨量接近GT，但仍有局部粘连、缺/错笔、过粗墨块和结体漂移。

`Skel-IoU=0.0253`是1像素骨架的精确位置IoU，不能直接解释为“拓扑只有2.5%”。阈值0.3–0.7复算均约0.025，排除二值化阈值错误；允许1/2/3像素容差后的对称骨架F1为0.124/0.191/0.256，说明主要问题是精确中心线对位过敏。同一`validation_val100.csv`下，P1@10k/P2@20k/通用预训练FT@25k分别只有0.0156/0.0186/0.0179，S0反而更高。历史Deep-Sym4的0.039–0.045来自另一套187字CSV；0.3–0.9历史值多为显式骨架条件或VAE同图往返，均不可直接比较。当前输出的set标签仍写`eval200fix`但实际CSV是`validation_val100.csv`，属于命名/报告问题。复算证据见[S0骨架指标审计](ANALYSIS/style_repr_injection_matrix_20261026/measurements/verification/s0_skel_metric_audit.json)。

S1仅把注入从Global AdaGN改为空间AdaGN，50k时除Frag外整体低于S0；海报还显示更重的墨量与局部墨块，说明当前Spatial AdaGN并非免费收益。S4（Writer×Local CNN + Spatial AdaGN）恢复到SSIM 0.59875、LPIPS 0.34055，并把Ink-IoU提高到0.29513、Frag降到2.299，是目前新结构中最均衡的一组；但仍未全面超过S0，且存在缺/错笔和结体漂移。S3相对S2在SSIM、LPIPS、Ink-IoU上小幅改善，但Frag从2.489升至2.601，海报差异也很弱。S5与S3的SSIM/LPIPS几乎相同，Frag改善到2.385，但Ink-IoU退到0.28509；海报显示整体稍淡、局部覆盖不足，明显不如S4均衡。因此局部CNN的收益依赖注入算子，当前`Local CNN × Spatial AdaGN`优于`Local CNN × Cross-Attention`。详细对照见[阶段报告](ANALYSIS/style_repr_injection_matrix_20261026/progress_20261028_0935.md)。S6尚无50k评测；S7刚开始，S8未开始。

## 路径与运行约定

远端仓库：4090 `/root/Workspace/xy/DiT`；48 `/home/ds/Workspace/DiT`。计划均在`assets/style_matrix_20261027/matrix64/{4090,48}/`，其中`execution/status.json`、`execution/runs/<arm>/train.jsonl`、`checkpoints/`与`evaluations/`为当前事实来源。两机compile、BF16、有效batch512，禁止梯度检查点。

48旧归档、损坏P1残片及普通版QwenImage21基础模型清理后可用磁盘约167G（df快照），串行每任务启动前有25 GiB空闲空间门槛；之后checkpoint和评测仍会增长，需持续核对。4090仓库盘约1.5T可用。禁止自动删除旧checkpoint。48整盘及各工作区归属见[Disk.md](Disk.md)；详细配置、LR restart、数据配对和历史事故见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)。

## 48关键结果本地备份

截至10-28 09:30，白名单11项中8项完整落盘：metadata、DeepSym4 50k/75k/85k、U100 Base55、J0/J2/J3@5k，共13.18GiB；J1续传到153.8MB时链路中断，09:30已从`.part`恢复。随后还剩J1和Qwen A/C，完成标记尚未生成。本地G盘剩余约70.6GiB。S2@50k与J组未来12k仍是后续追加项。
