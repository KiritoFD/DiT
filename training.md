# 训练进度

更新时间：2026-10-27 01:11（Asia/Shanghai）；4090停滞仍未解决。00:53确定延长轮计划（S8、J3、成对长训、J组两阶段LR），设计与预算见[1027.md](1027.md)延长轮专节。

本文件记录当前训练事实；[1027.md](1027.md)记录当天双机计划、变量与决策；[ckpts.md](ckpts.md)记录已落盘权重及用途。历史实验归档见[exp.md](exp.md)。每次报告进度、切换任务、结束训练或发现异常时同步更新这三份文件。

## 当前状态

| 机器 | 任务 | 最新日志步数 / 目标 | 状态 | 正常计步区间速度 | 峰值allocated |
|---|---|---|---|---|---|
| 4090 24GB | S0，Vector/AdaGN，从零训练 | 1,350 / 50,000，2.7% | **停滞待排查**：进程存在，日志数分钟不动，GPU 0% | 停滞前1.614 step/s | 19.166 GiB |
| 48GB | J0，Deep85 EMA续训，FM对照 | 450 / 5,000，9.0% | 正常计步 | 1.752 step/s | 18.263 GiB |
| 48GB | J2，Deep85 Decoder联训 | 50 / 5,000，1.0% | 前50步验证完成，已保存，等待串行恢复 | 第1–50步0.510 step/s | 39.812 GiB |

速度由日志区间计算，排除首次编译；不是日志中包含编译时间的累计`step_s`。本轮S/J没有最终生成质量结论，FM loss也不能跨不同起点直接排名。

4090的`status.json`仍写`training`，但日志、进程和GPU交叉检查显示停滞，不能仅凭状态文件视为正常。主进程PID3035255及DataLoader进程仍存在，主进程与抽查worker均在futex等待；当前没有足够证据确认具体根因，未重启或修改实验。最新完整checkpoint为S0@1k，日志到1,350不代表这些更新已持久化。

## 已挂串行与ETA

| 机器 | 已实际挂起的顺序 | 剩余预算 | ETA |
|---|---|---|---|
| 4090 | S0 → S1 → S4 → S6 →（延长轮）S8 → 一组成对50k→75k | S0还剩48,650步；S1/S4/S6/S8各50k；延长对各+25k | **停滞未解决，无有效ETA**；恢复后全队列名义约62h（区间57–67h，估算） |
| 48GB | J0 → J1 → J2 →（延长轮）J3 → S2/S3/S5/S7 → J0–J3延长至12k | J0还剩4,550步；J1 5k；J2从50恢复至5k；J3 5k；S组各50k；J延长各+7k | 当前A段00:49起暂估约6–7h（约07:00前后，J1未实测）；全队列名义约61h（区间51–67h，估算） |

4090的S8与成对延长、48的J3与J延长均为00:53决策、未封存未启动；S8须实测DynamicConv吞吐，J延长须按两阶段LR restart从5k checkpoint完整恢复Optimizer/EMA/RNG。预算保护顺序与超时削减规则见[1027.md](1027.md)延长轮专节。当前没有P1/P2旧队列或旧20格矩阵在训练。

## 运行位置

| 项目 | 4090 | 48GB |
|---|---|---|
| SSH别名 | `4090` | `48` |
| 仓库绝对路径 | `/root/Workspace/xy/DiT` | `/home/ds/Workspace/DiT` |
| Python | `/opt/conda/envs/cu121/bin/python` | `/home/ds/miniconda3/envs/vcp-embryo/bin/python` |
| 计划目录，相对仓库 | `assets/style_matrix_20261027/cold50k/4090` | `assets/style_matrix_20261027/deep85_decoder/48_v2` |

每个计划目录中的`plan.json`为封存计划，`run.sh`为启动脚本，`ready.json`记录数据/配置/源码SHA。实际状态在`execution/status.json`；各实验在`execution/runs/<实验名>/`，包含`console.log`、`train.jsonl`、`checkpoints/`、`diagnostics/`与结束后的`evaluations/`。

两端启用compile、BF16、channels_last、fused AdamW、缓存latent/STD与异步checkpoint；有效batch512、FM microbatch512，禁止梯度检查点。两端各自串行，GPU锁及失败停止机制已启用。

## 最近完成的检查

- 4090 S0@1k已原子落盘；尚未生成本轮固定集完整评测。
- 48 J2@50已原子落盘，冻结Encoder/quant_conv/post_quant_conv指纹通过；该50步计入正式5k预算。
- Deep85重新校准图像损失权重0.0212917472，初始图像/FM梯度比例10%；Decoder图像及重建梯度非零。
- [编译修正记录](ANALYSIS/style_repr_injection_matrix_20261026/compile_cold50k_20261027.md)；[Decoder设定与校准](ANALYSIS/style_repr_injection_matrix_20261026/decoder_deep85_20261027.md)。

原始状态快照：本地`assets/style_matrix_20261027/status_20261027_0048/{4090,48}.json`。快照时间00:48:56–57；它记录当时状态，不代表持续在线监控。
