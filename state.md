# state.md — 当前实验状态

运行状态复核：2026-10-18 19:09 CST（Asia/Shanghai）；三臂历史结果沿用15:25登记，CFG审计见下。
全部实验、详细设定、路径、历史对比与图表统一在 [exp.md](exp.md)。
更早前原始状态请查阅 [历史state归档](docs/experiments/ledger/state_history_20261018.md)。

**19:03只读审计补充**：Writer CFG原始文件与用户给出的DiC数值相符；但采样脚本Base/Gate走raw、DiC专用分支走EMA，协议却均标ema=true，条件哈希为空。旧/新CFG1.0基线差异尚未消除，不把这些跨模型行当作统一EMA横评。详见[实验总账§9](exp.md#9-writer-cfg审计2026-10-18-1903只读)及[原始数值快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。本轮未启停训练、未重新采样、未排Style Loss。

## 1. 当前运行任务表

| 机器 | 当前执行 | 当前训练日志 | 最新评测指标 | 队列后项 |
| --- | --- | --- | --- | --- |
| 48（4090，48GB） | DiC三臂完成状态沿用15:25记录 | 19:08核验无GPU compute进程，旧supervisor3540850已退出 | 65k历史记录：Arm3 SSIM0.6061、Arm2 Frag2.368/LPIPS0.3286；非同预算架构胜负结论 | 无新训练排队；本轮仅审计已有CFG产物 |
| 4090（24GB） | **U100-DiC-768**，75.83M CNN U-Net | **76,600 / 100,000**，约1.71step/s，日志显存22.24GiB，Diff0.0767；PID1738543存活 | 原50k指标仍为历史记录；本轮新增同50k CFG扫描核验见exp.md§9 | 当前100k训练继续，不改调度 |

凌晨历史配置/日志：[03:47快照](docs/experiments/ledger/live_snapshot_20261018.json)；晚间CFG及进程核验：[19:03–19:09快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。两份快照不能互充实时状态。
本地只拉取文档和关键小图像，不滥停训练，非指示不修改任何训练代码。

## 2. 48卡：DiC 三臂对照串行队列（50k Checkpoint 热启，已完赛）

| 核心项 | 登记 |
| --- | --- |
| 测试的是什么 | DiC（CVPR 2025）核心改进独立消融：1) `DiC_residual_gate`（75.04M）；2) `DiC_stage_cond_adapter`（72.88M）；3) `DiC_stage_cond_independent`（72.80M）。 |
| 结果总结 | 65k记录SSIM0.6054–0.6061、LPIPS最低0.3286、Frag最低2.368。与55k Base训练量不同，且无匹配续训Control，不能把差值全归结构或宣称显著优胜。 |
| 调度状态 | Supervisor PID 3540850 记录 `all_completed`，完成时间 14:32:41。完整 summary CSV 已落盘并归档。 |

## 3. 4090卡：U100-DiC-768（从头训练，全功能 DiC 架构 U-Net）

| 核心项 | 登记 |
| --- | --- |
| 测试的是什么 | 全功能 DiC 架构组合：全开 `residual_cond_gate` + `stage_specific_cond`（adapter 模式）+ `act_fn='gelu'`。 |
| 规模与吞吐 | Batch=768，75,831,524 参数，步速 ~1.71 step/s（1,313 samples/s），显存占用 22.24 GiB / 24.56 GiB。 |
| 50k历史记录 | B768×50k=38.4M曝光，Base B512×55k=28.16M；SSIM0.6047/IoU0.3234/Frag2.301不构成等曝光单变量收益。Frag不是越低越好。 |
| 当前进程 | 19:09核验PID1738543存活，日志76.6k，Diff0.0767，目标100k。 |

## 4. 已落盘核心资产

- **4090从头训练50k产物**：`4090:/root/Workspace/xy/DiT/assets/results/unet_dic_100k_b768/20261018-063716-Unet-flow/checkpoints/0050000.pt`（及 5k~50k 完整权重队列）。
- **4090 50k评测海报与指标**：`assets/results/eval_4090_b768_50k/posters/eval200fix_poster.png`，`eval_stdskel_summary.csv`。
- **48卡三臂完赛 Checkpoints 与评测**：`assets/results/unet_dic/` 下三臂 55k~75k 完整 checkpoints 与 summary CSV。
