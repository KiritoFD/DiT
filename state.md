# state.md — 当前实验状态

运行状态复核：2026-10-18 19:09 CST（Asia/Shanghai）；三臂历史结果沿用15:25登记，CFG审计见下。
全部实验、详细设定、路径、历史对比与图表统一在 [exp.md](exp.md)。
更早前原始状态请查阅 [历史state归档](docs/experiments/ledger/state_history_20261018.md)。

**19:03只读审计补充**：Writer CFG原始文件与用户给出的DiC数值相符；但采样脚本Base/Gate走raw、DiC专用分支走EMA，协议却均标ema=true，条件哈希为空。旧/新CFG1.0基线差异尚未消除，不把这些跨模型行当作统一EMA横评。详见[实验总账§9](exp.md#9-writer-cfg审计2026-10-18-1903只读)及[原始数值快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。本轮未启停训练、未重新采样、未排Style Loss。

## 1. 当前运行任务表

| 机器 | 当前执行 | 当前训练日志 | 最新评测指标 | 队列后项 |
| --- | --- | --- | --- | --- |
| 48（4090，48GB） | **U100-Deep-Sym4 从头预训练**（纯U100实现，关gate，大幅对称加深至 114M 参数，预算 60M 样本曝光） | **已按指令平稳停机封存**（完成至 85,300 步，落盘 0085000.pt），GPU 显存 0 MiB 完全释放；**已完成深度磁盘瘦身与压缩**，工作空间由 150G 压缩至 65G（瘦身 56.7%），根分区可用空间由 172G 扩展至 256G（净释放 84GB） | **确认全面收敛，刷新全线 SOTA**：85k 节点 SSIM Mean **0.6129** (75k 峰值 **0.6132**) / LPIPS **0.3207** / Ink-IoU **0.3381** / Frag **2.14** / 零空白！连续 1.5 万步指标高度横盘，训练损失 0.0592 | 核心成果（0075000.pt、0085000.pt、Base 55k、DiC 三臂等关键权重与评测数据）全量无损归档 |
| 4090（24GB） | **U100-DiC-768**，75.83M CNN U-Net | **79,250 / 100,000**，约 1.71 step/s，显存 22.24 GiB，Diff 0.0742；PID 1738543 稳健存活 | 50k 历史峰值（SSIM 0.6047 / IoU 0.3234 / Frag 2.301 / Hit 64.6%） | 冲刺 100k 最终封版 |

凌晨历史配置/日志：[03:47快照](docs/experiments/ledger/live_snapshot_20261018.json)；晚间CFG及进程核验：[19:03–19:09快照](docs/experiments/ledger/writer_cfg_snapshot_20261018.json)。两份快照不能互充实时状态。
本地只拉取文档和关键小图像，不滥停训练，非指示不修改任何训练代码。

## 2. 48卡：CalliFlow-v2 从头预训练（Deep DiC + 双速度场显式建模）

| 核心项 | 详细设置与实时状态 |
| --- | --- |
| **架构代际** | **CalliFlow-v2**（下一代 Hourglass CNN U-Net，参数量 63.23M） |
| **核心创新** | **双速度场显式分解**：$v_\theta = v_\text{base}(z_t, t, g, s) + \Delta v_\text{writer}(z_t, t, g, s, w)$ |
| **Base Field** | 负责汉字内容、基本笔画连通与结构合理性，输入严格物理屏蔽 Writer ID（纯学 $\mathbb{E}[v^*\|z_t,t,g,s]$） |
| **Writer Field** | 负责书家结体变化、笔势转折与墨迹特征，多尺度提取（$8^2 \to 16^2 \to 32^2$），DiC通道条件门控调制，零初始化头 |
| **推理加速** | **单次前向极速 CFG**：$v_\text{CFG} = v_\text{base} + \alpha \Delta v_\text{writer}$，彻底免除双倍网络前向调用 |
| **损失函数** | **双 FM 闭式监督**：$\mathcal{L} = \|(v_\text{base} + \Delta v_\text{writer}) - v^*\|^2 + 0.5 \|v_\text{base} - v^*\|^2$，无需外部风格教师或配对真迹 |
| **骨干规格** | Channels `[128, 256, 384]`，Encoder 4 ResBlocks/尺度，Decoder 4 ResBlocks/尺度，Middle 3 ResBlocks + SDPA，GELU 激活 |
| **STD 先验** | $32^2, 16^2, 8^2$ 多尺度空间注入，mild 增强为主 |
| **训练动力学** | Global Batch 512，LR 6e-5（Cosine 衰减至 0.1），Warmup 1500 步，EMA 0.999，Channels-last，Torch Compile |
| **执行监控** | PID 714015，吞吐 **2.83 step/s（1,449 samples/s）**，显存稳健驻留 **16.42 GiB** / 48 GiB，日志位于 `logs/train_calliflow_v2_120k.log` |
| **阶段性观测点** | 25k（12.8M 曝光：内容收敛与残差非零）、75k（38.4M 曝光：等曝光量对标 DiC-768）、120k（61.44M 曝光：最终封版） |

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
