# 磁盘台账

2026-10-29 20:28：仅新增本地明文报告及文档包，没有远端清理或模型操作。SSH入口关闭，远端容量仍引用19:22已核验的4090约1522G/48约81G，不声称为此刻实时空间。

2026-10-29 19:22现场：4090工作盘剩约1522G（df -BG），48根盘剩81G；未做清理。HWD输出两端约12M/25M，4090探索审美输出约2.2M；各机官方HWD权重约675MiB，独立下载，无跨机传输。S0/S5各新增55k约1.8GB checkpoint，后续仍每5k保留；48现有空间足以继续原S5队列，25GiB启动门槛不取消。原始结果仅本地无损压缩归档，不删除远端权重或数据。

10-29本轮64h队列仅做只读状态核查，无磁盘清理、资产迁移或新增大文件操作。

2026-10-29 09:17：本地报告zip为4,067,393 bytes（Git排除，校验信息单列）；两机未来源码暂存包各17,408 bytes。无远端权重清理/迁移，当前训练未因归档或后续代码部署而停止。

2026-10-29阶段报告归档仅增加本地小型原始指标副本及报告/海报zip；exports中的zip不纳入Git。远端数据与权重未移动或清理，活动队列不受影响。

2026-10-28 23:20：双机24h队列正常运行，继续执行每任务25GiB可用空间检查；维护存盘S9@403、J4@1,675保留，未删除或搬移权重。后续分叉通过本机只读软链接复用S9@50k，不复制GB级checkpoint。

2026-10-28 22:28：第二代验证新增J4/J5五步smoke checkpoint与J5配套生成评测，各自在新计划的`smoke/`下；均与正式run隔离。不清理或覆盖旧权重。旧J0停机新增11,654步完整checkpoint。新队列继续使用25GiB启动空闲门槛；本轮无数据迁移或删除。

最后更新：2026-10-27 15:45（Asia/Shanghai）。本页以 48 机器的实时 `df` / `du -x` 为准。`du` 的父子目录不可相加，表内 GiB/TiB 为文件系统显示的近似值；`df` 是整盘容量的权威口径。

## 48 整盘结论

| 项目 | 当前值 | 说明 |
|---|---:|---|
| 持久盘 `/dev/nvme0n1p2` | 1.8T 总量，1.6T 已用，167G 可用，91% | `/`、`/home`、仓库和数据均在同一 ext4 分区 |
| `/home/ds` | 约 1.5T | 用户可见项目、数据、环境和缓存的主体 |
| 系统目录 | `/usr` 49G，`/var` 7.1G | 当前用户权限下的近似盘点，不作为用户目录清理范围 |
| `/tmp`、`/dev/shm` | tmpfs | 属于内存文件系统，不能计入 136G 持久盘余量 |

2026-10-27 15:27 已按用户确认删除 48 的旧归档和损坏的 P1 残片，整盘可用空间由约 101G 增至 136G，净释放 **37,438,140,416 bytes（34.87 GiB）**。详细清单见 [cleanup_48_20261027.json](ANALYSIS/style_repr_injection_matrix_20261026/measurements/plan/cleanup_48_20261027.json)。其中旧 `results_nonkey` 约 22G 可能包含唯一的历史非关键结果，现已删除；当前矩阵中的 J checkpoint 副本、关键基线、训练数据和正在运行的任务均保留。

2026-10-27 15:44 又删除了明确不用的普通版 QwenImage21 基础模型 `/home/ds/Workspace/xy/qwen-lora-20261023/base`，释放 **33,131,773,952 bytes（30.85 GiB）**，可用空间升至167G。Turbo版 `/home/ds/Workspace/xy/qwen-calli-20261022/model` 完整保留。删除证据见 [cleanup_qwen_image_nonturbo_20261027.json](ANALYSIS/style_repr_injection_matrix_20261026/measurements/plan/cleanup_qwen_image_nonturbo_20261027.json)。

## 工作区归属

### 我们的 DiT 工作区

`/home/ds/Workspace/DiT` 合计约 **99G**，约占 `/home/ds/Workspace` 的 58%，约占整盘已用空间的 6%。

| DiT 子目录 | 大小 | 状态 / 用途 |
|---|---:|---|
| `assets/` | 64G | 当前实验资产与关键基线；运行中，不能整体清理 |
| `assets/style_matrix_20261027/matrix64/` | 43G | 当前 S/J 矩阵、checkpoint 和评测，**保护** |
| `assets/results/` | 15G | Deep-Sym4、U100、Qwen A/C 等关键模型，**保护** |
| `data/` | 20G | 训练/评测数据，需按 [Data.md](Data.md) 的来源和重建规则逐项审计 |
| 根 `results/` | 8.3G | 历史结果；尚未逐项确认引用，不能批量删除 |
| `exp-std/` | 3.8G | STD 实验资产；需核对当前配置引用 |
| `pretrained_models/` | 1.1G | 预训练权重，保留 |
| `experiments/` | 1.0G | 实验定义/产物，需逐项审计 |

当前必须保留：`matrix64/`、所有 `ready.json` 封存路径、原始 VAE、GT/STD 数据、Deep-Sym4 50k/75k/85k、U100 Base55k、Qwen A/C 90k，以及评测依赖。跨机缺失资产优先本机重建，不通过 GB 级传输补齐。

### 同一 Workspace 下的其他工作区

`/home/ds/Workspace` 清理后合计约 **139–140G**：

| 工作区 | 大小 | 主要内容 | 清理判断 |
|---|---:|---|---|
| `Workspace/DiT` | 99G | 本项目代码、数据、当前矩阵和关键结果 | 当前只允许逐项审计，不整体清理 |
| `Workspace/xy` | 41G | Qwen 书法/LoRA 系列实验 | 普通版基础模型已删，Turbo版保留 |

`Workspace/xy` 清理后的主要占用为：

- `qwen-calli-20261022/model`：31G，Turbo版基础模型，保留；
- `qwen-lora-turbo-20261023`：4.3G；
- `qwen-lora-base-20261024`：1.4G，普通版LoRA，基础模型链接现已失效；
- `qwen-lora-top10-base-20261024`：982M，普通版Top10 LoRA，基础模型链接现已失效；
- 其余 Turbo/Top10/reference/writer-res 等目录：约3–4G。

两个普通版LoRA目录的小权重仍被保留，避免把本次“删除非Turbo基础模型”扩大成删除实验历史；若确认这些普通版实验也无价值，可再释放约2.4G。它们当前的`base`软链接已失效，不能直接推理。

## 其他项目占用

| 项目/目录 | 总量 | 主要子目录 | 当前判断 |
|---|---:|---|---|
| `/home/ds/ERV_mouse` | 1.2T | `results/G_v9_multitissue` 687G；`work/preG` 280G；`work/preG_v3` 90G；`work/phaseD` 30G；`data/raw` 20G | 整盘最大来源；存在活动 Python 进程，未经该项目依赖审计不得删除 |
| `/home/ds/realsr` | 69G | `data/decoded` 32G；`data/pretrain` 15G；`experiments/diffusion` 15G | 其他项目；生成数据/实验结果可能可重建，但需项目负责人确认 |
| `/home/ds/VCP` | 57G | `external_data` 32G；`research` 23G | 其他项目；原始外部数据和研究结果，不自动清理 |
| `/home/ds/miniconda3` | 约 37G | `envs` 25G；`pkgs` 5G；base 库约 7.4G | `vcp-embryo` 8.3G 正被 48 训练使用；其他环境需按项目核对 |
| `/home/ds/.cache` | 23G | Hugging Face 17G；Torch 5.5G | 共享缓存；可重建但重下成本高，当前训练/评测可能引用 |
| `/home/ds/.vscode-server` | 3.8G | VS Code Server | 有活动服务，只能清理确认不用的旧版本 |
| `/home/ds/models` | 2.2G | 未完成来源审计 | 暂保留 |

注意：`ERV_mouse/results` 703G 和 `ERV_mouse/work` 417G 已包含在 `ERV_mouse` 的 1.2T 内，不能重复相加。

## 后续可清理范围

以下只是候选，**本轮未删除**：

| 优先级 | 候选 | 可能释放 | 前置条件 |
|---|---|---:|---|
| P0 | `ERV_mouse/work/preG` | 280G | 确认是可重建中间产物、无活动进程打开、最终结果已验收 |
| P0 | `ERV_mouse/work/preG_v3` | 90G | 同上；需与当前 ERV 进程及下游结果逐项核对 |
| P1 | `Workspace/xy` 两个普通版LoRA实验目录 | 约2.4G | 基础模型已删除；确认不保留checkpoint和评测历史后可删 |
| P1 | `realsr/data/decoded` | 32G | 有确定性重建路径且原始输入完整 |
| P1 | `realsr/experiments/diffusion` | 15G | 项目负责人确认不再需要 checkpoint/指标/图像 |
| P2 | Hugging Face Qwen 缓存 | 约 16G | 确认各 Qwen 项目不再离线依赖；删除后需要重新下载 |
| P2 | 未使用 Conda 环境 | 单环境约 8G | 先按环境名核对项目；`vcp-embryo` 明确保护 |

不建议把 `/home/ds/miniconda3/pkgs` 的 5G 直接视为可释放空间：`conda clean --dry-run --packages/--tarballs` 的当前结果均为 **0 bytes** 可清，说明这些包仍可能被环境硬链接引用。Torch inductor、PyIQA/LPIPS checkpoint 也服务于当前训练和统一评测，不列入自动清理。

## 清理与保护规则

1. 当前队列每个任务启动前检查至少 25GiB 持久盘余量；167G 目前足够，但 checkpoint 会继续增长。
2. 删除前必须确认目标不在 `ready.json`、[ckpts.md](ckpts.md) 关键权重表和活动进程打开文件中。
3. 大数据优先本机重建；不为清盘把 GB 资产跨机搬运。
4. 其他项目只提供候选范围，不因目录名含 `work`、`decoded`、`cache` 就直接删除。

## 4090 磁盘（2026-10-28 00:15 补充）

| 项目 | 当前值 | 说明 |
|---|---:|---|
| **仓库所在盘 `/root/Workspace`（nvme1n1p1）** | 3.5T 总量，**1.6T 可用（53%）** | 仓库在独立NVMe上，空间充裕 |
| 容器根 `df /`（overlay 1.8T） | 230G 可用 | **系统盘，与仓库无关**——此前误把 254G/230G 当仓库余量 |

2026-10-28 00:15 经确认执行清理：archive 中非关键项已删除（runs_legacy 49G、20261026 矩阵时代 14G、three_day/generic114/style_24h 等共约 119G）；archive 仅保留 `u100_p1p2_20261025`（P1@10k、P2@20k，永不删除）。仓库 `assets/` 自占用 190G→114G（含当日新增 checkpoint 约 45G）。清理后训练正常（S4 计步中）。

## 48 大幅 ckpt 清理（2026-10-30 23:10，用户指令：只留必须的基模）

保留：sd-vae-ft-ema、E-G判别器、std_shape_bank/tables、**Deep85@85k**（50k/75k删除）、**U100 Base 55k**、**S5@85k终点**（中间5k–80k全删）、各已评测实验终点之外的PNG/JSON。删除：matrix64 J0–J3 与 S2/S3/S5/S7 全部checkpoint（J0–J3有本地备份）、generation2 J4–J7/S11 全部checkpoint（**无备份，恢复需重训**）、Qwen A/C（有本地备份）。预计释放 ~130G；**执行因48入口断连而完成度未核验，恢复后第一件事是核对**。

## 变更记录

- 2026-10-28 21:30：新增`docs/1028`结果包约3.9MB（两张poster约3.8MB，其余Markdown/JSON小于0.1MB）及S7本地评测镜像约1.9MB；未迁移checkpoint/latent/数据集，未执行清理。48恢复J延长后仍受每任务25GiB空闲门槛保护。
- 2026-10-28 16:30：为S/J全面评测拉取已有poster及小型逐样本结果；本地新增`full_eval_20261028`约6.29MB、原始poster镜像约25.24MB。无远端checkpoint/数据迁移，无清理操作；两机训练盘状态未改变。
- 2026-10-28 15:55：48关键checkpoint本地备份已于12:42完成，11/11项共18.65GiB；本地G盘余约65.2GiB。48训练盘余约144G，仍高于25GiB任务启动门槛。
- 2026-10-28 00:15：4090 执行 archive 非关键项删除（约119G，P1/P2保留）；确认仓库实际位于独立3.5T盘（1.6T可用），此前df误读为容器overlay系统盘。48侧复核：16:05清理后全部保护项（calibration/CSV/matrix64/Deep85 85k/bank）原位完好。
- 2026-10-27 16:05：启动48关键checkpoint向本地`G:\DiT-48-artifacts-20261027`的白名单备份，总量18.65GiB；本地启动前可用83.9GiB。48仅新增69MB元数据压缩包`assets/transfer_staging/remote48_results_metadata_20261027.tar.gz`，训练权重未移动、未删除。（23:52记录：下载器曾因ssh无keepalive停滞5.5h，已恢复，详见ANALYSIS/remote48_backup_20261027/README.md）
- 2026-10-27 15:45：删除普通版QwenImage21基础模型，释放30.85GiB；Turbo模型及当前DiT训练完整保留。
- 2026-10-27 15:31：完成 48 全盘刷新；确认 DiT 99G、`Workspace/xy` 72G、ERV 1.2T、RealSR 69G、VCP 57G、Conda 约 37G、共享缓存 23G。
- 2026-10-27 15:27：经用户确认删除 48 旧 `_archive_20261027` 与损坏 P1 残片，净释放 34.87GiB；当前训练未中断。
- 2026-10-27 03:40：旧实验首次同盘归档；归档本身不释放空间。
## 2026-10-29 15:46 暂停扩展并汇总

按用户要求暂停H0/H1实现与启动；双机controller和later观察器SIGSTOP，当前S11训练、S9评测子任务保留，之后不再自动接下一实验。4090 S9已完成50k：val100 SSIM0.59814/LPIPS0.34543/Ink-IoU0.27672，未超过S0；48 J4–J7各5k高度重合，Decoder晋级为空，S11约44.3k/50k，纯训练剩约1.2h。共享训练面板方向cosine约0.886不是留出风格泛化证据；187字差分n_cells=0。完整数据、实际读图、边界及暂停PID见[10-29汇总](docs/1029/RESULTS_AND_PLAN_REVIEW.md)。旧自动队列ETA暂不适用，恢复前重新定计划。

## 2026-10-29 新评测空间快照

48仓库盘约86GB可用、使用96%；4090仓库盘约1.5TB可用。新结果目录`assets/eval_unpaired_20261029/`约4090 63MB、48 148MB（末段评测会有少量增长）。各机IQA缓存共约380MB，其中48既有KONIQ约104MB复用；另有约92MB官方Inception临时权重和独立评测venv。

本轮未清理、移动或删除checkpoint。统计/特征缓存可按[Data.md](Data.md)重建；校准与原始统计已入[新评价专题](docs/1029/eval_unpaired/REPORT.md)。任何后续清理仍单独确认，不能据此删除历史权重。
## 2026-10-29 书写领域指标首轮

已完成缓存特征上的Style-FID/KID、Writer Verification池化AUC/EER、书家宏平均检索mAP，以及k=5 Density/Coverage；两端complete均存在。按用户要求先交付已算指标，暂停扩展实现。官方HWD、CHAED/TabNet尚无分数，不混入结果表。[已完成指标报告](docs/1029/domain_metrics/REPORT.md)，原始统计在同目录及远端`assets/eval_domain_20261029/`。S5的E-G身份核验点估计最好；S7池化Style-FID更低但身份核验不是最好；J2分布有小幅改善，J4–J7仍接近。未进行新生成器训练、未清理权重，无GB级跨机传输。
## 2026-10-29 focused85k保存预算

4090原有约1.5TB可用，预计S0/S4各7个新增checkpoint共约26GB；48当前约83GB可用，S5新增7份约13GB，预计仍高于25GiB串行保护门槛。不删除历史权重。新50k起点是本机软链接，不重复复制1.8GB文件；48两次启动前失败目录仅保留小型计划/日志与链接。[有效目录](docs/1029/style_focus_85k/README.md)。
