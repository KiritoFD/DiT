# 磁盘台账

最后更新：2026-10-27 15:31（Asia/Shanghai）。本页以 48 机器的实时 `df` / `du -x` 为准。`du` 的父子目录不可相加，表内 GiB/TiB 为文件系统显示的近似值；`df` 是整盘容量的权威口径。

## 48 整盘结论

| 项目 | 当前值 | 说明 |
|---|---:|---|
| 持久盘 `/dev/nvme0n1p2` | 1.8T 总量，1.6T 已用，136G 可用，93% | `/`、`/home`、仓库和数据均在同一 ext4 分区 |
| `/home/ds` | 约 1.5T | 用户可见项目、数据、环境和缓存的主体 |
| 系统目录 | `/usr` 49G，`/var` 7.1G | 当前用户权限下的近似盘点，不作为用户目录清理范围 |
| `/tmp`、`/dev/shm` | tmpfs | 属于内存文件系统，不能计入 136G 持久盘余量 |

2026-10-27 15:27 已按用户确认删除 48 的旧归档和损坏的 P1 残片，整盘可用空间由约 101G 增至 136G，净释放 **37,438,140,416 bytes（34.87 GiB）**。详细清单见 [cleanup_48_20261027.json](ANALYSIS/style_repr_injection_matrix_20261026/measurements/plan/cleanup_48_20261027.json)。其中旧 `results_nonkey` 约 22G 可能包含唯一的历史非关键结果，现已删除；当前矩阵中的 J checkpoint 副本、关键基线、训练数据和正在运行的任务均保留。

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

`/home/ds/Workspace` 合计约 **170G**：

| 工作区 | 大小 | 主要内容 | 清理判断 |
|---|---:|---|---|
| `Workspace/DiT` | 99G | 本项目代码、数据、当前矩阵和关键结果 | 当前只允许逐项审计，不整体清理 |
| `Workspace/xy` | 72G | Qwen 书法/LoRA 系列实验 | 属于其他工作区，需确认引用后再动 |

`Workspace/xy` 的主要占用为：

- `qwen-calli-20261022/model`：31G；
- `qwen-lora-20261023/base`：31G；
- `qwen-lora-turbo-20261023`：4.3G；
- 其余 Top10、reference、writer-res 等 LoRA 目录：约 6G。

前两个 31G 目录是这里最有价值的清理候选，但很可能是基础模型或独立实验的唯一副本；在确认其他项目不再引用、或可以可靠重建/重下之前不删除。

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
| P1 | `Workspace/xy` 两个 31G Qwen 基础目录 | 最多约 62G | 确认无唯一权重、无软链接/配置引用，并评估重新下载成本 |
| P1 | `realsr/data/decoded` | 32G | 有确定性重建路径且原始输入完整 |
| P1 | `realsr/experiments/diffusion` | 15G | 项目负责人确认不再需要 checkpoint/指标/图像 |
| P2 | Hugging Face Qwen 缓存 | 约 16G | 确认各 Qwen 项目不再离线依赖；删除后需要重新下载 |
| P2 | 未使用 Conda 环境 | 单环境约 8G | 先按环境名核对项目；`vcp-embryo` 明确保护 |

不建议把 `/home/ds/miniconda3/pkgs` 的 5G 直接视为可释放空间：`conda clean --dry-run --packages/--tarballs` 的当前结果均为 **0 bytes** 可清，说明这些包仍可能被环境硬链接引用。Torch inductor、PyIQA/LPIPS checkpoint 也服务于当前训练和统一评测，不列入自动清理。

## 清理与保护规则

1. 当前队列每个任务启动前检查至少 25GiB 持久盘余量；136G 目前足够，但 checkpoint 会继续增长。
2. 删除前必须确认目标不在 `ready.json`、[ckpts.md](ckpts.md) 关键权重表和活动进程打开文件中。
3. 大数据优先本机重建；不为清盘把 GB 资产跨机搬运。
4. 其他项目只提供候选范围，不因目录名含 `work`、`decoded`、`cache` 就直接删除。

## 变更记录

- 2026-10-27 15:31：完成 48 全盘刷新；确认 DiT 99G、`Workspace/xy` 72G、ERV 1.2T、RealSR 69G、VCP 57G、Conda 约 37G、共享缓存 23G。
- 2026-10-27 15:27：经用户确认删除 48 旧 `_archive_20261027` 与损坏 P1 残片，净释放 34.87GiB；当前训练未中断。
- 2026-10-27 03:40：旧实验首次同盘归档；归档本身不释放空间。
