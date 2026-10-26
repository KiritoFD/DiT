# Checkpoint登记

更新时间：2026-10-27 00:49（Asia/Shanghai）。登记当前训练、关键对照及明确不能使用的权重；完整历史实验索引见[exp.md](exp.md)，实时训练状态见[training.md](training.md)。

## 路径与检查口径

下文路径均相对对应机器的仓库：4090为`/root/Workspace/xy/DiT`，48为`/home/ds/Workspace/DiT`。文件在远端，路径不是本地下载链接。

本轮状态扫描核对文件大小、ZIP目录结构，以及新checkpoint的`.done`清单大小；SHA列来自已完成落盘清单或起点校准。没有为了登记而重新哈希所有GB级历史文件。目录名或文件存在本身不代表可用。

## 当前实验已保存

| 机器 | 实验 / 步数 | 大小，十进制GB | 可用性与用途 |
|---|---|---|---|
| 4090 | S0@1,000 | 1.827 | ZIP结构及`.done`大小相符；当前S0最新持久化恢复点；无完整生成评测 |
| 48 | J2@50 | 2.617 | ZIP结构及`.done`大小相符；含配套Decoder raw/EMA及优化器；串行到J2时从这里恢复至5k |
| 48 | J0@450日志位置 | — | **尚无checkpoint**；当前训练内存状态，不是可加载权重 |
| 两端 | S1/S4/S6、J1 | — | 排队，未产生本轮checkpoint |

S0@1k：

```text
assets/style_matrix_20261027/cold50k/4090/execution/runs/S0/checkpoints/0001000.pt
SHA256 e93dc2142629749fb682c0137c5c4ed3e6b71c5bf4ba966c868119b62164281a
```

J2@50：

```text
assets/style_matrix_20261027/deep85_decoder/48_v2/execution/runs/J2/checkpoints/0000050.pt
SHA256 2962f1c8cbb46f2a4cdc63f6fa1ea34c5254ef412484ed93394820482019c0f0
```

本轮外部J2的checkpoint内部`joint_arm`为旧代码代号J3，外部实验名见`matrix_identity`；它是完整图像监督+Decoder重建联训，不是旧“仅重建J2”。

## 成熟模型与参考权重

| 机器 | 模型 | 现存步数 | 本轮用途 |
|---|---|---|---|
| 48 | Deep-Sym4 114M | 50k、75k、85k | **J0/J1/J2统一加载85k EMA**；另外两个保留作历史对照 |
| 48 | U100 Base 72M | 55k | 历史视觉参考，非当前S/J初始化 |
| 48 | Qwen condition A | 90k | 历史条件消融，使用修正Writer映射后的评价 |
| 48 | Qwen condition C | 90k | 同上，非当前主线初始化 |
| 4090 | P1 | 10k | 保留；当前S组从零训练、不加载它 |
| 4090 | P2 | 20k | 保留；当前J组从Deep85开始、不加载它 |

上述文件本次扫描ZIP目录完整。精确路径：

```text
# 48：Deep系列，每个文件1,824,546,895 bytes
assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0050000.pt
assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0075000.pt
assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt
# Deep85校准时确认的完整SHA256
f80304d7de17f334ebf57dfda312988dbd5e994a18f68e6d36f8cb30d5696349

# 48：Base与Qwen对照
assets/results/unet_connectivity_100000/baseline_resume/20261018-030921-Unet-flow/checkpoints/0055000.pt
assets/results/qwen_cond_arm_a/20261025-033733-Unet-flow/checkpoints/0090000.pt
assets/results/qwen_cond_arm_c/20261025-042652-Unet-flow/checkpoints/0090000.pt

# 4090：P1/P2，每个文件1,824,677,314 bytes
assets/u100_p1p2_20261025/p1_runs/20261025-113332-Unet-flow/checkpoints/0010000.pt
assets/u100_p1p2_20261025/p2_runs/20261025-152857-Unet-flow/checkpoints/0020000.pt
```

## 排除项与保留记录

| 机器 | 文件 | 状态 |
|---|---|---|
| 48 | 上述P1同名路径的10k文件 | **不可用**：仅1,323,630,592 bytes，ZIP目录不完整，是中止传输残留；不能当作48已有完整P1 |
| 4090 | `assets/style_matrix_20261026/plan_final/4090/execution/runs/S0/checkpoints/0000569.pt` | 完整旧P1 warm-start实验；已停止，排除本轮随机初始化S组比较 |
| 48 | `assets/style_matrix_20261027/deep85_decoder/48/` | 首次配置校验失败目录；未产生正式训练checkpoint；有效计划是`48_v2` |

上述产物全部保留，不自动删除或用新文件覆盖。

## VAE配套与后续登记

原始VAE目录两端均为`pretrained_models/sd-vae-ft-ema`，latent scale=0.18215、shift=0。J2冻结编码部分指纹为`01a66068d9d27a17bb3d7d78abf66b14afe55ba6d023cdae3b258a09fe96512c`。

J2推理必须从同一checkpoint导出所选raw/EMA Decoder，不能直接使用原始Decoder。现有统一评测入口`tools.style_matrix.runtime.evaluate`负责配套导出、冻结指纹和scale/shift核验。普通S组继续使用原始VAE。

新权重落盘后补充机器、模型、步数、完整路径、`.done`状态、SHA、raw/EMA及Decoder配套信息，再附对应固定集指标和海报。尚未评测时明确写“未评测”，不将短预检当作生成质量结论。
