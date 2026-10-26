# Deep85 Decoder 联训：48机三组5k

## 目标与起点

三组均严格加载 `assets/results/unet_deep_sym4_114m_60m/20261019-011647-Unet-flow/checkpoints/0085000.pt` 的EMA，重新构建优化器，不继承P1/P2。原文件SHA256：`f80304d7de17f334ebf57dfda312988dbd5e994a18f68e6d36f8cb30d5696349`。

| 外部实验名 | 实际目标 | Decoder | 旧实现内部代号 | 预算 |
|---|---|---|---|---|
| J0 | FM | 冻结 | J0 | 5k |
| J1 | FM + 生成图像监督 | 冻结，但保留对latent的梯度 | J1 | 5k |
| J2 | FM + 生成图像监督 + GT重建 | 训练 | J3 | 5k |

外部J2对应旧实现内部J3；旧内部J2是“仅重建”，本次没有该实验。checkpoint同时保留外部实验identity与内部objective代号，避免历史结果混淆。

J2−J1同时包含Decoder解冻和重建正则，不能进一步将收益独立归因于其中一项。

## 共同数据与latent契约

使用Deep85原训练CSV中的26,002条未做图像增强的记录，原77,823条中其余记录带图像增强。三组统一筛选，避免用未增强PNG监督增强后的latent。现有latent缓存复用，不重编码。STD strong增强保留。

保留Deep85原始语义Writer ID及原remap：`53→0,365→1,461→2,479→3,601→4,672→5,703→6,760→7,956→8,9001→9`。不能使用P2的canonical 0–9标签直接读Deep85的书家表。

原配VAE为 `pretrained_models/sd-vae-ft-ema`，scale=0.18215、shift=0；Encoder、quant_conv、post_quant_conv冻结并记录指纹。GT读取按img_id精确对应，校准中检查GT编码及缓存重建误差；checkpoint与评测持续核验冻结参数，J2 raw/EMA都携带配套Decoder。

## 配方与执行

- FM有效batch512，microbatch512；BF16、channels_last、fused AdamW、torch.compile(default)；不使用梯度检查点。
- 共同seed90、Writer配对比例30%；U-Net LR2e-5，Decoder LR1e-5；warmup300，5k cosine horizon，最低比例0.1；EMA每步0.999。
- 图像辅助监督每步从同一有效batch选16张，microbatch8；不是对512张全部计算图像损失。沿用此前明确约定的Decoder小批图像监督协议。
- 图像损失为L1 + 0.1 LPIPS(Alex) + 0.1多尺度Sobel；生成图像监督取t∈[0.2,0.5]。
- 新起点重新校准图像梯度至初始FM梯度的10%，不复用P2校准系数；GT重建权重1。所有组共享校准文件与抽样随机规则。
- 为验证最重分支，先将正式J2运行50步并保存，随后串行J0→J1→J2；J2从该50步恢复到5k，保持同一5k调度、数据游标和RNG。50步计入总预算，不额外训练。
- 1k/2.5k/5k保留中间checkpoint；三组结束各做固定187字评测，Writer空间显式raw。保留全部产物，不删除旧checkpoint。

计划及运行根目录：`assets/style_matrix_20261027/deep85_decoder/48_v2`。`phase.txt`标记校准、最重分支预检或正式串行；`execution/status.json`与各组`train.jsonl`记录实际进度。失败停止，不盲目重试。

首次准备目录`48`因checkpoint/eval间隔校验未通过，在训练前停止；保留该失败记录，修正后使用独立`48_v2`，不混入正式结果。

## ETA口径

启动前历史eager短测：J0约1.02、J1约0.65、完整联训（现J2）约0.43 step/s，三组各5k纯训练合计约6.7h；这些是旧fixture，不是本轮Deep85正式速度。

本次开启编译，预计会缩短FM部分耗时，Decoder与LPIPS部分不承诺同比提速。先以约7–8h包含校准、首次编译、保存和评测作保守估算，再以本轮J2预检及J0区间计步更新。

## 已完成的起点校准

Deep85真实数据校准已通过：FM梯度范数0.3096065，单位图像损失的U-Net梯度范数1.4541152，公共图像权重0.0212917472，实际初始比例0.1。Decoder的图像/GT重建梯度范数分别为3.4018762/3.4077587，真实GT与latent配对检查通过。校准没有执行optimizer更新。

冻结VAE编码部分指纹：`01a66068d9d27a17bb3d7d78abf66b14afe55ba6d023cdae3b258a09fe96512c`。

## J2正式前50步通过，串行进入J0

00:43左右J2完成50步并保存checkpoint，编码部分指纹检查及异步落盘检查通过。第1步包含约144秒首次编译；从第1至50步约96秒，区间速度约0.51 step/s，allocated峰值39.812GiB。不能将包含编译的累计0.208 step/s作为稳定吞吐。

J2 checkpoint SHA256：`2962f1c8cbb46f2a4cdc63f6fa1ea34c5254ef412484ed93394820482019c0f0`。50步FM=0.05614、U-Net梯度范数0.3121、Decoder梯度范数1.3162，均有限；这些仅为执行正确性证据，不是生成质量结论。

正式队列已进入J0@0→5k，接着J1@0→5k，最后从J2@50恢复到5k。J2按实测约2.7h完成剩余训练；J0预计约0.8–0.9h，J1尚未获得本轮稳定实测，暂按约1.5–2.1h估算。加上加载、编译、保存、187字评测，暂估从00:44起还需6–7h，即2026-10-27上午约7点前后。此ETA不是质量/收敛保证。
