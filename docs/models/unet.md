# CNN U-Net：独立模型，共用训练器

状态：按最新指令仅保留基线与基线+Writer-only两组，由同30k checkpoint分叉至累计100k。
此前60k组合版队列取消；实际切换/运行状态见[state.md](../../state.md)。
条件残差门控已预先实现、默认关闭，**不修改已在训或排队的远端源码快照**。

## 边界

U-Net 的模型和严格构造器放在 `src/unet/`，不访问 DiT 的 `blocks`、`final_layer` 或 embedding 属性。它不是骨架预测器，直接输出 `[B,4,32,32]` FM velocity。

训练入口不复制 DiT 训练循环，而是组装**同一个 `src.train.trainer.TrainState`，调用同一个 `training_loop`**。

| 部分 | 共用实现 |
|---|---|
| latent/STD 数据与 sampler | `src.train.data_setup.build_data` |
| 主循环、日志、有限性检查 | `src.train.trainer.training_loop` |
| 条件装配、输入、clip/optimizer | `src.train.step` |
| FM 主目标 | `src.train.runtime.mainline.LatentStdObjective` |
| LR | `src.train.step.build_cosine_scheduler`，DiT 也调用同一函数 |
| EMA | `src.train.ema`，含相同 warmup 和更新间隔语义 |
| checkpoint | `CheckpointManager`、原有异步写、完成 SHA256 清单和 RNG |
| 评测 | 原有训练评测回调、Heun/CFG、指标和 PNG/DONE 协议 |

U-Net 特有的只有模型构造，以及 `condition_transform(model,batch,kwargs,x0,step)` 适配器：沿用已有 STD 离线 bank 和概率调度，不运行 DiT 的 glyph 健康度探针，也不重复传输已在 GPU 的干净 STD。

## 步数由调用方决定

`max_steps` 没有隐式训练预算，未提供有效值就报错。`configs/fm_48.json` 的200k只是可修改的示例配置，不是启动器或循环内覆盖值。

```powershell
python -m src.unet.train --config src/unet/configs/fm_48.json --max-steps 80000 --global-batch-size 192
```

分段暂停但保持完整调度：

```powershell
python -m src.unet.train --config src/unet/configs/fm_48.json --max-steps 80000 --stop-at-step 3000
```

`stop_at_step` 不改变 LR horizon；`lr_decay_steps=0` 表示使用 `max_steps`，也可以单独显式配置。复用 DiT 的 warmup/cosine 函数，不另维护一份数学实现。

`ckpt_every` 与 `eval_every` 遵循公共训练器的同一刻度规则；额外里程碑通过配置 `checkpoint_steps` 给出，不在代码中硬写100、1k、5k。阶段停止点或最终步会保存 checkpoint。

`tools/launch_unet48.py` 现在只做有限 batch profile：不启动训练，不修改 horizon、LR、EMA 或增强日程。正式启动需要用户显式执行训练命令。

## 模型和条件

- 默认128/256/384三尺度，72M配置为192/384/576；空间尺度32²/16²/8²。
- 基线实际14个ResBlock、5个Attention，GroupNorm/SiLU、标准encoder–decoder skip；16²/8² SDPA Attention。
- Time、Writer、Script 各自投影后相加，再经 MLP 调制所有 ResBlock；STD 仅经空间卷积编码注入，不直连输出。
- 可选`writer_film_rank=16`：每个ResBlock额外读取32D Writer embedding，经32→rank→2C输出独立scale/shift，叠加到原FiLM；不读取time/script。默认0关闭。
- **DiC (CVPR 2025) 扩展开关**：
  - `residual_cond_gate=true`（75.04M）：逐通道有界残差门控 $g(c) = 1 + 0.5\tanh(W_g c) \in [0.5, 1.5]$，零初始化保证步 0 恒等。
  - `stage_specific_cond=true`：多尺度解耦条件路由（32²/16²/8²），支持 `adapter`（72.88M，零初始化残差）与 `independent`（72.80M，全独立投影层）。
  - `act_fn='gelu'`：全网激活函数升级为高阶光滑 GELU。
  - 全功能 DiC 组合版：`DiC_U100_b768`（75.83M），Batch=768，TorchInductor 0-break 编译，MFU 达到 26.43%。
- **U100-Deep-Sym4 (114M) 规模加深突破版（当前 SOTA 基模）**：
  - 参数量：**114,003,044（114.00M）**；
  - 骨干通道：保持宽底 `[192, 384, 576]`；
  - 深度倍增：`num_res_blocks=4`，Encoder 与 Decoder 对称加深至每尺度 4 个 ResBlock（全网共 26 个 ResBlock）；
  - 核心设计原则：**彻底关闭 Gate 门控（`residual_cond_gate=false`）**，释放纯净恒等残差流 $x + \text{residual}$；采用原生 SiLU 激活；
  - 性能表现：SSIM Mean 达到 **0.6132**，LPIPS 降至 **0.3207**，Ink-IoU 达 **0.3381**，Frag 压低至 **2.14**，全面刷新项目历史纪录。详见 [U100-Deep-Sym4 加深收敛报告](../experiments/2026-10-19-unet-deepening-114m-and-convergence.md)。
- 初始 Writer/Script 从现有 `callig_table.npy`、`font_table.npy` 加载，映射沿用原表；恢复 checkpoint 时不要求再次加载这些初始化文件。
- `y_char` 只作接口兼容，不参与预测。GT 仅为 FM 目标。
- style CFG 保持与 DiT 一致：unconditional 分支仅置 Writer-null，保留 STD 和 Script；CFG=1 只算一次条件前向。训练的 all/style dropout 不抹白 STD。
- 首版只开放 FM，不启动 DDPM 或新辅助监督。

## 算力复杂度与 MFU 基准（RTX 4090）

- **单样本前向 FLOPs**：22.158 GFLOPs（其中 Conv2D 占 99.5%）；
- **单样本训练 FLOPs (fwd+bwd)**：66.473 GFLOPs（标准 $3\times$ 前向计算量）；
- **Node 4090 (Batch 768 @ 1.71 step/s)**：Achieved 87.30 TFLOPS $\implies$ **MFU 26.43%**（显存 22.18 GiB / 24.56 GiB）；
- **Node 48 (Batch 512 @ 2.76 step/s)**：Achieved 93.93 TFLOPS $\implies$ **MFU 28.44%**（显存 14.74 GiB / 49.14 GiB）。
- 完整推导与演化全景参见 [U-Net 演化与 DiC 扩展报告](../experiments/2026-10-18-unet-progress-and-dic-scaling.md)。

## 恢复与评测接口

`src.unet.io.load_model_from_ckpt(path,device,use_ema=True)` 与 DiT 的公共加载接口一致，但使用独立 U-Net 构造器，`strict=True`，拒绝其他架构。`use_ema=False` 明确选择 raw 权重；两种路径都处理编译前缀。

共享 checkpoint 额外记录 `architecture` 和 `config`，不再使用 U-Net 私有的存盘格式。恢复使用公共 optimizer/scheduler/RNG 数据，但不声称 DataLoader 逐样本精确重放。

## 验证与未完成工作

已验证：模型前后向、Writer/Script/STD 梯度、时间尺度、CFG=1、Writer-null 语义、严格加载；共用循环的 CPU 小批次优化、周期存盘、阶段停止存盘和不缩短 horizon。

基线另已验证真实B512训练、compile、EMA214/214配对、20.1k新checkpoint严格加载与有效评测DONE。
新结构尚未做GPU batch/profile或质量实验，不把CPU等价性当作连贯性提升证据。

## 连贯性改进（单变量候选）

默认模型不新增任何state_dict key。可选三尺度Skip/STD门控使用`1+.5*tanh(a)`，从1起步；
首轮Skip只训练32²。Decoder可独立指定深度[3,3,2]，新增块零初始化、暖启动前向等价。
严格迁移只允许明确的新参数初始化，旧权重缺失即失败；受控A/B/C全部重建优化器与EMA。
P0包含阈值/小组件、同点Raw/EMA、Heun50/100及0.8尺度干预，后者只作OOD敏感性诊断。
见[配置与预算](../../src/unet/configs/connectivity/README.md)及[诊断入口](../../tools/analysis/unet_connectivity/README.md)。

## 调制与门控

GroupNorm后的条件scale/shift称为AdaGN；历史`film`/`writer_film`参数名保持不变。
Writer-only残差AdaGN与条件残差门控独立配置；前者不等于从共享条件中删除Writer。
公式、开关、初始化、参数量、迁移约束与本轮整理记录统一见[调制说明](unet_modulation.md)。
