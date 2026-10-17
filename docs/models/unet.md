# CNN U-Net：独立模型，共用训练器

状态：实现阶段，**未启动正式训练**。48 上旧 L24 已按要求停止，最近完整 checkpoint 为45k；旧产物保留。之前的 batch 短测属于旧实现，不能直接作为本版最终性能验收。

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

- 128/256/384 三尺度，32²/16²/8²；ResBlock、GroupNorm、SiLU、标准 encoder–decoder skip；16²/8² SDPA Attention。
- Time、Writer、Script 各自投影后相加，再经 MLP 调制所有 ResBlock；STD 仅经空间卷积编码注入，不直连输出。
- 初始 Writer/Script 从现有 `callig_table.npy`、`font_table.npy` 加载，映射沿用原表；恢复 checkpoint 时不要求再次加载这些初始化文件。
- `y_char` 只作接口兼容，不参与预测。GT 仅为 FM 目标。
- style CFG 保持与 DiT 一致：unconditional 分支仅置 Writer-null，保留 STD 和 Script；CFG=1 只算一次条件前向。训练的 all/style dropout 不抹白 STD。
- 首版只开放 FM，不启动 DDPM 或新辅助监督。

## 恢复与评测接口

`src.unet.io.load_model_from_ckpt(path,device,use_ema=True)` 与 DiT 的公共加载接口一致，但使用独立 U-Net 构造器，`strict=True`，拒绝其他架构。`use_ema=False` 明确选择 raw 权重；两种路径都处理编译前缀。

共享 checkpoint 额外记录 `architecture` 和 `config`，不再使用 U-Net 私有的存盘格式。恢复使用公共 optimizer/scheduler/RNG 数据，但不声称 DataLoader 逐样本精确重放。

## 验证与未完成工作

已验证：模型前后向、Writer/Script/STD 梯度、时间尺度、CFG=1、Writer-null 语义、严格加载；共用循环的 CPU 小批次优化、周期存盘、阶段停止存盘和不缩短 horizon。

未做：本版完整 GPU 训练/评测闭环、NCHW/Channels Last 与编译的最终 profile、长训稳定性、条件推理缓存和生成质量验收。不把旧短测速度或结构正确性当作这些工作已经通过。
