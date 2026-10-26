# 2026-10-27：恢复编译与50k从零训练

## 修正原因

新矩阵入口曾强制 `compile=False`，并在训练器中拒绝编译。这是执行方式回退，不能据其约1 step/s推断新模型需要放弃实验或缩短到不充分训练。

已有基础设施见 [infra_1015](../../docs/infra_1015.md)、[runtime_20261017](../../docs/infra/runtime_20261017.md) 与 [operator_fusion_infra](../../docs/operator_fusion_infra.md)。其中明确保留BF16、编译、fused AdamW、latent缓存与异步checkpoint。Deep114原运行配置也实际启用compile，历史后期约1.72–1.73 step/s，B512显存约20.47GiB。

新矩阵此前48机eager完整B512约0.96–0.99 step/s、峰值约42GiB；并非仅由梯度累积导致。但历史结果不是新矩阵编译后的实测，不能承诺完全等速。

## 本次配置与代码

- 使用 `torch.compile(raw, mode='default')`，仅编译前向模型，checkpoint/EMA仍对应原始模块，沿用已有缓存目录。
- 不开启梯度检查点。有效batch512，首个任务microbatch512。
- S组整个U-Net随机初始化，不加载P1/P2或Deep85权重；旧Writer路径固定null，新Writer路径独立。
- Top10缓存数据、原VAE、STD strong、FM保持不变；Writer映射明确为当前CSV的canonical 0–9，模型10个Writer槽位。
- 每组50,000 optimizer steps，同seed90、LR6e-5、warmup1500、weight decay0.02、cosine最低比例0.1。scratch模式所有可训练参数组使用相同LR，避免错误沿用续训LR。
- EMA每步0.999；每5k保留checkpoint，不删除已有checkpoint。
- 新准备入口 `tools/style_matrix/prepare/cold_start.py`；新计划独立目录，源文件、数据、配置写SHA清单；串行锁、GPU所有者检查、断点恢复、失败停止继续沿用。

## 队列和估算

新编号遵循最新附件，与旧S编号不混用。

| 机器 | 顺序 | R / I | 预算 |
|---|---|---|---|
| 4090 | S0 Vector/AdaGN | R0/I0 | 50k |
| 4090 | S1 Vector/Spatial AdaGN | R0/I1 | 50k |
| 4090 | S4 Writer×Local CNN/Spatial | R2/I1 | 50k |
| 4090 | S6 Vector/DynamicConv | R0/I3 | 50k |
| 48，待启动 | S2、S3、S5、S7 | R0/I2、R1/I2、R3/I1、R3/I3 | 各50k |

按历史1.7 step/s估算每组8.17h，4090四组合计32.68h纯训练；为启动编译、保存与统一评测留4h，暂估约37h。速度未实测前不以此作为保证。

4090计划：`assets/style_matrix_20261027/cold50k/4090/plan.json`。

运行状态：`execution/status.json`；各组日志：`execution/runs/S*/console.log`、`train.jsonl`。启动命令由同目录 `run.sh` 固化；失败不会自动重试。

J组仍按最新方案从Deep85 EMA派生，这次恢复4090 S队列没有将其误换成随机初始化，也没有启动旧P2队列。
