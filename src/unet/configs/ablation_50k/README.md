# U-Net 50k 笔画拓扑与丢笔消融实验（单变量严格受控）

本目录包含针对 U100-Base (SiLU 无门控) 与 DiC (GELU + 残差门控丢笔) 的单变量隔离消融配置。

## 实验目的
查明 DiC-768 从头训练出现笔画蚕食与断裂的首要诱因：
- 对照基准：U100-Base @ 50k（SiLU，无门控，无 Stage Adapter，B512，完整笔画）
- **Arm 1 (单独加门控)**: `arm1_resgate_only.json` —— 保持 SiLU，唯一变量为开启 `residual_cond_gate=true`。
- **Arm 2 (单独换 GeLU)**: `arm2_gelu_only.json` —— 保持无门控，唯一变量为将激活函数换为 `act_fn='gelu'`。

## 规格与预算
- **全局 Batch**: 512
- **训练步数**: 50,000 steps（从头训练，无损等价起点）
- **硬件步速**: RTX 4090 约 2.65 ~ 2.75 step/s
- **预计耗时 (ETA)**: 各约 5.1 小时，两臂串行执行总计约 10.2 小时。
