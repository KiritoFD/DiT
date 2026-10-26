# 训练进度

更新时间：2026-10-27 03:20（Asia/Shanghai）。延长轮已实现并双机重新拉起：实施与实测见[延长轮记录](ANALYSIS/style_repr_injection_matrix_20261026/extension_round_20261027.md)，计划细节见[1027.md](1027.md)，权重见[ckpts.md](ckpts.md)。

## 当前状态

| 机器 | 任务 | 最新步数 / 目标 | 状态 | 实测速度 | 峰值allocated |
|---|---|---|---|---|---|
| 4090 | S0（matrix64计划，从1k续训） | 5,450+ / 50,000 | 正常计步，已越过旧停滞点1,350；5k评测已触发 | 区间约1.54–1.61 step/s | 20.0 GiB |
| 48 | J1（从575续训；J0@5k已完成并评测） | 1,000+ / 5,000 | 正常计步 | J1实测0.880 step/s | 38.6 GiB |

旧队列`cold50k/4090`（S0停滞@1350）与`deep85_decoder/48_v2`（J0完成5k、J1@575中断）已停止并保留存档；进度已迁移进新计划。S0停滞堆栈显示主进程与全部worker均为futex用户态等待，根因未定位，重启后未再现，按暂态处理。

## 已挂串行与ETA

| 机器 | 队列（matrix64） | 剩余预算 | ETA |
|---|---|---|---|
| 4090 | S0→S1→S4→S6→S8 各50k →（selection后）一队50k→75k | S0剩44,550步；其余各50k；延长2×25k | 实测速率估算56.6h（wall上限64h，超时从延长段截断）；stage2需人工写入selection.json |
| 48 | J0→J1→J2→J3 各5k → S2/S3/S5/S7 各50k → J0–J3延长至12k | J0已完成、J1剩约4,000、J2剩4,950、J3 5k；S 200k；J延长4×7k | 实测速率估算53.0h（wall上限64h），全自动无人工门 |

实测速率（compiled，探测窗口）：4090 S1 1.550 / S4 1.542 / S6 1.577 / S8 1.566 step/s——DynamicConv（S6/S8）与Spatial AdaGN同速，S8可全速跑。48：J1 0.880（旧eager估算0.65）、J3 0.765、S2 1.714 / S3 1.722 / S5 1.687 / S7 1.714 step/s；J3首次运行即验证无重建分支（reconstruction_samples=0）、decoder图像分支梯度0.022>0。

## 运行位置

| 项目 | 4090 | 48GB |
|---|---|---|
| SSH别名 | `4090` | `48` |
| 仓库绝对路径 | `/root/Workspace/xy/DiT` | `/home/ds/Workspace/DiT` |
| 计划目录 | `assets/style_matrix_20261027/matrix64/4090` | `assets/style_matrix_20261027/matrix64/48` |
| 启动 | `run_stage.sh`（stage1→轮询selection24h→stage2） | `run_stage.sh`（单stage全自动） |

封存口径变更：>1MB大资产在ready.json只记path+size不哈希（用户决策，64h运行不做GB级重复哈希）；小文件（spec/代码/CSV头）保留SHA，每次启动与每任务前校验。实际进度在`execution/status.json`与`runs/<arm>/train.jsonl`。

## 最近完成的检查

- 4090四臂探测完成（上表），S8=R2×I3组合原生支持、无速度损失；fm曲线正常。
- S2/S3 KV token预检通过：8个可区分KV token（token_std≈0.98），attention熵2.032<均匀2.079，无单token退化。
- J3（内部臂J4）首跑通过：图像分支正常、decoder梯度非零、无重建分支。
- 48的STD bank配对校验两次按设计拦截错误bank（bank_64与bbox64 store不配对），随后按[Data.md](Data.md)配方在48本地重建bank，SHA与4090完全一致（`bc3db05f…`）——零GB级传输。数据规则见[AGENTS.md](AGENTS.md)。
