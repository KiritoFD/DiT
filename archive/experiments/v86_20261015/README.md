# v86 后训练代码归档

2026 年 10 月 15 日按用户决定停止后训练，主线转为 v88 基模。这里归档 20 份配置和 16 个实验专用脚本，未删除模型权重、评测图像或日志。逐文件原位置和 SHA256 见 [manifest.json](manifest.json)，移动后逐项核验字节一致。

## 使用边界

这里是历史源码记录，不是新的启动入口。脚本保留原内容，其中 `parents[2]`、相对配置路径和日期日志名仍指向原目录布局。需要复现时，在独立 checkout 按 manifest 的 `original` 路径恢复；先核对目标不存在，再复制，不能直接覆盖当前代码或在归档目录启动训练。还需准备原数据、预训练表、Critic 和 checkpoint。

归档的源配置曾随新计划调整，不能代替历史 run 的 `resolved_config.json`。例如 A100 源配方后来改为10k，但完成的 run 是20k；B25 完成到20k时用的是30k horizon。实际参数与结果见 [实验归档](../../../docs/experiments/2026-10-15-posttraining-and-v88-review.md)。

## 保留的通用模块

`src/train/strong_style.py`、`src/train/writer_table_tuning.py` 和 `src/loss/writer_objectives.py` 仍有测试与历史复现用途，不属于死代码。B25 的逐字、梯度回放和通用校准工具仍在 `tools/analysis/`，梯度回放已改读本归档中的 B25 配方。

15% 巩固与80%匹配对照被取消，未产生训练结论。旧队列不再作为活动入口；当前入口为 `tools/analysis/run_v88_base.sh`。本次只整理本地仓库，没有热更新4090正在运行的训练环境。
