# 48/64h后续实现：任务边界交接

2026-10-29 09:17。阶段结论报告已提交为`3acab7e`，250行；研究归档包约4.1MB，版本与SHA见[archive_info.json](exports/archive_info.json)。该zip冻结于报告提交，不随之后的运行文档变化。

## 当前训练不变

- 4090：S9约44.8k/50k、1.239step/s。之后仍按当前计划从同一个50k分叉FM/Pair各至60k。
- 48：J4–J7的5k阶段已完成，门槛选择为空，S11已约14.15k/50k、1.299step/s。
- 本次未向当前活动源码目录解包、未停止进程、未修改当前实验spec或权重。

## 后续修正

当前S9-FM/Pair的24h计划已固定：50k→60k为一次10k LR重启，之后60k→75k另一次15k重启。
若S10或强基线直接50k→75k只做一次25k余弦，虽然曝光相同，LR轨迹不同，无法作严格的成熟架构归因。

新版为S10、S0/S4、S11补齐相同的两段LR：50k→60k、60k→75k，重启点、200步warmup、峰值比例和衰减区间均与S9分支对应。总训练量仍增加25k，没有额外增加10k曝光。
新spec使用独立`*_60_matched`与`*_75_matched`名称、身份链及任务清单，不改当前24h的配置。

4090后续继续S10冷启动、S9分支成熟、退化检查和强基线选择；48在当前S11@50k后接续匹配LR成熟对照与独立seed。原64h时钟与预算门槛不重置。

## 交接机制

独立代码包仅17,408 bytes，SHA256为`5fd7225655d0bbe22796bb286f8e0737f96fefa247085243fecc98daed0487c5`。
代码与观察器暂存在两机`assets/style_matrix_20261028/later_stage/`，没有覆盖当前训练器。

| 机器 | 等待的完成边界 | 当前后台登记 |
|---|---|---|
| 4090 | S9Pair@60k的Generator-swap机制评测完成 | waiting_for_completed_boundary |
| 48 | 当前S11@50k的Generator-swap机制评测完成 | waiting_for_completed_boundary |

观察器只使用标准库，不加载模型或占GPU。边界产物完成后必须取得全局GPU锁并检查GPU空闲，才结束旧CPU控制器、解包未来模块并重新封存。
当前GPU任务仍持锁时持续等待；如果旧控制器已进入另一条成熟LR轨迹，就登记deferred并保留原任务，不回滚权重。
未知GPU所有者、旧控制器身份无法核实或代码包SHA不符时同样保留现场，不强行接管。

## 验证与溯源

`python -m tools.style_matrix.later.checks`已通过CPU检查：四种对照的两段LR逐点相同，身份链正确，源spec保持不变。
没有新增GPU实验、重跑当前训练或跨机传输checkpoint；后续代码尚未被提前启用。
激活状态读取各计划根的`later_handoff.json`；成功后为activated，并保存原截止时间。
源代码入口为`tools/style_matrix/later/`，共5个文件；正式后续模型结果仍需等待对应训练与固定集评测。
