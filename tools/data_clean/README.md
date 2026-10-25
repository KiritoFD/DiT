# 非破坏性数据清洗

范围是当前Top10基础真迹和固定评测集，Top5由同一清洗结果派生。所有输出写新目录；不删除、重画、裁切或反相原图，不改字符/书家/书体ID，不修改训练配置，不启动训练。需 Python 3.10+、NumPy、Pillow、SciPy、scikit-image。

## 三个入口（仓库根目录运行）

```powershell
python -m tools.data_clean.scan --train-csv exp-std/csv/train_top10_noaug.csv --eval-csv exp-std/csv/eval200_fixed.csv --top5-csv exp-std/csv/train_top5_noaug.csv --root . --output assets/data_clean/NEW/scan --workers 8
python -m tools.data_clean.review --scan assets/data_clean/NEW/scan --font tools/fonts/SimHei.ttf --output assets/data_clean/NEW/review
python -m tools.data_clean.build --scan assets/data_clean/NEW/scan --decisions tools/data_clean/known_decisions.json --output assets/data_clean/NEW/candidate
```

`scan`只接受基础/noaug CSV。输入CSV快照、逐图文件/RGB/灰度哈希、尺寸、墨量、连通域、质量候选、重复分组和近重复对都保存。缺图、不可读图、错误画布、几乎空白明确记录，不静默跳过。

`build`核验输入快照和审计记录哈希，再检查保留原图是否发生变化。输出目录已存在即报错，不覆盖上一轮。已审计的原图若修改，必须重新扫描。

## 重复和划分策略

- 灰度像素逐位相同的图归组，能识别不同PNG压缩、文件名、异体标签下的重复；另存RGB与文件哈希以便追溯。
- 近重复仅搜索同一字符或显式候选族（目前仅`靑/青`），不宣称穷尽所有跨标签近重复。按原墨迹bbox归一化后，用pHash筛选，再计算灰度SSIM、Ink-IoU和MAE。
- 自动接受近重复连接需同时满足：Hamming≤8、SSIM≥0.985、IoU≥0.95、MAE≤0.02。较弱相似只入`near_pairs.csv`供审核，不自动归组。分组取传递闭包，仍可能有误合并，必要时核查该表。
- 同一`src_image_path`还会进入**划分组**，用于隔离同原始裁切来源，不据此断言同作品/同扫描集；这里不是严格作品级留出。
- 任何划分组只要涉及**原始**评测集，就整体不进入训练，即使那张评测图也被隔离。保留集通过像素哈希、重复组和划分组的跨集合互斥断言。
- 同重复组书家/书体/字符标签存在不能解释的差异，整组待审，不猜哪个标签正确。`靑/青`只作为已声明的候选等价族，保留现有代表行，不改Unicode、glyph_id或STD条件。
- 训练组内固定顺序保留一个代表；评测保留原顺序中可用的代表。重复图也不能通过增强副本重新进入训练，故本工具不输出增强CSV。

## 质量规则不是脏污真值

边缘长条、碎片多、墨量过低/高等只算候选。默认候选保守隔离，而不是删除；合法的刂、亻、长竖和碑刻飞白会误报，必须看审核页恢复，不能直接把连通块小当作噪点。

决定清单逐图绑定文件SHA256及原因，`keep`仅豁免统计质量标记，不能绕过空白/坏图、标签冲突和训练评测隔离。`exclude`会传播到重复组。已确认的脏图示例在`known_decisions.json`，不是全量质量真值。

视觉审核后记录决定：

```powershell
python -m tools.data_clean.review --scan assets/data_clean/NEW/scan --output assets/data_clean/NEW/decisions.json --decision-action keep --paths data/top10_style23/imgs/001218.png --reason "视觉审核：右侧为合法刂旁" --base-decisions tools/data_clean/known_decisions.json
```

把新的决定文件传给`build`，输出另一个目录。不能凭字符名批量放行未看过的图，也不自动用NFKC合并所有简繁异体。

## 产物与使用边界

`train_top10_candidate.csv`、`train_top5_candidate.csv`、`eval_top10_candidate.csv`、`eval_top5_candidate.csv`：保持原CSV行/ID契约，供后续经确认的实验使用。只筛选原图，若原latent缓存确实与原像素/VAE一致，无需为筛选本身重编码；本工具不认证旧缓存的像素内容。

`quarantine.csv`记录逐图排除原因（原因可重叠，不能相加当唯一图片数）。`eval_annotations.csv`分别标注全局见字、writer×script×character见字及候选族见字，不再混用“原图Seen/零样本”标签。`report.json`包含分槽位和来源覆盖、输入输出数量、文件哈希及待审数量。

输出标记`candidate_not_training_approved`：这是完成了规则扫描和部分视觉复核的候选，不是全量专家验收，也不承诺未标记图全部干净。新评测数量和组成已变化，不能把新旧均值直接写成模型提升。旧模型学过被移出的训练图，不会因此获得干净的重新评测资格；严格结论应来自使用新划分的新训练。

本次结果位置和数量见 [专项记录](../../ANALYSIS/data_cleaning_top10_20261021.md)。运行关键测试：`python -m pytest tests/test_data_clean.py -q`。
