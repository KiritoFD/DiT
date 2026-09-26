# 归档：未修复简繁错配的历史旧数据 (Archived Unfixed Data)

> **归档日期**：2026-09-26  
> **归档原因**：杜绝未来实验误用包含简繁错配的旧版数据。

## 归档文件清单
1. `train_50k_v2_unfixed.csv`：原始未清洗训练集，包含 1,921+ 处简繁错配（如标为简体“争/书/国”但真值图片实际写繁体“爭/書/國”）。
2. `eval_v13_strict_unfixed.csv`：未清洗的严苛外推评估集，包含 19 处简繁错配（如“赵孟頫-隶-升”，真值字形为繁体“陞”）。
3. `train_50k_v2_augmented_unfixed.csv`：早先基于未清洗版拼装的合成字体数据。

## 正确替代基准
- 全量清洗训练基准：`assets/train_50k_v2_fixed.csv`（包含 2,150 处简繁与骨架修复）
- 规范严苛评估基准：`assets/eval_v13_strict_fixed.csv`（及软链接/安全副本 `assets/eval_v13_strict.csv`）
- 稀缺字增广新训练集：`assets/train_50k_v2_fixed_augmented.csv`
