# 37 — 数据布局与归档（2026-09-06 大整理）

活跃训练/评测数据见仓库根 **`data_path.json`**（唯一权威清单）。
其余历史数据集已归档至远程 `data/archive/`（8 类，75G，含 README 清单），原始 `dataset/` 未动。

## 归档摘要
legacy_latents 6.7G / legacy_images 18G / legacy_skeletons 6.9G / legacy_plans 0.16G /
legacy_csv 87M / std_legacy 269M / mid_data 3.8G / results_legacy 40G（5script 之前的结果）。

## 根目录脚本整理
~470 散文件归入 `scripts/{scratch,legacy_py,legacy_sh,exp_configs,misc}` 与 `logs/`；
运行入口保留在 `_sync_work/`（本地同步）与 `tools/`。

## 教训
批量 mv 的排除保护要用"先 mv 后恢复"之外的方式核对；整理后必须跑活跃资产在位检查
（本次凭 data_path.json 清单抓回 3 个误归档文件）。
