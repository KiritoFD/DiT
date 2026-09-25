# 20260913 旧备份抢救记录

`/root/Workspace/xy/dit_data_backup_20260913/`（622 GB）已于 2026-09-23 删除，释放 536 GB。

## 抢救出的内容

| 位置 | 内容 | 体积 |
|---|---|---|
| `records/` | 1810 个记录文件（csv/json/md/txt/log） | 59 MB |
| 服务器 `_rescued/20260913_backup/eval_imgs/` | 236175 张 eval 生成结果图（56 个实验 × 各 step） | 6.3 GB |
| 服务器 `_backups/dataset_20260923.tar.gz` | dataset 打包（24 G → 15.45 G） | 15.45 GB |

## 记录数据重点文件

- `5script/master_results.csv` —— **核心汇总表**（所有实验的总指标）
- `5script/results/*/eval_stdskel_summary.csv` / `eval_stdskel_batch.csv` —— 骨架 IoU 逐批指标
- `5script/results/*/*/eval_auto_*.json` —— 逐 step 自动 eval 结果
- `5script/results/*/*/metrics.json` —— 各 run 指标
- `5script/results/*/*/resolved_config.json` / `source_manifest.json` —— 完整配置溯源
- `5script/eval_strata/*.csv` —— 评测集分层定义

## 备份背景

该备份是 2026-09-13 项目重构（`5script/` → `DiT/assets/` + `DiT/data/`）前的整目录快照，
含 v8/v9/v10/v11 代共 15 个实验目录 + 615 个 ckpt。

删除时注意：备份内大量文件有硬链接，目录体积 619.61 GB 但实际仅释放 536.05 GB
（差额 84 GB 是与当前项目共享 inode 的部分，如 `pretrained_models/`）。
