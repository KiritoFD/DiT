# legacy/ — 历史归档（2026-09-13 整理）

**只读归档，勿在新代码中 import。** 正式代码在 `src/`，正式工具在 `tools/`。

| 子目录 | 内容 |
|---|---|
| `dit_core/` | 迁移到 `src/` 之前的旧核心：`models.py`、`train.py`、`sample*.py`、`diffusion/`、`lora.py`、`controlnet_dit.py`、`dataset.py`、`losses.py` 等 |
| `configs/` | s5–s19 时代实验配置（`exp_*.json`、`s*.json`、resume/probe 配置） |
| `scripts/` | 旧脚本：数据准备（`prepare_mccd_*`、`preprocess_*`）、监控（`pull_monitor`、`monitor_s13`）、远程辅助（`remote_*.py`）、旧前端（`flask_app.py`、`gradio_app.py`） |
| `data/` | 旧 CSV 与报告 JSON（fame/gt/kailishu 审核、eval 划分等） |
| `misc/` | 杂项产物（日志、旧 npz/tgz；被 .gitignore 忽略，仅本地保留） |

说明：
- 这些文件在历史实验中有用，但当前流水线（`src/train/train.py` + `src/eval/*` + `tools/*`）不依赖它们。
- 仅两处旧工具 `tools/evaluation/batch_test_fp16.py`、`tools/remote_b4_memtest.py` 曾 `import models/diffusion`；如仍要跑，请把它们也归档或改 import 到 `legacy.dit_core`。
