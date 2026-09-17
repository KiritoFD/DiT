# src/train/legacy — 已归档的训练入口

2026-09-17 归档。**当前唯一在用的训练入口是 `src/train/train.py`**。

| 文件 | 行数 | 说明 |
|---|---|---|
| `train_controlnet.py` | 745 | ControlNet 双臂训练。**ControlNet 线已废弃** |
| `train_repa.py` | 523 | 早期 REPA 训练脚本（REPA 已并入 `train.py` 的 `--w-repa`） |
| `train_skel_1cond.py` | 195 | 骨架单条件训练（对应已归档的 `dit_skel_1cond.py`） |
| `_debug_repa_smoke.py` | 94 | 调试脚本 |
| `_debug_repa_l2_smoke.py` | 93 | 调试脚本 |
| `_debug_stageB_dryrun.py` | 40 | 调试脚本 |

## 引用已同步更新

`scripts/ops/run_v8_3stage.sh`、`scripts/legacy_sh/*.sh`、`tools/**` 等 10 处
引用已改为 `src/train/legacy/<name>.py`（或 `-m src.train.legacy.<name>`）。

## 现状

`src/train/` 只剩：
```
train.py            # 主入口
cli.py              # argparse（162 个参数）
early_stop.py       # 早停（被 train.py 用）
latent_structure.py # 实例骨架结构 loss（被 train.py 用）
legacy/             # ← 本目录
```
