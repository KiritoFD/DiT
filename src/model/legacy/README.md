# src/model/legacy — 已归档的模型实现

2026-09-17 归档。**ControlNet 双臂方案已废弃**，主线是 `DiT_2Cond`（`src/model/dit.py`）。

| 文件 | 原位置 | 说明 |
|---|---|---|
| `controlnet.py` | `src/model/` | ControlNetDiT / ControlConditionEncoder / load_main_model。572 行 |
| `dit_skel_1cond.py` | `src/model/` | 骨架单条件 DiT。只被 `train_skel_1cond.py`（同时归档）使用 |

## ⚠ 相对 import 已调整

归档多了一层目录，原 `from . import modules as M` 会解析到 `legacy.modules`（不存在）
→ 已改为 **`from .. import modules as M`**。移动这类文件时务必检查相对导入。

## 仍需要它们的地方

- `src/eval/cpu_eval_worker.py` 的 `--mode ctrl_pair`（**已废弃**，默认已是 `pretrain_g`）
- `src/model/__init__.py` 用 `try/except` 做**可选**导入，顶层名字仍可见（向后兼容）

```python
from src.model.legacy.controlnet import ControlNetDiT
```
