# src/eval/legacy — 已停用的评测脚本

2026-09-17 从 `src/eval/` 移入。**这些文件没有任何 Python import**，
只被别的文件的注释/文档提及（见 [docs/system/70](../../../docs/system/70_code_review_refactor.md) §3.1）。

移入而非删除的原因：`tools/legacy/auto_eval_ctrl_flow.py` 与
`scripts/legacy_py/auto_eval_ctrl_flow.py` 两个 shim 会 `import *` 它们
（shim 的路径已同步更新）。若要彻底删除，先确认这两个 shim 也没人用。

| 文件 | 原行数 | 为什么停用 |
|---|---|---|
| `auto_eval_gpu.py` | 539 | 已过期：硬编码 `learn_sigma=True`，缺 `glyph_inject_mode`/`norm_type`/`rope`/`glyph_vec_cond` 等字段，且用 `strict=False` |
| `gpu_batch_eval.py` | 446 | 旧 ControlNet 时代（用 `MCCDDataset`、不传 `g`） |
| `gpu_batch_eval_v2.py` | 402 | 同上 |
| `auto_eval_ctrl_flow.py` | 334 | 被 `auto_eval_ctrl.py` + flow 分支取代 |
| `eval_ctrl_ckpt.py` | — | 0 引用 |
| `eval_controlnet_cpu.py` | — | 0 引用 |
| `eval_models.py` | — | 仅注释提及 |
| `metrics_png.py` | — | 仅自身 docstring |
| `latent_condition_probe.py` | — | 仅注释提及 |

**它们的共同问题（也是停用的核心理由）**：模型重建用 `load_state_dict(strict=False)`，
字段漏一个就**静默加载随机权重**，指标照样算得出来但全是错的。

**替代**：统一用 [`src/eval/model_io.py`](../model_io.py) 的 `load_model_from_ckpt`
（`strict=True` 护栏）。
