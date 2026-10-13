# src/eval/legacy — 已停用的评测代码

2026-09-17 整理。**`src/eval/` 现在只剩 5 个核心文件**：

| 文件 | 职责 |
|---|---|
| `inference.py` | 采样 / VAE decode / 落盘（核心库，被 40+ 处引用） |
| `metrics.py` | 指标唯一实现（MSE / SSIM / skel_iou / LPIPS） |
| `model_io.py` | 模型重建 / ckpt 加载（`strict=True` 唯一入口） |
| `in_mem_eval.py` | **在训**评测（`--in-mem-eval`，含 `maybe_run_in_training()`） |
| `batch_eval.py` | **事后**批量评测（`--eval-mode deferred` 的配套） |

## 归档清单与理由

### ① 独立进程评测架构（daemon / worker）—— 已被 in-mem 取代

| 文件 | 说明 |
|---|---|
| `cpu_eval_daemon.py` / `cpu_eval_worker.py` | CPU 评测守护进程 + worker。训练与评测分进程、靠 `eval_pending_*.json` 通信 |
| `eval_metrics_daemon.py` / `eval_ctrl_metrics_daemon.py` | 指标守护进程（后者属 ControlNet 线） |
| `universal_metrics_daemon.py` | 上面两个的合并版 |
| `auto_eval_cpu.py` / `auto_eval_gpu.py` | 自动评测入口 |
| `eval_auto.py` | 被 `auto_eval_cpu` 当库用 |

**为什么停用**：`--in-mem-eval` 在同进程用常驻 EMA 采样，不需要跨进程通信、
不需要重新加载 ckpt，也不会出现"daemon 落后于训练"的竞态。

### ② ControlNet 线（整体废弃）

`auto_eval_ctrl.py` / `auto_eval_ctrl_flow.py` / `eval_ctrl_ckpt.py` /
`eval_controlnet_cpu.py` / `in_process_ctrl_eval.py` / `gradio_controlnet.py` /
`sample_controlnet.py` / `test_controlnet.py` / `gpu_batch_eval*.py`

**为什么停用**：ControlNet 双臂方案已废弃，模型实现也归档到 `src/model/legacy/`。

### ③ in-process GPU eval（0 配置启用）

`in_process_eval.py`

**为什么停用**：由 `--auto-eval` 门控，而**全仓 0 个配置把它设为 true**（70 个显式 false）。
train.py 里的对应分支已删除。

### ④ 其它一次性工具 / 死代码

| 文件 | 说明 |
|---|---|
| `loop.py` | 只被 `tools/eval/gpu_eval_loop.py`（独立实验脚本）用 |
| `gpu_ablate_eval.py` | 消融评测，只被 `tools/eval/e4a/e4b` 用 |
| `posters.py` | 拼图海报，只被 `cpu_eval_daemon`（已归档）用 |
| `eval_facade.py` | 只被 `train_controlnet` / `train_skel_1cond`（已归档）用 |
| `eval_compose.py` / `eval_gen.py` / `eval_full_3cond.py` / `eval_metrics.py` | 0 引用 |
| `backfill_eval.py` | 0 引用 |
| `metrics_png.py` / `eval_models.py` / `latent_condition_probe.py` | 0 引用 |

## 共用问题（也是归档的核心理由）

这些文件里的**模型重建大多用 `load_state_dict(strict=False)`**，
字段漏一个就**静默加载随机权重**，指标照样算得出来但全是错的。
现在统一走 [`src/eval/model_io.py`](../model_io.py) 的 `strict=True` 入口
（见 [docs/system/70](../../../docs/system/70_code_review_refactor.md) §1.1）。

## 另有：从 eval 挪走的

| 原路径 | 新路径 | 理由 |
|---|---|---|
| `eval/vae_fast.py` | **`utils/vae_fast.py`** | SD-VAE 高性能 encode/decode 轮子，与评测无关 |
| `eval/cpu_sampler.py` | **`utils/cpu_sampler.py`** | CPU Heun 采样器，通用 |
| `eval/std_dino_embedder.py` | **删除** | 与 `model/std_dino_embedder.py` **逐字节相同**的重复文件 |
