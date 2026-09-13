# scripts/ — 运维 / 调试 / 构建脚本（ad-hoc）

> 这里是**一次性或运维型**脚本；**可复用的正式工具在 `tools/`**。
> 2026-09-13 由 `_sync_work/` + `_ot_scratch/` 的 tracked 脚本归并而来。

| 子目录 | 内容 |
|---|---|
| `ops/` | 启动/停止/状态/传送脚本。**现行启动器**：`_launch_v10b_stdskel_fame3_variant.sh`（训练 + GPU eval 循环） |
| `debug/` | 探针与对比：`probe_*.py`（数据/latent/条件核查）、`_chk_*.py`（运行态检查）、`compare_*/plot_*/validate_*` |
| `build/` | 一次性数据构建：`cpu_encode_v8/v9.py`、`gpu_encode_v8.py`、`build_v8_dashboard.py`、`make_v8_grid.py` |

注意：
- 这些脚本多数**硬编码远端路径**（`/root/Workspace/xy/DiT`），在远端运行。
- 长期复用请迁到 `tools/` 并参数化。
- 远端仍存在同名 `_sync_work/`（未纳入 git），运行中的 tmux 引用的是远端路径；本地这批是归档副本。
