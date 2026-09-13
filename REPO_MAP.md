# REPO_MAP — 仓库结构总览（2026-09-13 整理）

> 原则：**代码 / 配置 / 文档 / 小样本 CSV 入库；权重 / latent / 图片集 / 训练结果一律不入库**
> （见 `.gitignore`，它们只存在于远端 `/root/Workspace/xy/DiT`）。

## 活跃代码
| 目录 | 说明 |
|---|---|
| `src/model/` | DiT 主模型（`dit.py`，含 adaLN/xattn/风格等全部可配置开关）、ControlNet、字嵌入组件 |
| `src/train/` | 训练入口 `train.py`；**当前所有 config 在 `src/train/configs/`** |
| `src/loss/` | FlowMatching / GaussianDiffusion / REPA / losses |
| `src/eval/` | 推理与评测：`inference.py`（采样/指标/eval cache）、`gpu_eval_loop.py` 的底层、`posters.py`、`vae_fast.py`（高性能 VAE encode/decode） |
| `src/utils/` | 数据集（`latent_dataset.py`：latent + aux + skel 条件 + 变体多目录）、采样器、书家词表 |
| `tools/eval/` | 评测运维：`gpu_eval_loop.py`（**评测时暂停训练 SIGSTOP → in-mem → SIGCONT**）、`eval_stdskel_batch.py`（strict/seen 批量）、`eval_ctl.sh`（旧 CPU daemon 控制器） |
| `tools/data/` | 数据构建：latent shards、std 骨架、aux(canny/skel) latent、struct 像素图 |
| `tools/aug/` | 墨迹增强（**现行 = `aug_renders_v4.py` 对称 ±，均值不变**；v1-v3 为历史） |
| `tools/` 其他 | `controlnet/`、`structnet/`、`legacy/`（旧工具） |
| `scripts/ops/` | 运维脚本（launch/stop/status/ad-hoc，含 `_launch_v10b_stdskel_fame3_variant.sh`） |
| `scripts/debug/` | 探针/对比/诊断脚本（一次性） |
| `scripts/build/` | 数据/图表构建脚本（一次性） |
| `gradio_stdskel.py` | 当前模型（std-skel-g, 12ch→取前4ch）的 CPU gradio 前端 |
| `gradio_fame_local.py` | 旧 v8/s21 ControlNet 前端（历史保留） |

## 文档 / 参考
| 目录 | 说明 |
|---|---|
| `docs/system/` | 系统文档 + 实验复盘 + poster 资产（`imgs/`） |
| `docs/` 其他 | 各阶段报告（s6 等） |
| `ref/` | 参考实现（`moyi/` = 12ch 联合目标来源；`net1.pdf` 论文） |

## 归档
| 目录 | 说明 |
|---|---|
| `legacy/dit_core/` | 迁移到 `src/` 之前的旧 DiT 核心（`models.py/train.py/diffusion/` 等，勿引用） |
| `legacy/configs/` | s5–s19 时代旧实验配置 |
| `legacy/scripts/` | 旧时代脚本（数据准备/监控/前端） |
| `legacy/data/` | 旧 CSV / 报告 JSON |
| `legacy/misc/` | 杂项产物（本地保留，不入库） |

## 远端大资产（不入 git，仅在 4090 主机）
- 训练数据：`assets/train_fame3_*.csv`（元数据入库）、`final_imgs_fame_*/`、`assets/fame3-*/`
- VAE latent：`final_latents_fame_*/`、`std_skel*_latents_*/`（g 条件）、`aux_*_latents_*/`（12ch 监督）
- 结果：`assets/results/<run>/`（ckpt + eval_auto json + poster）
- 清单见远端 `ASSETS.md` / `assets/results/MANIFEST.md`（由整理脚本生成）

## 当前推荐流水线（2026-09）
```bash
# 训练（远端 tmux）: 配置在 src/train/configs/，launcher 在 scripts/ops/
bash _sync_work/_launch_v10b_stdskel_fame3_variant.sh <tmux> <config> <results_dir> [resume_ckpt]
# 评测（与训练配合）: 自动暂停训练、GPU in-mem 评测、恢复
python tools/eval/gpu_eval_loop.py --results-dir assets/results/<exp> --device cuda --pause-train
# 前端（CPU, base env）
/opt/conda/bin/python gradio_stdskel.py --device cpu --share
```
