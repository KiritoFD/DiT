# 代码分层与兼容层

## 1. 分层原则

用户要求：**核心代码在 `src/` 下按 `model / loss / train / eval / utils` 分层**；`tools/` 只放杂项脚本与数据处理；**ControlNet 相关代码不得在 `tools/` 下**。

```
src/
├── model/          # 模型定义：dit.py, controlnet.py, lora.py
├── loss/           # 扩散/损失：gaussian_diffusion.py, flow_matching.py,
│                   #            respace.py, timestep_sampler.py,
│                   #            diffusion_utils.py, losses.py
├── train/          # 训练入口：train.py, train_controlnet.py, configs/
├── eval/           # 评测：inference.py（核心）+ 薄壳 + CPU daemon
└── utils/          # 数据/工具：dataset.py, latent_dataset.py, samplers.py,
                    #            glyph_latent.py, latent_structure.py, download.py
```

**根目录 = 薄 shim 兼容层**：历史脚本（40+ 个）以 `from models import ...`、`from diffusion import ...`、`from losses import ...` 等形式导入的旧路径全部保留，每个根文件只剩一两行转发：

```python
# models.py
"""Backward-compat shim: 模型代码已迁移至 src.model (见 src/model/dit.py)."""
from src.model import *  # noqa: F401,F403
```

```python
# diffusion/__init__.py  （另有 gaussian_diffusion.py / flow_matching.py /
#                          respace.py / timestep_sampler.py / diffusion_utils.py 子模块转发）
from src.loss import *
from src.loss import gaussian_diffusion, flow_matching, respace, timestep_sampler, diffusion_utils, losses
```

```python
# train.py（根）
"""训练入口 (launcher): 实际实现移至 src/train/train.py."""
from src.train.train import main_from_cli
if __name__ == "__main__":
    main_from_cli()
```

**约定**：新代码一律 import `src.*`；旧脚本不需要改就能跑。若某种 `import *` 无法带出下划线符号（如 `_extract_into_tensor`），在 shim 里显式补一行导出。

## 2. src/ 各目录职责

### src/model/
| 文件 | 内容 |
|---|---|
| `dit.py` | DiT_2Cond（二因子条件）、DiT_3Cond（三因子）、DiTBlock、TimestepEmbedder、ConditionEmbedder、`DiT_2Cond_models` / `DiT_3Cond_models` 工厂字典、`forward_with_cfg`（CFG 空间处理） |
| `controlnet.py` | ControlNetDiT（12 层 ctrl 分支，逐层 zero-init 注入）、`load_main_model`（加载已训练主模型）、ZeroConv / zero_init_linear |
| `lora.py` | LoRA 注入工具 |

### src/loss/
| 文件 | 内容 |
|---|---|
| `gaussian_diffusion.py` | 经典 DDPM（epsilon 预测、learned sigma、DDIM 采样、`sample_t` 返回 `randint(0, num_timesteps)`） |
| `flow_matching.py` | FlowMatching：直线插值 `x_t=(1-t)x0+t·noise`、velocity 目标 `v=noise-x0`、`TIME_SCALE=1000.0`、Euler 采样（t: 1→0）、`sample_t` 返回 `rand([0,1))` |
| `respace.py` | SpacedDiffusion（DDIM 时间步抽取） |
| `timestep_sampler.py` | 时间步重采样策略（历史 API） |
| `diffusion_utils.py` | 公用张量工具（含 `_extract_into_tensor`） |
| `losses.py` | 结构损失家族：StructDecoder、LatentStructLoss、EdgeGradientLoss、SkeletonLoss、REPALoss |
| `__init__.py` | `create_diffusion`（DDPM 工厂）、`create_flow_matching`、`create_diffusion_or_flow(diffusion_type)` 开关工厂 |

### src/train/
| 文件 | 内容 |
|---|---|
| `train.py` | 主模型预训练/微调：`main(args)` + `main_from_cli(argv=None)`；配置默认值、4-way dropout 参数、EMA、in-process GPU eval |
| `train_controlnet.py` | ControlNet 训练（warm-start / from-scratch 两模式），统一 `sample_t` |
| `configs/` | 训练配置 json（ctrl_skel*.json、calligraphers.json、char_meta.json、supported_chars.txt） |

### src/eval/
| 文件 | 内容 |
|---|---|
| `inference.py` | **评测唯一核心**：VAE 单例、采样（bf16）、decode（fp32）、PNG 落盘、CPU 指标、eval cache、pair eval、pending marker |
| `in_process_eval.py` | 主模型训练内 GPU eval 薄壳（旧签名保留） |
| `in_process_ctrl_eval.py` | ControlNet 训练内 GPU eval 薄壳（旧签名保留） |
| `eval_ctrl_metrics_daemon.py` | ControlNet 的 CPU 指标 daemon（读 step_tag） |
| `eval_metrics_daemon.py` | 主模型的 CPU 指标 daemon |
| `auto_eval_cpu.py` / `auto_eval_ctrl.py` / `auto_eval_ctrl_flow.py` / `auto_eval_gpu.py` / `eval_auto.py` / `eval_gen.py` / `gpu_batch_eval*.py` / `eval_metrics.py` / `eval_models.py` / `eval_test.py` / `eval_full_3cond.py` / `eval_compose.py` / `eval_controlnet_cpu.py` / `sample_controlnet.py` / `test_controlnet.py` / `gradio_controlnet.py` / `backfill_eval.py` / `latent_condition_probe.py` | 各类评测/采样壳与工具 |

### src/utils/
| 文件 | 内容 |
|---|---|
| `dataset.py` | 像素级数据集（MCCD 图像 + canny/skel） |
| `latent_dataset.py` | latent shard 数据集（预编码 latent + 条件 id），parse csv 的 `img_id` |
| `samplers.py` | factor_balanced 平衡采样器（char/callig 温度逆频） |
| `glyph_latent.py` | 标准字形 latent 工具 |
| `latent_structure.py` | 结构探针工具 |
| `download.py` | 模型下载 |

## 3. tools/（杂项与数据，禁止放核心代码）

- `aug6.py` —— mid-clean 数据流水线（Phase A 增广 / Phase B 编码 / Phase C 合并），见 `05_dataset.md`。
- `remote_sync/` —— 遗留同步脚本（原样保留，未动）。
- 其他：数据准备（prepare_mccd_dataset*.py、preprocess_256.py、resize_256.py）、评测构建（build_eval_strict_top6.py、build_eval500.py）、分析（analyze_*.py、cfg_sweep_ctrl.py）、dashboard（index.html、chart.umd.min.js）等。

> ⚠️ `tools/` 里**没有** controlnet 相关文件；ControlNet 的实现/训练/评测全部在 `src/{model,train,eval}`。

## 4. 根目录其他入口

| 文件 | 说明 |
|---|---|
| `make_latent_shards.py` | 把 329,715 张 latent 打包成 `final_latents/shard_XXXXX.npz`（每 5000 张，`latents` f16 + `img_ids` int32） |
| `sample.py` / `sample_3cond.py` / `gradio_app.py` / `flask_app.py` | 采样 / 演示入口 |
| `eval_*`、`in_process_*`、`auto_eval_ctrl_flow.py`、`eval_ctrl_metrics_daemon.py` 等根文件 | 全部是 `from src.X import *` 薄壳 |
| `download_vae*.py`、`vae_hash.py` | VAE 工具 |
| `configs/` | 冒烟测试等补充配置 |
| `_smoke_midclean.py`、`_verify_gpu_e2e.py`、`_verify_midclean.py` | 一次性验证脚本（`_` 前缀，待清理或移入 archive） |

## 5. archive/ 与 docs/

- `archive/diag_backup_20260828/` —— 126+ 个诊断脚本（`_*.py`）归档，不再参与 py_compile 检查。
- `docs/system/` —— **本目录（当前权威文档）**；`docs/legacy/`、`docs/model/`、`docs/training/`、`docs/data/`、`docs/eval/` 为重构前的历史文档。

## 6. 边界与铁律

1. 时间步采样：**只有 `diffusion.sample_t(n, device)` 一个入口**，调用方绝不自行分支（flow/ddpm 错配是历史最大 bug，见 `02_diffusion.md`）。
2. 调用方不区分 flow/ddpm 的采样器：统一 `ddim_sample_loop(..., clip_denoised=False)`，由 diffusion 对象内部决定 Euler / DDIM。
3. 新模块一律进 `src/` 对应子包；根目录只允许 shim。
4. 训练/推理的评测只允许通过 `src.eval.inference` 的核心函数，壳脚本不得复制采样循环。