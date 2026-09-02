# tools/ 脚本地图

> 原则：**少、精、鲁棒**。一次性探索脚本归档在 `legacy/`（git 历史可查），
> 生产脚本按层归入子目录。远程运行时以 `/root/Workspace/xy/DiT/tools/` 同构。

## data/ — 数据集构建与骨架/latent 派生

| 脚本 | 用途 |
|---|---|
| `build_fame_fast.py` | fame 数据集快速构建（CSV + GPU encode，v8 前身） |
| `flip_all_fame.py` | 全量极性归一（ink>0.5 翻转，mp64） |
| `build_std_glyph_latents.py` | 标准字库渲染 + VAE encode（v1 骨架库来源） |
| `build_std_skel_latents.py` / `build_std_skel1_latents.py` | 标准字库骨架 latents |
| `build_skel_latents.py` | GT 图 → 1px/3px 骨架 PNG + latents（断点续跑） |
| `build_fame_skel1px.py` / `build_std_skel1_latents.py` | fame 1px 骨架派生 |
| `build_std_glyph_bbox.py` | 字库 bbox 统计 |
| `charsets/` | 字表（8105 规范字 / mid_clean 分书体字表） |

## eval/ — 评测与出图

| 脚本 | 用途 |
|---|---|
| `eval_unified.py` | 多实验统一口径评测（n=500, cfg 可配, 分书体） |
| `build_eval_strict_midclean.py` / `_top6.py` | 严格凸包 eval 集构建 |
| `build_eval500.py` | 旧口径 eval500 |
| `make_ctrl_posters.py` | ctrl/base 对照 poster 生成 |
| `print_eval_trend.py` / `build_results_csv.py` | 指标汇总 |
| `plot_experiment_axes.py` / `plot_all_experiments.py` | 可视化（docs/system/imgs/） |

## clean/ — 数据清洗与归档

| 脚本 | 用途 |
|---|---|
| `noise_clean_v2.py` | **v7 去噪算法**（Otsu 滞后二值化 + 切边条 + 去小件，6 轮迭代收敛） |
| `denoise_probe.py` | 去噪探测（before/after 网格出图） |
| `scan_image_pollution.py` | 全量污染扫描（连通域统计） |
| `cleanup_old_experiments.py` | 旧实验 ckpt 收敛（best-2）+ 归档 |

## serve/ — 前端

| 脚本 | 用途 |
|---|---|
| `gradio_fame_local.py`（根目录另有一份运行副本） | fame 书法生成 Gradio（本地 GPU，share 公网） |

## src/eval/ — 常驻服务（非 tools）

| 模块 | 用途 |
|---|---|
| `universal_metrics_daemon.py` | 递归扫描全部系列的 eval 指标 daemon（ctrl/pretrain 双格式） |
| `eval_supervisor.sh`（根目录） | daemon 崩溃自动重启（tmux `eval_supervisor`） |

## legacy/ — 一次性/历史脚本归档

早期探索、像素时代管线、单次修复脚本。git 历史可查，不再维护。
根目录 `_*.py` 会话脚本同样归档于此。
