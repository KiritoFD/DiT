# 远程数据说明

> 远程主机: `ssh 4090` (root@10.176.54.17:36430)
> 工作目录: `/root/Workspace/xy/DiT`

## 核心数据集

### 真迹数据 (fame)
| 路径 | 内容 | 大小 |
|---|---|---|
| `5script/train_fame.csv` | 训练集索引 (51,322 样本) | - |
| `5script/eval_fame_strict.csv` | 严格评估集 (500 样本) | - |
| `final_imgs_256/` | 真迹图 256×256 | ~2GB |
| `final_latents_fame/` | 真迹 VAE latent 分片 | ~1.5GB |

### 骨架数据 (skel)
| 路径 | 内容 | 大小 |
|---|---|---|
| `final_skeleton_d3/` | GT 骨架 PNG (3px 白底黑线) | 1.3GB, 329,715 张 |
| `final_skel_latents_fame/` | GT 骨架 VAE latent | 340MB |
| `final_skel_latents_fame_std/` | **标准字库骨架 VAE latent** | 62MB |
| `std_skeleton_d3/` | 标准字库骨架 PNG | 7926 张 (kai 4521 + li 3402) |

### 标准字形库
| 路径 | 内容 | 大小 |
|---|---|---|
| `std_glyph_latent_v2/` | 标准字形 VAE latent (6 字体) | ~500MB |
| `std_glyph_latent_v2/coverage_report.json` | 覆盖率报告 | - |

### IDS 数据
| 路径 | 内容 | 大小 |
|---|---|---|
| `_sync_work/cjkvi-ids/cjkvi-ids-master/ids.txt` | IDS 字典 (cjkvi) | 2.1MB |

## 预训练模型
| 路径 | 内容 |
|---|---|
| `pretrained_models/sd-vae-ft-ema/` | VAE (SD 1.5 fine-tuned) |
| `pretrained_models/dino_embeddings/` | DINO 字嵌入 (glyph 级 384d, 字级 768d) |

## 实验结果
| 路径 | 内容 |
|---|---|
| `5script/results/s25_ids_pretrain/` | s25 IDS 预训练 (进行中) |
| `5script/results/s20_ctrl_skel_flow_v2/` | s20 ControlNet GT skel |

## 数据流

```
真迹图 (final_imgs_256)
    ↓ VAE encode
真迹 latent (final_latents_fame)
    ↓ 训练
s25 IDS 预训练模型
    ↓ 后训练 (ControlNet)
s26 GT skel / s27 标准字库 skel
```

## 关键脚本

### 数据构建
- `tools/build_skel_latents.py` — GT 骨架 → latent
- `tools/build_std_skel_latents.py` — 标准字库骨架 → latent
- `tools/build_std_glyph_latents.py` — 标准字形 → latent

### 训练
- `src/train/train.py` — 主训练脚本
- `src/train/configs/s25_ids_pretrain.json` — IDS 预训练配置
- `src/train/configs/s26_ctrl_gt_skel.json` — GT skel ControlNet
- `src/train/configs/s27_ctrl_std_skel.json` — 标准字库 skel ControlNet

### 远程操作
- `_sync_work/_btn.py` — SSH 班车 (up/env/run/poll/pull)
- `_sync_work/_monitor_s25.py` — s25 监控
- `_sync_work/_launch_s26_s27.py` — s26/s27 启动

## 注意事项

1. **Python 路径**: 远程用 `/opt/conda/bin/python`（不是系统 python3）
2. **PYTHONPATH**: 训练前需 `export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH`
3. **GPU 显存**: 23.5GB，s25 训练占 ~20GB，不能同时跑其他 GPU 任务
4. **标准字库 skel 覆盖**: 只有 kai/li 两个书体（7926 字），缺 cao/zhuan/xing
