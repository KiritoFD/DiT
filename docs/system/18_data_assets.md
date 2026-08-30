# 数据资产清单（2026-08-30）

> 自动生成：`_scan_assets.py` → `5script/data_assets.csv`（78 行）
> 覆盖：数据集 CSV、latent/骨架目录、标准字形库、预训练权重。

---

## 1. 数据集 CSV

### 1.1 全量与主力数据集

| 文件 | 行数 | 书家 | 字 | 书体 | 备注 |
|---|---:|---:|---:|---:|---|
| `final_train.csv` | 318,715 | **1,853** | 7,765 | 12 | 全量训练集 |
| `train.csv` | 298,281 | 1,850 | 7,765 | 12 | 全量（旧） |
| `5script/train.csv` | 147,841 | 1,004 | 7,011 | 5 | |
| `val.csv` / `test.csv` | 37,285 / 37,286 | 883 / 882 | 6,503 / 6,529 | 12 | |
| `final_test.csv` | 10,000 | 532 | 4,092 | 12 | |
| `kailishu_train.csv` | 51,098 | 447 | 4,933 | 2 | 楷书单书体 |
| **`5script/train_fame.csv`** | **51,322** | **44** | **4,765** | **7** | **当前主力（fame）** |
| `5script/train_fame_3script.csv` | 28,385 | 41 | 4,039 | 3 | fame 三书体子集 |
| `5script/train_top30.csv` | 128,842 | 108 | 6,952 | 5 | |
| `5script/train_top6.csv` | 10,866 | **11** | 3,154 | 2 | 最小集，指标虚高 |

### 1.2 评测集

| 文件 | 行数 | 书家 | 字 | 书体 | 用途 |
|---|---:|---:|---:|---:|---|
| **`5script/eval_fame_strict.csv`** | **500** | 42 | 432 | 6 | **当前主力评测** |
| `5script/eval100_top.csv` | 100 | 46 | 97 | 5 | |
| `5script/eval100_top30.csv` | 100 | 42 | 97 | 5 | |
| `5script/eval100_top6.csv` | 100 | 11 | 99 | 2 | |
| `5script/eval_strict_top6.csv` | 271 | 11 | 263 | 2 | |
| `5script/eval_unseen_top6.csv` | 255 | 11 | 242 | 2 | **未见字评测** |
| `final_eval.csv` | 1,000 | 202 | 839 | 12 | |
| `5script/eval500_clean.csv` | 455 | 122 | 421 | 5 | |
| `overfit_500.csv` | 500 | 151 | 455 | 11 | 过拟合测试 |

### 1.3 fame 数据集的组合覆盖（关键）

```
书家 44   字 4,765   样本 51,322
理论组合 209,660   实际组合 36,512   覆盖率 17.41%
每字平均 10.77 个样本
仅出现 1 次的字       784 (16.5%)
仅 1 位书家写过的字   836 (17.5%)
eval 字未见   0/500      ← 100% 见过
eval 书家未见 0/500      ← 100% 见过
组合泄漏     0          ← 组合泛化，成立
```

**含义**：当前 fame 评测测的是**组合泛化**（新「书家×字」配对），
而非**未见字泛化**。每个 eval 字在训练集里平均被 ~17 位书家写过，
模型对该字的形态是充分见过的。

**未见字能力完全未被评估** —— 这是当前评测体系的最大盲区。
`eval_unseen_top6.csv`（255 条）是现成的未见字评测集，但 fame 线未使用。

---

## 2. latent / 骨架目录

| 目录 | 类别 | 文件数 | 大小 | latent 形状 | 说明 |
|---|---|---:|---:|---|---|
| `final_imgs_256` | 图片 | 329,715 | — | — | 256px RGB，平铺 `{id}.png` |
| `final_latents_fame` | latent | 20 | 358.7 MB | (N,4,32,32) f16 | fame 图 latent |
| `final_latents_f4` | latent | 26 | 3,020.7 MB | (N,4,32,32) | 全量 kl-f4，**仅 128,842 个 id** |
| `final_latents_mid_clean` | latent | 25 | 807.9 MB | (N,4,32,32) | midclean |
| `final_latents` | latent | 66 | 2,577.2 MB | (N,4,32,32) | 早期 |
| `final_skel3_fame` | 骨架 PNG | 51,822 | — | — | 3px 膨胀骨架 |
| **`final_skel1_fame`** | 骨架 PNG | **51,822** | — | — | **1px 细骨架（新）** |
| `final_skel_latents_fame` | 骨架 latent | 20 | 339.3 MB | (N,4,32,32) f16 | 3px |
| **`final_skel_latents_fame_1px`** | 骨架 latent | **21** | **668.7 MB** | (N,4,32,32) f16 | **1px（新）** |

**注意**：
- `final_latents_f4` 只有 128,842 个 id，**不覆盖全量 329,715 张图**。
  需要新 latent 时必须现编码，不能假设已存在。
- 1px 骨架目录是本次新建的（51,822 张，覆盖 train+eval 全量，无 stale 缺口）。

---

## 3. 标准字形库

| 目录 | 状态 | 字数 | 覆盖书体 | fame 命中率 |
|---|---|---:|---|---:|
| `src/utils/std_glyph_latent` (v1) | **目录不存在** | — | 楷(0)、隶(4) | **0.0%** |
| `src/utils/std_glyph_latent_v2` (v2) | 存在 | 43,755 | 楷(0)、行(3)、隶(4)，6 字体 | **53.1%** |

### 3.1 v2 对 fame 的分书体覆盖（`_check_glyph_lib.py` 实测）

| 书体 | 样本 | v1 命中 | v1% | v2 命中 | v2% |
|---|---:|---:|---:|---:|---:|
| 行 | 14,393 | 0 | **0.0** | 9,370 | 65.1 |
| 楷 | 11,921 | 0 | **0.0** | 7,469 | 62.7 |
| 草 | 10,910 | 0 | **0.0** | 0 | **0.0** |
| 六体 | 9,049 | 0 | **0.0** | 9,048 | 100.0 |
| 篆 | 2,977 | 0 | **0.0** | 0 | **0.0** |
| 隶 | 2,071 | 0 | **0.0** | 1,360 | 65.7 |
| **合计** | **51,322** | **0** | **0.0** | **27,248** | **53.1** |

**两个要点**：
1. v1 命中率 **0.0%** —— 不是覆盖不全，是**完全失效**（目录不存在）。
   而 `latent_dataset.py` 接线的正是 v1，且缺失时静默返回零张量 →
   `w_glyph_cond` 启用后条件全程为零（见 15 文档 P0）。
2. 草 10,910 样本、篆 2,977 样本**零覆盖**（无标准字体）。
   六体的 100% 是「借用其他书体字形」的语义，不是真正的六体字形。

---

## 4. 预训练权重与 embedding

| 文件 | 大小 | 说明 |
|---|---:|---|
| `pretrained_models/dino_embeddings/glyph_dino_embeddings_384.npy` | — | 35,130×384 字形表 |
| `pretrained_models/dino_embeddings/glyph_dino_index.json` | — | glyph → (script_id, char_id) |
| `pretrained_models/dinov2_vits14_pretrain.safetensors` | — | DINOv2 ViT-S/14（本地） |
| `pretrained_models/sd-vae-ft-ema` | — | kl-f4 VAE |

**DINO embedding 覆盖率**：20,468 / 35,130 行有真实 DINO 值，其余填均值。
有效秩 34.1/384（见 14 文档）。

---

## 5. 数据管道要点

### 5.1 img_id 是唯一键

所有 latent / 骨架 / 图片通过 `img_id` 关联：

```
train.csv 的 image_path（如 final_images/12345.png）
  → 正则提取数字 → img_id = 12345
  → final_imgs_256/12345.png
  → final_latents_fame/shard_*.npz 里的 img_ids
  → final_skel1_fame/12345.png
```

### 5.2 已知的 stale 陷阱

- **eval 骨架缺失**：fame 的 skel latent 最初只覆盖 train，eval 侧后来单独补
  （曾因 `/tmp` 路径导致 stale bug，见 13 文档 §5）。
  新脚本 `tools/build_fame_skel1px.py` 已改为**一次性覆盖 train+eval**，
  并自带覆盖率自检。
- **shard 区间重叠**：`build_skel_latents.py` 按「区间命名」写 shard
  （`shard_{first}_{last}.npz`），若分两次跑指向同一输出目录，
  第二批的 id 区间会覆盖第一批 → 静默丢失。
  **必须一次跑完全量**。

### 5.3 生成工具

| 工具 | 用途 |
|---|---|
| `tools/build_fame_fast.py` | fame CSV + 图像 latent |
| `tools/build_fame_dataset.py` | fame 全流程（含 3px 骨架） |
| `tools/build_skel_latents.py` | 通用骨架 1px/3px + latent（已加 `--latent-src`） |
| **`tools/build_fame_skel1px.py`** | **fame 1px 骨架 + GPU 编码（新）** |

---

## 6. 复现

```bash
python _scan_assets.py     # 重新清点 -> 5script/data_assets.csv
python _check_glyph_lib.py # v1/v2 对 fame 的覆盖率实测
python _probe_fame.py      # fame 组合覆盖密度
```
