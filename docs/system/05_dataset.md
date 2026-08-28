# 数据系统与 mid-clean 流水线

> 对应源码：`tools/aug6.py`（增广/编码/合并）、`make_latent_shards.py`（原始 latent shard）、`src/utils/latent_dataset.py`（训练读取）、`src/utils/dataset.py`（像素级读取）、`tools/build_eval_strict_top6.py`（评测表）。
> 数据只存在于远程 `/root/Workspace/xy/DiT`；本地 git 忽略数据目录。

## 1. 原始数据（MCCD）

- 官方中文书法数据集：**329,715 张** 256×256 图，含 **楷 / 行 / 隶**（另有草/篆用于其他实验）等多书体。
- 每张图带：`script`（书体）、`character`（字）、`calligrapher`（书家）。
- 条件 id 编码：**`glyph_id = script_id * 7026 + character_id`**（每 script 7026 个字符槽位，见 `tools/remote_sync/_add_glyph_col.py`）；DINO 字形索引 `glyph_dino_index.json` 同样用 `[script_id, character_id]` 编码 —— 两处必须一致（曾误读导致假告警，已核对 100% 覆盖）。

## 2. latent 缓存（原始）

- `make_latent_shards.py`：把 329,715 个 latent 按 img_id 排序打包为 `final_latents/shard_XXXXX.npz`（每 5000 张）：`{"latents": (N,4,32,32) f16, "img_ids": (N,) int32}`。
- 另有 `final_latents_mid_clean/`（25 shards，118,776 latents，mid-clean 专用，见 §4）。
- 图片根：`final_imgs_256/{img_id}.png`；骨架：`final_skeleton_d3/{img_id}.png`（3px 骨架）；VAE：`pretrained_models/sd-vae-ft-ema`（sd-vae-ft-ema，scaling 0.18215，f8 32×32 latent）。

## 3. 清洗原则（train_3top30_common）

- 按 **GB2312 一级 + 二级汉字** 过滤（去掉生僻字/异体字）。
- 各 (script, char, calli) 组合应至少有若干样本（top30 常见字 → 之、也、人 …）。
- 输出 `5script/train_3top30_common.csv`（**23,597 行**）作为 mid-clean 的「干净原样本」。
- 评测表 `5script/eval_strict_top6.csv`（271 行：楷 169 + 隶 102）从严格同分布组合抽取，`eval100_top30_clean.csv` 等为派生表。

## 4. mid-clean 增广流水线（tools/aug6.py）

目标：**每个 (script, char, calli) 组合恰好 6 个样本**，抹平长尾，同时保持类别计数可预测

> 增广后：每组合 6 样本 ⇒ `字符频次 = 6 × 书家数`；`书家频次 = 6 × 字符数`。5461 字符（稀疏难点）与 67 书家（充足易学）的分布差异被量化，供 dropout 配比决策（`03_model.md` §2.1）。

| Phase | 执行 | 内容 |
|---|---|---|
| **A. 增广（CPU 多进程）** | `--phase aug` | 对组合样本数 <6 的，生成 `(6-n)` 个增强变体（随机仿射/弹性等）→ `final_imgs_mid_clean/{new_id}.png`（**new_id ∈ 1000000–1095178**，与原始 img_id 3073–325944 空间分离）；写 `aug_meta.csv`（new_id, script, char, calli, calli_id, char_id, glyph_id, script_id）。产出 **95,179 张**。 |
| **B. VAE 编码（GPU）** | `--phase encode` | 读 `final_imgs_mid_clean/*.png` → sd-vae-ft-ema 编码（fp32，scaling=0.18215）→ `/tmp/mid_clean_tmp/` 20 shards + aug_meta.csv。 |
| **C. 合并（CPU）** | `--phase merge` | ① 从**原始** latent shards 里只保留 train_3top30_common.csv 出现过的 img_id（23,597 张，先过滤再合并，避免混入脏样本）；② 追加 B 的增广 shard → `final_latents_mid_clean/`（**25 shards / 118,776 latents**）；③ 写 `5script/train_mid_clean.csv`（**118,776 行**）：原行 + 增广行，`image_path = final_imgs_mid_clean/{id}.png`（原样样本 image_path 仍指向 final_imgs_256）。 |

命令形态：`python tools/aug6.py --phase all [--csv ... --out-imgs ...]`；`--phase` 可 `all|aug|encode|merge`（断点续跑用）。

### 4.1 最终数据事实（已校验）

| 项 | 值 |
|---|---|
| train_mid_clean.csv 行数 | 118,776（楷 47,976 / 行 45,804 / 隶 24,996） |
| 唯一 glyph 数 | **5,461** |
| 唯一书家数 | **67** |
| latent shards | final_latents_mid_clean/ 25 个（118,776 latents，f16 (4,32,32) + img_ids int32） |
| 缺失图片数 | 0（CSV 全部可解析，`re.search(r"(\d+)\.png")` 取 img_id） |
| DINO 索引覆盖 | 100%（20468 glyph 条目，glyph_id=sid*7026+cid 与 mid-clean 全对齐） |
| 每组合样本 | 恰 6 |

## 5. 训练读取（src/utils/latent_dataset.py）

```python
MCCDLatentDataset(csv_file=..., latent_shards_dir=..., img_root=...,
                  skel_root=..., load_canny=..., load_skel=...,
                  preload=True, num_preload_workers=..., structure_size=256)
```

- 按 CSV 行找 img_id → 查 shard（latent 预加载到 RAM，`preload=true` 时多进程并行）→ 返回 `latent / y_callig / y_char / (g) / (skeleton) / (canny)`。
- CSV 行解析：`image_path` 正则提取数字 id；`glyph_id` 优先，回退 `character_id`。
- 平衡采样（`src/utils/samplers.py`）：`factor_balanced` 用 **温度逆频** 对 (char, callig) 两个频率维度取权重（`balance_char_alpha=0.35` / `balance_callig_alpha=0.15`）。

## 6. 其他数据工具（tools/）

- `build_eval_strict_top6.py`：评测表（271 行）—— 与训练表严格同分布（top6 组合）抽取，保证 eval 对训练分布的代表性。
- `gen_skel_d3.py` / `gen_canny_skel.py`：骨架/边缘图生成（3px 骨架用于 ControlNet）。
- `prepare_mccd_dataset*.py`、`preprocess_256.py`、`resize_256.py`：MCCD 原始整理与预处理。
- `verify_dataset_quality.py`：全量质量检查。
- `dino_extract.log` / `remote_check_dino_cover.py`：DINO 字形嵌入提取与覆盖核对。