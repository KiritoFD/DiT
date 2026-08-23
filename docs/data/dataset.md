# 数据管线文档 (dataset.md)

> DiT 书法生成 — 数据管线全景：原始图像 → VAE 编码 → latent 分片 → 数据集加载。
> 本文档以远程服务器 `/root/Workspace/xy/DiT/` 为权威路径，本地仓库仅保留只读脚本与文档。
> 编码/验证命令详见 [tools/vae/DATA_PIPELINE.md](../../tools/vae/DATA_PIPELINE.md)。

---

## 1. 原始数据 (Raw Data)

### 1.1 MCCD 数据集规模

| 项 | 值 |
|---|---|
| 数据集 | MCCD (Chinese Calligraphy Dataset) |
| 图片总数 | **329,715** 张 |
| 图像规格 | 256×256 RGB PNG |
| 图像间关系 | 互不重合，label 与图片一一对应 |
| 元数据 | `final_manifest.json`（img_id → metadata） |

### 1.2 图片目录

| 目录 | 内容 | 备注 |
|------|------|------|
| `final_images/` | 329,715 张 256×256 RGB PNG | CSV `image_path` 指向此处 |
| `final_imgs_256/` | 329,715 张 256×256 RGB PNG | 与 `final_images/` **md5 完全一致**（镜像副本） |

- `final_images/` 与 `final_imgs_256/` 内容完全相同（md5 校验通过），训练脚本默认 `--img-root` 二选一即可。
- 采样 500 张检查：**96.8% 为 256×256，~3% 为非标尺寸** → VAE encode 阶段强制 `resize((256, 256))` 保证输入一致。

### 1.3 结构条件图

| 目录 | 内容 | 用途 |
|------|------|------|
| `final_skeleton/` | 329,715 张 256×256 骨架图 | skeleton 结构条件 |
| `final_skeleton_d3/` | 329,715 张骨架图（膨胀 3px） | dilated skeleton 条件 |
| `final_canny/` | 329,715 张 256×256 canny 边缘图 | canny 结构条件 |

> `final_skeleton_d3/` 是 `final_skeleton/` 的 3px 形态学膨胀版本，用于更宽松的结构监督。

### 1.4 标签映射

| 文件 | 说明 |
|------|------|
| `labels/calligrapher_to_id.json` | 书家名 → 整数 id |
| `labels/character_to_id.json` | 汉字 → 整数 id |
| `labels/script_to_id.json` | 字体/书体 → 整数 id |
| `labels/final_id_maps.json` | 综合映射：书家 1873（含 `others`）/ 字体 12 / 汉字 7765 |

> 旧派生 `labels/*.json` 与新 `_id_maps.json`（`final_id_maps.json`）规模略有差异；
> 训练/评估/采样**必须锁定同一套映射**，推荐使用 `final_id_maps.json`。

---

## 2. 数据切分 (Data Splits)

所有 CSV 位于 `5script/` 目录，由 `final_manifest_split.json` 稳定切分派生。

| CSV | 行数 | 书家 | 字 | 用途 |
|-----|------|------|------|------|
| `5script/train_top6.csv` | 10,866 | 6（top） | 3,154 | s6 实验（小规模） |
| `5script/train_top30.csv` | 128,842 | 30（top） | 6,952 | **s7 当前训练（大规模）** |
| `5script/train.csv` | ~329k | 全部 | 全部 | 完整数据集 |
| `5script/eval100_top6.csv` | 100 | — | — | top6 评测集 |
| `5script/eval100_top30.csv` | 100 | — | — | top30 评测集 |
| `5script/eval_unseen_top6.csv` | — | — | — | 未见书家评测（泛化） |
| `5script/eval500_top6.csv` | 500 | — | — | 较大评测集 |
| `5script/seen5_top30.csv` | — | — | — | seen-calligrapher 评测子集 |

### CSV 字段格式

```csv
image_path,calligrapher,script,character,calligrapher_id,script_id,character_id,glyph_id
final_images/190.png,鏅烘案,妤?銞?377,0,18,18
```

| 列 | 说明 |
|---|---|
| `image_path` | `final_images/<img_id>.png`，**仅用于正则解析 `img_id`** |
| `calligrapher` / `script` / `character` | 中文名（纯展示，不参与训练） |
| `calligrapher_id` / `script_id` / `character_id` | 训练用整数 id（3 个条件标签） |
| `glyph_id` | 标准字形 id（可选，用于 glyph 条件查询；缺省回退 `character_id`） |

> CSV **不含** `canny_path` / `skeleton_path` 列。canny/skeleton 由 `img_id` 按文件名 `{img_id}.png` 直接定位。

---

## 3. VAE Latent 编码管线

### 3.1 两个 VAE 方案

项目支持两套 VAE，对应不同下采样因子 `f` 与 latent 通道数：

| VAE | 路径 | f | latent_ch | latent_shape | scaling_factor | shards |
|-----|------|---|-----------|-------------|----------------|--------|
| sd-vae-ft-ema | `pretrained_models/sd-vae-ft-ema` | 8 | 4 | (4, 32, 32) | 0.18215 | 66 shards × 5000 |
| **kl-f4** | `pretrained_models/kl-f4` | 4 | 3 | (3, 64, 64) | 0.102079 | 26 shards × 5008 |

- **sd-vae-ft-ema (f8)**：SD 官方 VAE，83.7M 参数，4 通道 32×32 latent，scaling 用 SD 论文经验值 0.18215。
- **kl-f4 (f4)**：从 ldm 转换（`convert_klf4.py`，204 个 key remap），55.3M 参数，3 通道 64×64 latent，scaling 由样本估计。

> 当前 s7 训练采用 **kl-f4**（重建质量更高、latent 分辨率更高、参数更少）。

### 3.2 编码流程 (`tools/vae/encode_latents_klf4.py`)

对 CSV 中每张图执行以下步骤：

1. **加载 VAE**：`AutoencoderKL.from_pretrained(--vae).eval()`，冻结全部参数。
2. **读图**：`Image.open(path).convert("RGB").resize((256, 256))`（强制 resize 处理 ~3% 非标尺寸）。
3. **归一化**：`arr / 127.5 - 1.0` → [-1, 1]，permute 到 `(C, 256, 256)`。
4. **编码**：`z = vae.encode(x).latent_dist.sample()`（采样而非取 mean，保留随机性）。
5. **缩放**：`z = z * scaling_factor`（存入 latent 分布的缩放空间）。
6. **存盘**：fp16 写入 `shard_XXXXX.npz`：
   - `latents`: `(N, C, H, W)` float16
   - `img_ids`: `(N,)` int64（对应 `final_images/{img_id}.png`）

```python
# 核心编码逻辑（摘自 encode_latents_klf4.py）
z = vae.encode(t).latent_dist.sample().mul_(args.scaling_factor)
lats = z.cpu().float().numpy().astype(np.float16)
np.savez(path, latents=np.stack(shard_lats).astype(np.float16),
         img_ids=np.array(shard_ids, dtype=np.int64))
```

> 编码命令与验证命令详见 [tools/vae/DATA_PIPELINE.md](../../tools/vae/DATA_PIPELINE.md)。

### 3.3 Shard 格式

```
shard_XXXXX.npz:
  latents: (shard_size, C, H, W) float16   # scaled latent (z * scaling_factor)
  img_ids: (shard_size,) int64             # 对应 final_images/{id}.png
```

- **kl-f4**：`(5008, 3, 64, 64)` fp16，每张 24,576 bytes；最后一个 shard 可能 < 5000（128,842 = 25×5000 + 3,842）。
- **sd-vae**：`(5000, 4, 32, 32)` fp16，每张 8,192 bytes。

### 3.4 VAE 对比 — 重建底噪 (floor noise)

floor noise = encode→decode 重建误差（无扩散），衡量 VAE 本身的失真下界：

| VAE | Floor MSE | Floor SSIM | Samples | Source |
|-----|-----------|------------|---------|--------|
| sd-vae-ft-ema (f8) | 0.003660 | 0.9655 | 100 | remote verify, stored latent round-trip |
| **kl-f4 (f4)** | **0.001910** | **0.9882** | 100 | remote verify, stored latent round-trip |

- **kl-f4 floor MSE 是 sd-vae 的 ~52%**（0.0019 vs 0.0037），SSIM 接近无损（0.988）。
- 换 kl-f4 后 eval MSE 的天花板（VAE 底噪）降低约一半；但 latent 分布完全不同，DiT 需从头训练，eval MSE 不可跨 VAE 直接比较。

### 3.5 Scaling Factor 验证（全量 128,842 latents）

由 `verify_latents_f4.py` / `_verify_sdvae.py` 对存储 latent 做全量统计：

| VAE | scaling_factor (使用) | latent mean | latent std | 1/std (理论) |
|-----|--------------------|------------|------------|------------|
| sd-vae-ft-ema | 0.18215 | 0.3178 | 1.1305 | 0.885 |
| kl-f4 | 0.102079 | -0.0559 | 0.9838 | 1.016 |

> **kl-f4 std ≈ 0.98，接近 1.0**，证明 scaling factor 校准良好。
> sd-vae 的 0.18215 并非 `1/std(1.1305)=0.885`，而是 SD 原始论文的经验值。
> latent 在 encode 后乘 scaling_factor 存储，decode 时除回。

---

## 4. Latent 数据集加载

### 4.1 `latent_dataset.py` / `src/latent_dataset.py` — `MCCDLatentDataset`

**latent 缓存数据集**：从预构建 shard 加载 pre-encoded latent，跳过 on-the-fly VAE encode。

| 特性 | 说明 |
|------|------|
| 自动检测 latent 形状 | 从第一个 shard 探测 `latent_channels`、`latent_spatial`（支持 f8=4ch/32×32、f4=3ch/64×64 等） |
| 全量预加载 | `preload=True` 时一次性读入 RAM：5.9G（128k f4）/ 2G（f8），训练过程零磁盘 IO |
| 返回内容 | pre-encoded latent + 条件标签 + 可选 image/canny/skel |
| 无条件字形 | `g_t = torch.zeros(latent_channels, latent_spatial, latent_spatial)`（glyph 缺失时保 collate 一致） |

**`__getitem__` 返回字典**：

| 键 | 类型 | 说明 |
|----|------|------|
| `latent` | `(C, H, W)` float32 | scaled latent（已乘 scaling_factor） |
| `image` | `(3, 256, 256)` float32 | 原图归一化到 [-1,1]（可选） |
| `canny` | `(1, S, S)` float32 | 二值 [0,1]，`structure_size` 控制 256 或 32（可选） |
| `skeleton` | `(1, S, S)` float32 | 二值 [0,1]（可选） |
| `y_callig` | long | 书家 id |
| `y_script` | long | 字体 id |
| `y_char` | long | 字 id（`glyph_id` 优先，缺省回退 `character_id`） |
| `g` | `(C, H, W)` float32 | 标准字形 latent（glyph 条件），缺失给零 |

**构造参数要点**：

```python
MCCDLatentDataset(
    csv_file,              # 5script/train_top30.csv 等
    latent_shards_dir,     # final_latents_f4 / final_latents
    img_root,              # final_images / final_imgs_256
    canny_root=None,       # final_canny
    skel_root=None,        # final_skeleton_d3
    load_canny=False,      # 是否加载 canny 条件
    load_skel=False,       # 是否加载 skeleton 条件
    load_image=True,       # 是否加载原图（REPA 等需要 GT 图时）
    preload=False,         # 是否全量预加载到 RAM
    structure_size=256,    # 条件图分辨率（256 或 32，32=8×8 max-pool 降采样）
    use_glyph_cond=False,  # 是否附加标准字形 latent g
)
```

> **非预加载路径**：`_get_latent(img_id)` 按 `_id_to_shard` 索引打开单个 shard，故意保持无状态，避免 DataLoader workers 保留过多解压 shard。
> **预加载路径**：按 shard 分组、每个 shard 只 `np.load` 一次再 scatter 到 RAM，配合多进程并行 PNG 解码（`num_preload_workers`）。

### 4.2 `dataset.py` / `src/dataset.py` — `MCCDDataset`

**像素空间数据集**：on-the-fly VAE encode，当 latent 缓存不可用时回退使用。

| 特性 | 说明 |
|------|------|
| 加载内容 | image + canny + skeleton maps |
| 变换 | BICUBIC resize + `Normalize(0.5, 0.5)`（RGB）；NEAREST resize（二值图） |
| 数据增强 | `is_train=True` 时 `RandomAffine(degrees=2, translate=(0.02,0.02), scale=(0.95,1.05))` |
| 空间对齐 | 同一随机仿射参数同时应用于 image 和 canny/skeleton，保证结构监督像素对齐 |
| 无水平翻转 | **中文汉字禁止水平翻转**（字形不可镜像） |

> 仿射参数每个 item 现抽现用：image 用 BILINEAR、canny/skeleton 用 NEAREST，确保二值边缘不模糊。

---

## 5. 目录结构 (远程 `/root/Workspace/xy/DiT/`)

```
final_images/          329,715 PNG (256×256 RGB)
final_imgs_256/        identical to final_images
final_latents/         66 shards, sd-vae f8, (5000, 4, 32, 32) fp16
final_latents_f4/      26 shards, kl-f4 f4, (5008, 3, 64, 64) fp16
final_skeleton/        skeleton maps
final_skeleton_d3/     dilated skeleton maps (3px)
final_canny/           canny edge maps
pretrained_models/
├── sd-vae-ft-ema/     f8 VAE (83.7M params)
└── kl-f4/             f4 VAE (55.3M params, converted from ldm)
5script/
├── train_top6.csv
├── train_top30.csv
├── eval100_top6.csv
├── eval100_top30.csv
└── seen5_top30.csv
labels/
├── calligrapher_to_id.json
├── character_to_id.json
├── script_to_id.json
└── final_id_maps.json
```

---

## 6. VAE 工具 (`tools/vae/`)

| 文件 | 用途 |
|------|------|
| `convert_klf4.py` | ldm → diffusers `AutoencoderKL` key remap（204 keys） |
| `convert_grayscale_vae.py` | 单通道外科手术（conv_in 求和 + conv_out 取均值） |
| `estimate_scaling_factor.py` | N-sample scaling factor 估计 |
| `encode_latents_klf4.py` | 全量编码 → `shard_XXXXX.npz` |
| `verify_latents_f4.py` | 编码后验证：latent 统计 + 重建 floor noise |
| `benchmark_vae.py` | 4-VAE 对比（MSE / SSIM / latent 统计） |
| `train_vae.py` | VAE 微调 / 从头训练（plan B/C） |

> 完整编码/验证命令、benchmark 结果、训练运行状态见 [tools/vae/DATA_PIPELINE.md](../../tools/vae/DATA_PIPELINE.md)。

---

## 7. 可写字范围（"训好后能写哪些字"）

条件标签来自两套映射，**训练/评估/采样必须锁定同一套**：

- **新映射（推荐）**：`labels/final_id_maps.json` → 书家 **1,873**（含 `others`）/ 字体 **12** / 汉字 **7,765**。
- 旧派生：`labels/calligrapher_to_id.json`(2,243) / `labels/script_to_id.json`(12) / `labels/character_to_id.json`(7,655)。

即模型可生成上述映射中全部 **7,765 个汉字**（含生僻字、扩展区字），条件是用户提供（书家, 字体, 字）三元组且三者都落在所用映射内。
