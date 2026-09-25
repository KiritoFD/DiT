# MCCD 数据集说明（dataset.md）

> 本文档描述**当前生效的数据组织与使用方法，以远程服务器为权威**。
> 远程主机：`root@10.176.54.17:36430`，项目目录 `/root/Workspace/xy/DiT`。
> 最后核对：2026-08-14。

---

## 1. 数据集来源与规模

官方数据 `MCCD_Character`：共 **329,715 张**书法字图，label 与图片一一对应，图片间**互不重合**。

| 项 | 值 |
|---|---|
| 官方 train 图片 | 234,255（`trainset_dataset/train/`） |
| 官方 test 图片 | 95,460（`trainset_dataset/test/`） |
| **_id_maps 清洗后映射** | 书家 1873（含 `others`） / 字体 12 / 汉字 7765 |
| `labels/*.json`（旧派生） | 书家 2243 / 字体 12 / 汉字 7765 |
| 远程图片派生 | `final_images` / `final_imgs_256` / `final_canny` / `final_skeleton` 各 **329,715 张** |
| 远程 latent 预编码 | `final_latents/` 共 **66 个 shard**（`shard_00000.npz` ~ `shard_00065.npz`） |

官方 label 文件名格式（`train_label.txt` / `test_label.txt`）：

```
字-字体-朝代-书家/出处-样本id.png        # 5 段
（少数 6 段，最后一段前为碑帖名；书家字段 = parts[3]）
```

示例：`㐁-印-宋-广韵-161207.png` → 字=`㐁`、字体=`印`、朝代=`宋`、书家=`广韵`、样本=`161207`。

---

## 2. 当前数据切分（新方案：图片级不重合）

由 `final_manifest_split.json`（329,715 行）定义的稳定切分：

| split | 数量 | 对应 CSV（远程） |
|---|---|---|
| train | **318,715** | `final_train.csv` |
| test | **10,000** | `final_test.csv` |
| eval | **1,000** | `final_eval.csv` |

- `img_id` = 全局连续编号 `0..329714`，与 `final_images|imgs_256|canny|skeleton/<img_id>.png` 直接对应。
- 远程另有 `final_train_small.csv`（2,000 行），仅用于调试/冒烟。
- 三份 CSV 行数（含表头）：`final_train.csv = 318,716`、`final_test.csv = 10,001`、`final_eval.csv = 1,001`。

### manifest 示例（每行字段）

```json
{
  "img_id": 0,
  "split": "train",
  "char_id": 2175,
  "script_id": 2,
  "calli_id": 12,
  "orig_path": "train/㐁/㐁-印-宋-广韵-161207.png",
  "orig_char": "㐁",
  "orig_script": "印",
  "orig_calli": "广韵",
  "orig_calli_raw": "广韵",
  "orig_seq": "161207",
  "final_split": "train"
}
```

`final_split` 字段与 `split` 在官方文件夹之外几乎一致（train 多并入 84,460 张官方 test 图）。

---

## 3. 数据目录（远程，权威）

当前**唯一生效**的数据文件（新方案）：

```
/root/Workspace/xy/DiT/
├── final_images/            原图缩放 256x256 RGB（329,715）  ← csv 中 image_path 指向
├── final_imgs_256/          256x256 RGB（329,715）           ← 训练 --img-root 默认使用
├── final_canny/             256x256 灰度（329,715）          ← canny 条件图
├── final_skeleton/          256x256 灰度（329,715）          ← 骨架条件图
├── final_latents/           66 个 shard_*.npz（VAE 预编码 latent）
│     每个 shard 含：img_ids / latents 两键
├── final_train.csv          318,715 条
├── final_test.csv           10,000 条
├── final_eval.csv           1,000 条
└── final_train_small.csv    2,000 条（调试用）
```

### CSV 格式（新）

```csv
image_path,calligrapher,script,character,calligrapher_id,script_id,character_id
final_images/0.png,广韵,印,㐁,12,2,2175
...
```

| 列 | 说明 |
|---|---|
| `image_path` | `final_images/<img_id>.png`，仅用于解析 `img_id` |
| `calligrapher` / `script` / `character` | 中文名（纯展示） |
| `calligrapher_id` / `script_id` / `character_id` | 训练用整数 id（**3 个条件标签**） |

> 注意：CSV **不含** `canny_path`/`skeleton_path` 列。canny/skeleton 由 `img_id` 从
> `final_canny/`、`final_skeleton/` 按文件名 `{img_id}.png` 直接定位。

---

## 4. 训练用哪个图片目录？

- 训练入口默认 `--img-root final_imgs_256`（`train.py` 的该参数默认值，本地与远程代码一致）。
- `final_images/` 与 `final_imgs_256/` 均为 256x256，`--img-root` 决定实际读取哪个目录加载 **gt 原始图**（像素 loss 用）。
- 当前远程配置（`train_full_3cond_skel0.json`）即使用默认 `final_imgs_256`，**无需在配置里显式指定**。

### latent 读取路径

`latent_dataset.py` 的 `MCCDLatentDataset`：

1. 从 CSV `image_path` 用正则提取 `img_id`；
2. latent：查 shard 索引 → `np.load(shard)["latents"][j]`（fp32 tensor）；
3. image：`os.path.join(img_root, f"{img_id}.png")` → [0,255] → 归一化到 [-1,1]；
4. canny（可选）：`os.path.join(canny_root, f"{img_id}.png")` → 二值 [0,1]；
5. 返回 `latent / image / canny / y_callig / y_script / y_char`。

若 `--latent-shards-dir` 为空则回退到 `MCCDDataset`（解码原图 + canny/skeleton 路径模式，旧方案）。

---

## 5. 使用方式

### 5.1 训练（远程训练用 latent 缓存）

```bash
# 放到远程项目目录后执行
torchrun --nproc_per_node=1 train.py --config train_full_3cond_skel0.json
```

本地等价参数（本地代码与远程同步）：
`--data-csv`, `--data-dir`, `--img-root`(默认 `final_imgs_256`),
`--canny-root`, `--latent-shards-dir`, `--num-calligraphers`, `--num-scripts`, `--num-characters`,
`--image-size 256`, `--eval-csv`, `--eval-n`, `--global-batch-size`, `--ckpt-every`。

### 5.2 配置文件中数据相关字段（训练 JSON）

```json
{
  "use_canny": true,
  "use_skel": true,
  "data_csv": "final_train.csv",
  "eval_csv": "final_test.csv",
  "img_root": "final_imgs_256",
  "canny_root": "final_canny",
  "skel_root": "final_skeleton",
  "latent_shards_dir": "final_latents",
  "num_calligraphers": 1873,
  "num_scripts": 12,
  "num_characters": 7765
}
```

> 注意：旧配置里 `num_calligraphers=2021` 属于早期 VOCAB，**必须与 `_id_maps.json` 一致改为 1873**。

### 5.3 数据构建脚本

| 脚本 | 作用 |
|---|---|
| `build_id_maps.py` | 解析官方 label → `_id_maps.json`（书家 1873 / 字体 12 / 字 7765，脏值并入 `others`） |
| `gen_split_csv.py` | 由 `final_manifest_split.json` 生成 `final_train.csv / final_test.csv / final_eval.csv` |
| `final_manifest_split.json` | 329,715 行的最终切分 manifest（字段见 §2） |

### 5.4 同步（本地 ↔ 远程）

本地仅需保留只读脚本与文档，**数据一律以远程为准**：

```powershell
# 拉取新 csv 到本地
scp -P 36430 "root@10.176.54.17:/root/Workspace/xy/DiT/final_train.csv" .
scp -P 36430 "root@10.176.54.17:/root/Workspace/xy/DiT/final_test.csv" .
scp -P 36430 "root@10.176.54.17:/root/Workspace/xy/DiT/final_eval.csv" .

# 本地旧的 0 行占位 csv 已废弃，删除：
Remove-Item train.csv,test.csv,val.csv,_remote_train.csv,_remote_test.csv,_remote_val.csv
```

---

## 6. 已废弃的旧数据（8:1:1，不再使用）

以下为上一版 80/10/10 随机切分方案产物，**已被新方案取代，应删除**（远程）：

| 远程路径 | 说明 | 规模 |
|---|---|---|
| `dataset/` | 旧 images/canny/skeleton/latents 及 `canny_raw_bak`、`eval_small` | ~24 GB |
| `dataset.zip` | 旧数据打包 | ~13 GB |
| `train.csv` / `test.csv` / `val.csv` | 旧 8:1:1 切分（298,281 / 37,286 / 37,285） | 共 ~372,855 行 |

本地对应副本：`_remote_train.csv / _remote_test.csv / _remote_val.csv`（旧切分的本地拷贝）以及
0 行占位 `train.csv / test.csv / val.csv`。

**删除命令**（远程，确认后执行）：

```bash
cd /root/Workspace/xy/DiT
rm -rf dataset dataset.zip train.csv test.csv val.csv
```

---

## 7. 可写字范围（回答"训好能写哪些字"）

条件标签来自两套映射，**训练/评估/采样必须锁定同一套**：

- 新映射（推荐，与当前数据一致）：`_id_maps.json` → 书家 **1873** / 字体 **12** / 汉字 **7765**。
- 旧派生：`labels/calligrapher_to_id.json`(2243) / `labels/script_to_id.json`(12) / `labels/character_to_id.json`(7765)。

即模型可生成上述映射中全部 **7,765 个汉字**（含生僻字、扩展区字），条件是用户提供
（书家, 字体, 字）三元组且三者都落在所用映射内。