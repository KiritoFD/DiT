# `tools/ablation/` — OT × 数据增强 消融工具链

以 **v68** 为 base（`DiT-2Cond-Sp/2` 59.17M，lr 5e-5，cosine，`w_repa=0.03`，
真值条件表，drop 0.08/0.08/0.08(+all 0.05)，`eval200fix` n=187），
把 **OT 策略** 与 **数据增强方式** 两个轴做成单变量消融，全部对齐 v68 的
**总样本预算 200,000 × 288 = 57,600,000**（换 batch 只改步数）。

> 设计原则：**形态学增强才允许复用 REPA 缓存**。
> `build_aug_variants.py` 只做笔画粗细（二值膨胀/腐蚀），字形/位置/书家/书体不变，
> DINOv2 patch-token 基本不变，于是增强行的 REPA 目标可复用源图特征
> （判据见 `tools/dino_expand_top10_aug.py`）。
> ⚠ 几何类增强（仿射/弹性/旋转/裁剪）会改变 DINO 特征，**不能**复用；本工具链不提供。

---

## 1. 工具

| 文件 | 作用 |
|:--|:--|
| `build_aug_variants.py` | 通用形态学增强构建器：图像 + 全景 CSV + `manifest.json`（记录 tag→id 基址） |
| `expand_repa_cache.py` | 把 `data/dino_cache/top10_v1` 扩到任意增强全集（增强行复用源图特征） |
| `make_ablation_configs.py` | 以 v68 为 base 批量生成 `src/train/configs/ablation/*.json` |
| `prepare_data.py` | 编排：noaug CSV → 增强图像 → REPA 缓存 → latent shards（幂等） |
| `launch_ablations.sh` | 单卡顺序跑臂（tmux 后台 / `status` 查看） |

配套改动：`tools/encode_aug_latents.py` 新增 `--encode-mode {mode,sample}`
（**Calli-VAE 必须 `sample`**；SD-VAE 用 `mode`）。

### 增强策略（`--strategy`）

| 策略 | 定义 | 倍数 | 复用现有数据? |
|:--|:--|:--|:--|
| `sym` | 历史 v4：`tp`/`tn`，± 对同 `p∈{1,2}`，8-邻域 | 3x | ✅ 48 上已有图像+CSV+shards+cache |
| `sym5x` | `tp1/tn1/tp2/tn2`（v69 约定，id 7.0–7.3M） | 5x | ⚠ 48 上 CSV 在、**图像缺失** |
| `thick` | 只变粗 `tp1(p=1)/tp2(p=2)` | 3x | ❌ 需新建 |
| `thin` | 只变细 `tn1/tn2` | 3x | ❌ 需新建 |
| `symwide` | ±`p=1,2,3` 六变体 | 7x | ❌ 需新建 |
| `sym4` | 4-邻域十字结构元 ±1px | 3x | ❌ 需新建 |

保护逻辑与 v4 一致（腐蚀面积 <15% 原墨迹或 <20px 降档；膨胀面积 >3.0× 降档）。
`img_id` 分块：`sym` 7.0/7.1M、`sym5x` 7.0–7.3M、`thick` 8.0/8.1M、`thin` 8.2/8.3M、
`symwide` 8.4–8.9M、`sym4` 9.0/9.1M（各留 100k，源 id 最大 38,583）。

---

## 2. 臂矩阵（`make_ablation_configs.py` 默认全集）

| 臂 | 数据增强 | OT | `data_csv` | shards（latent=calli） |
|:--|:--|:--|:--|:--|
| `noaug_c2ot` | 无（仅原图） | C2OT(slot) | `train_top10_noaug.csv`\* | `shards_img_aug_calli`\* |
| `sym3x_noOT` | 3x 对称 | **不开** | `train_top10_aug_sym.csv` | `shards_img_aug_calli` |
| `sym3x_naiveOT` | 3x 对称 | **朴素 OT** | `train_top10_aug_sym.csv` | `shards_img_aug_calli` |
| `noaug_noOT` | 无 | 不开 | `train_top10_noaug.csv`\* | `shards_img_aug_calli`\* |
| `thick_c2ot` | 只变粗 3x | C2OT(slot) | `train_top10_aug_thick.csv` | `shards_img_aug_thick_calli` |
| `thin_c2ot` | 只变细 3x | C2OT(slot) | `train_top10_aug_thin.csv` | `shards_img_aug_thin_calli` |
| `symwide_c2ot` | ±1/2/3px 7x | C2OT(slot) | `train_top10_aug_symwide.csv` | `shards_img_aug_symwide_calli` |
| `sym4_c2ot` | 4-邻域 3x | C2OT(slot) | `train_top10_aug_sym4.csv` | `shards_img_aug_sym4_calli` |

\* `noaug` 复用 3x 的 shards 与 DINO 缓存（其中含原图行），只需一张**过滤 CSV**
（`prepare_data.py` 步骤 A 自动生成），无需重编码。

参考：`v68` 自身 = `sym` + C2OT（= 上表 `sym3x_noOT` 加上 OT），已在 `assets/results/v68_aug_sp_c2ot/`。
`v71` = v68 但换 Calli-VAE latent（**旧 `mode()` 编码约定**，已被 2026-10-09 的根因分析判定为错误）。

---

## 3. 数据准备

```bash
# 只看计划
python tools/ablation/prepare_data.py --latent calli --dry-run

# 建新增强策略的数据 + REPA 缓存 + latent shards（幂等）
python tools/ablation/prepare_data.py --latent calli --strategies thick,thin,symwide,sym4

# 用修正约定重编 3x 基线 shards（48 上现有 calli shards 是旧 mode() 约定，必须重编）
python tools/ablation/prepare_data.py --latent calli --strategies "" --reencode-baseline
```

- 编码 VAE：`--vae`（calli 默认 `data/pretrained/pretrained_models/calli_vae` 即 v2 修正版）。
- 编码约定：calli → `--encode-mode sample`；sd → `mode`。
- 产物：`exp-std/data/shards_img_aug_<s>[_calli]/shard_*.npz`（含 `names` 键，逐行硬校验）。

## 4. 生成配置并开跑

```bash
python tools/ablation/make_ablation_configs.py --latent calli --batch 560
# 先标定 batch: 目标 ~40 GB 显存。Sp/2 在 48 的 48GB 卡上估 batch ≈ 544–576
# （v68: 24G 卡 batch 288 用 21.96 GiB -> 48G 卡按比例 ~576；留余量取 560）
ARMS="noaug_c2ot sym3x_noOT sym3x_naiveOT" bash tools/ablation/launch_ablations.sh run
bash tools/ablation/launch_ablations.sh status
```

- 每臂 `steps = 57,600,000 / batch`（batch 560 → 102,857 步）。
- 成本：约 **~15 h/臂**（单卡 4090，compute-bound，总样本量决定）。
- 8 臂顺序 ≈ 5 天；建议先跑 3–4 个核心臂。

## 5. 开跑前的三处校验

1. `expand_repa_cache.py` 的 `miss=0 xcheck_fail=0`（增强行 `src_image_path` 反解一致）。
2. `encode_aug_latents.py` 的 `img_id 无重复`、`CSV 行数 == 落盘行数`。
3. 训练启动后看日志的 `decode 约定自检` 与 `reserved` 显存是否 ~40 GB。
