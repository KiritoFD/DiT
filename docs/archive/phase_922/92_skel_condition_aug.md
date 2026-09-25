# 922 / 92 — 骨架条件增强（skel condition augmentation）

> 起因：用户提出两个方向——① 弹性形变（Grid Distortion / TPS，模拟同一书家不同次书写的结体波动）
> ② **条件侧扰动**（增强输入条件而非 GT：加噪 / 随机裁剪 / 丢笔画，迫使模型不依赖完美对齐的骨架），
> 并要求「先看历史，确定数据增强（GT 均匀变粗变细，v4）没有好处吗」。
>
> **三条结论先说**：
> 1. 历史那句「数据增广没好处」**不能推出条件侧扰动无效** —— 那次增的是 GT 侧，且用途是类别平衡。
> 2. **「坚决用 3px」这条要改**：当前 g 条件本来就是 ~4px，实测**比 3px 更好**；1px 才是灾难。
> 3. 三个条件增强开关（noise / noise_prob / patch_drop）**从未生效过**（cli.py 没定义 → 配置被静默丢弃）→ **已接线**。

---

## 1. 历史核查：GT 增广到底有没有好处？

### 1.1 记录里写的
| 来源 | 内容 |
|---|---|
| `HISTORY_REVIEW_2026-09-22.md:37` | 数据增广（mid-clean 118,776 = 23,597 真实 + **95,179 增广**）**−0.007**（0.5222 vs 0.5294）|
| `docs/922/10_history_results.md` | 「✗ 数据增广 / 11.9 万增广 / −0.007（增广不增加信息量）」 |

### 1.2 ★ 但那个实验有三个问题

**问题 1：增的是 GT 侧，不是条件侧。**
`tools/aug6.py` Phase A 的算子（读源码确认）：
```
_elastic(img_arr, rng, alpha=uniform(4,8), sigma=uniform(2.5,3.5))   # Simard 式弹性
img2 = img2.rotate(angle, BICUBIC)                                    # + 旋转
```
作用于 `arr`（**GT 图像**）。条件（std 骨架）原样不动。

**问题 2：用途是「类别平衡」，不是「通用正则」。**
`docs/system/05_dataset.md` 原文：**「对组合样本数 <6 的，生成 (6-n) 个增强变体」**
→ 它是为了把稀疏的 (书体, 字, 书家) 组合补到 6 个，**属于数据量问题**；
而条件侧扰动解决的是**「模型对条件过拟合」的正则问题**。两者机制完全不同。

**问题 3：v11 那个 sym 对照是混杂的。**
| 配置 | data_csv | global_batch_size | lr |
|---|---|---|---|
| `v11_pretrain_Sp2_base` | `train_base_noaug.csv` | 192 | 1e-4 |
| `v11_pretrain_Sp2_base_sym` | `train_base_sym.csv` | **128** | **1.5e-4** |

**三个变量同时变**（数据 / batch / lr）→ 无法把差异归因于增广。而且两者停在不同 step（50k vs 120k）。

**问题 4（补刀）**：当前训练集 `train_50k_v2_fixed.csv` 的 `aug` 列**全为空**（50786/50786）——
那批增广样本**根本不在当前流水线里**。

### 1.3 结论
**「数据增广没好处」这个结论证据不足，而且完全不适用于条件侧扰动。**
- GT 侧弹性 + 类别平衡：−0.007，但混杂、且用途不同
- **条件侧弹性 / 加噪 / mask：从未测过**
- GT 侧弹性确实试过（α∈[4,8], σ∈[2.5,3.5]），但只在稀疏组合上

---

## 2. ★ 粗细：实测推翻了「必须用 3px」

### 2.1 当前 g 条件的真实形态
`tools/rebuild_one_shard.py` 给出了唯一权威的编码链路：
```python
im = Image.open(STD_PNG).convert("RGB")      # 直接读 data/50k/std/{id}.png
im = im.resize((256, 256), Image.LANCZOS)
lat = vae.encode(tf(im)[None]).latent_dist.sample() * 0.18215
```
**不 skeletonize、不 binary_dilation。** 所以 g = STKAITI 渲染图的原始笔宽。

**实测（400 张 std png，EDT 法）**：`2*mean(EDT @ 骨架点)` = **4.03px**，`2*max(EDT)` = 4.47px（p50）。

**解码实测（确认 latent 里就是它）**：
| id | 解码墨迹px | 2*max EDT | 原 png 2*max EDT |
|---|---|---|---|
| 0 | 4231 | 5.66 | 5.66 |
| 100 | 1575 | 4.47 | 4.47 |
| 44447 | 1964 | 5.66 | 5.66 |
→ 解码结果与原 png 逐条吻合，**VAE 忠实往返，当前 ~4px 条件没有断裂**。

### 2.2 VAE 往返保真度（我实测，同阈值口径）
| 条件 | 往返 IoU（mean / min，8 张）|
|---|---|
| **orig（现状 ~4px）** | **0.3433 / 0.2752** |
| skel 3px | 0.2514 / 0.1807 |
| skel 1px | **0.0111 / 0.0017** ← 几乎全毁 |

- **1px 确实是灾难**（0.011 ≈ 条件信号在潜空间被打断）——你的判断方向完全正确 ✓
- **但现状 4px 是三者里最好的，改成 3px 反而更差**（0.343 → 0.251）✗

⚠ **绝对值的口径说明**：历史记录的 0.861（1px）/ 0.960（3px）与我这组数不可比
（阈值与度量定义不同）。但**相对排序是同口径的**（同一 binarize + 同一 IoU 定义）。
两处唯一的共识是：**1px 显著劣于 3px**。

### 2.3 结论
**不要改成 3px。** 当前 ~4px 已在安全区且是三档最优；"1px 会断"的隐患**在当前流水线里不存在**
（因为流水线根本没 skeletonize）。真要动粗细，方向应该是 **4px → 5–7px 加粗**（配合 VAE 更稳），
而不是变细到 3px。

---

## 3. ★★ 三个条件增强开关是「死的」—— 已接线

### 3.1 诊断
`src/train/train.py:1795-1806` **早已实现**条件增强：
```python
_gn_scale = float(getattr(args, 'glyph_noise_scale', 0.0))
_gn_prob  = float(getattr(args, 'glyph_noise_prob', 0.0))
_gpd      = float(getattr(args, 'glyph_patch_drop', 0.0))
if 'g' in model_kwargs and (_gn_scale > 0 or _gpd > 0):
    if _gn_scale > 0 and _gn_prob > 0:
        _mask = (torch.rand(N,1,1,1) < _gn_prob)
        _g = _g + _mask * torch.randn_like(_g) * (torch.rand(N,1,1,1) * _gn_scale)
    if _gpd > 0:
        _pmask = (torch.rand(N,1,H,W) > _gpd)
        _g = _g * _pmask
```

**但 `src/train/cli.py` 从未定义** `--glyph-noise-scale` / `--glyph-noise-prob` / `--glyph-patch-drop`。
实测确认：**配置合并只应用 argparse 已知的键**（塞一个 `__probe_new_key__` 进去会被直接丢弃）
→ `getattr(..., 0.0)` 恒拿默认值 → **这三项从未生效过**。

> 同一类 bug 在本项目已出现第 N 次：**产物存在 ≠ 产物被使用**。
> （前有 `inst_skel` 加载条件、`glyph_concat_input` 的配置继承、数据修复的配置继承。）

### 3.2 已修
`_sync_work/patch_cli_glyph_aug.py`（CRLF 安全的字节级插入）在 `--glyph-drop-prob` 之后补了三个参数。
远端已验证：配置里的 `0.3 / 0.5 / 0.2` 现在能穿透到 `args`。备份 `_sync_work/_inj3_bak/cli.py.bak2`。

### 3.3 语义验证（`_sync_work/verify_glyph_aug.py`，逐条断言）
| 性质 | 实测 |
|---|---|
| 形状不变 | ✓ 三种组合都保持 (N,4,32,32) |
| **对称 ± 不漂移均值** | ✓ gn_scale 到 1.0 时均值漂移仅 **+0.00001**（原 −0.00339 → −0.00338）|
| 只有 gn_prob 比例的样本被加噪 | ✓ |
| patch_drop 置零比例 = gpd | ✓ 0.10→0.100 / 0.20→0.202 / 0.50→0.499 |
| 两者叠加 | patch_drop 会盖掉那些格子上的噪声（顺序使然，属预期）|

⚠ **`gn_scale` 与 `gn_prob` 必须同时 > 0** 才加噪（外层 `or`、内层 `and`）——
只设 scale 不设 prob 是静默无效的。

---

## 4. 扰动算子与建议参数

预览图：`_sync_work/skel_aug_preview.png`（3 个字 × 9 种变体，含等效笔宽标注）。
生成脚本 `_sync_work/preview_skel_aug.py`。

| 变体 | 等效笔宽 2*max / 2*mean@骨架 | 说明 |
|---|---|---|
| orig（现状） | 4.5 / 4.03 | 基线 |
| skel 1px | 2.0 / 2.00 | ✗ VAE 打断（往返 0.011）|
| skel 3px | 4.5 / 3.42 | 比现状细 |
| skel 5px | 7.2 / 5.18 | 比现状粗 |
| elastic α8 σ3 | 5.7 / 3.75 | 弱弹性 |
| elastic α16 σ4 | 6.0 / 3.76 | 中弹性 |
| affine 5°/×0.95/shear3° | 4.5 / 3.66 | 结体布局波动 |
| patchdrop 0.2 | 4.5 / 3.63 | Cutout 式局部缺失 |
| regiondrop 48px | 4.5 / 4.03 | 整块缺失（模拟丢笔画）|

### 落地路径（关键：分成「构建期」与「前向期」两层）
| 层 | 内容 | 代价 | 位置 |
|---|---|---|---|
| **构建期（离线）** | 弹性 / 仿射 / 区域丢弃 → 新 PNG → 重编 shards | **6.5 分钟/全量**（已实测，50,786 张）| `build_skel_latents.py` 风格的离线脚本 |
| **前向期（在线）** | 高斯加噪 / latent patch drop / 整图 dropout | **≈0**（已在 train.py 里）| 训练循环 |

- **弹性/仿射必须离线**：潜空间 32×32 太粗（1 patch = 8px），在 latent 上做几何形变 ≈ 手工近似，
  正是本项目已经否定过的路线（PNG 路线的教训）。
- **加噪/patch drop 留在线**：每步随机、代价为 0、且已在代码里（现在已接线）。

### 建议的起点参数（保守，先做单变量）
```
# 前向期（立即可用）
glyph_drop_prob   = 0.15   # 已在 inj3 里验证过
glyph_patch_drop  = 0.10   # Cutout 式，别一上来 0.2
glyph_noise_prob  = 0.30   # 只对 30% 样本加噪
glyph_noise_scale = 0.20   # 潜空间幅度；先小后大
```
⚠ 三个一起开会让"是谁在起作用"无法归因。**建议逐个加**，且每个都跑够到 55k+（见 §5）。

---

## 5. ★ 判据：必须跑到 55k 之后，否则等于没测

这一条是本次会话最重要的教训（详见 `docs/922/frag_backfill/README.md`）：
- E0 的**断笔比是唯一随训练恶化的指标**：15k 触底 1.339 → **100k 达 3.207**（+140%）
- 而 `ssim` / `ink_ssim` 同期**都在涨**，完全看不见笔画碎裂
- **碎裂从 ~55k 开始**；而 E1 停 40k、gate 停 25k、midskel20 停 40k、inj3(旧) 停 26.4k
  → **此前每个实验都停在"连 baseline 都分不开"的区间**

**所以条件增强的判据**：
1. **主**：`frag_ratio`（断笔比）在 **55k–100k** 的表现 —— 它会不会像 E0 一样涨到 3.2？
2. 次：`ink_ssim` 不变差、`strict ssim` 不退化
3. 归因辅助：`t=0.1` 换书家速度场差（注意：该量在 t=0.1 结构性饱和在 0.02，**分辨力为零**，见 `probe_vel/README.md`）

**一个待验证的假设**：条件增强是正则项，若 E0 的碎裂源于"对条件/样本过拟合"，
那么条件增强应当**抑制碎裂**。这是当前最有价值的一个可测预测。

---

## 6. 已改动 / 产物清单

| 文件 | 改动 |
|---|---|
| `src/train/cli.py` | **+9 行**：补 `--glyph-noise-scale` / `--glyph-noise-prob` / `--glyph-patch-drop`（CRLF 安全插入）|
| `_sync_work/patch_cli_glyph_aug.py` | 上述补丁脚本 |
| `_sync_work/verify_glyph_aug.py` | 增强语义验证（对称性/比例/形状）|
| `_sync_work/preview_skel_aug.py` | 扰动预览 + 笔宽实测 |
| `_sync_work/skel_aug_preview.png` | 9 变体 × 3 字预览图 |
| `_sync_work/measure_std_width.py` | std png 笔宽实测 |
| `_sync_work/roundtrip_std.py` | VAE 往返保真度（4px / 3px / 1px）|
| `_sync_work/decode_shard_width.py` | 解码 shard 确认 latent 内容 |

远端备份：`_sync_work/_inj3_bak/cli.py.bak2`

## 附：一个踩了两次的坑
`np.load(npz)` 返回的是**懒加载**对象 —— 循环里写 `z['latents'][j]` 会**每次都解压整个数组**
（51036 次解压，慢到像卡死）。必须先 `L = z['latents']` 缓存。
第一次排查时我误以为是 VAE 加载卡住，实际是这里；
而 VAE `from_pretrained` 不加 `local_files_only=True` 会走**网络回退**（也是 15 分钟级卡顿）。
两个坑都会伪装成"进程卡死"。
