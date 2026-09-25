# 922 / 95 — 模块取舍裁决与 v19 计划

> 来源：用户 2026-09-25 的明确裁决。这是**项目当前唯一有效的模块清单**，
> 任何新实验的默认配置都应以此为准。

## 主线保留（默认开启）

| 模块 | 说明 |
|---|---|
| Flow Matching | 训练目标与采样 |
| REPA | `w_repa=0.03`，layer 8，DINOv2-S/14 缓存 |
| 标准骨架 g | `skel_as_glyph_cond=True`，`skel_latent_shards_dir` 单目录（无扰动变体）|
| `glyph_embedder` | `glyph_embedder_depth=2` |
| **4-layer ZeroAdaLN injection** | `glyph_inject_layers=4`，`glyph_inject_mode=adaln` |
| `glyph_vec_cond` | `glyph_vec_dim=128`，`pool=mean` |
| `factorized_cat` | `condition_fusion` |
| 45-callig `LabelEmbedder` | `num_calligraphers=45` |

## 主线关闭（不再启用）

`script_id` 输入 / `StyleHierarchy` / `ScriptGlyphFiLM` / flat 87-pair map /
`MultiStyleEmbedder` / `CalligStyleCrossAttn` / style token context / `callig_spatial` /
`style_ln` / `style_ada_rank` / `SpatialStyleFiLM` / `LowRankSpatialStyleFiLM` / Legacy LocalCA

对应配置键：`hier_style=0`、`num_pairs=0`、`num_scripts=0`、`script_film=False`、
`pair_residual=0`、`lowrank_spatial_rank=0`、`spatial_film_rank=0`、
`local_ca_impl` 不再用 `legacy`。

## 隔离待验（默认关闭，需要时单独开关）

| 模块 | 状态 | 理由 |
|---|---|---|
| **GlyphQuery** | 默认关闭，**最有设计价值** | 唯一有清晰设计论证的模块（QK-Norm 挡爆炸，实测风格放大 1e4 倍 logit 仍 2.25）。v19 正在单独验它。 |
| `glyph_concat_input` | 默认关闭 | **没有干净收益证据** |
| `glyph_drop` / geometry augmentation | 默认关闭 | 不是无效，而是**"seen 拟合" 与 "frag 稳定性" 的取舍** |
| `style-rank loss` | 默认关闭 | **当前结果不能采信**（见下）|

## style-rank 为什么不能采信

`docs/922/94` 里报的 "cal_enrich 1.11× → 1.66×，+50%" **已撤回**。配对检验（`tools/style_rank_paired.py`）：

| 统计量 | 覆盖列 | 均值差 | SE | t |
|---|---|---|---|---|
| `tgt_spec`（inj3-aug → v18 @100k）| **218** | +0.00131 | 0.00153 | **+0.86** |
| `tgt_spec`（v18 @100k → @125k）| 218 | +0.00010 | 0.00087 | +0.12 |
| `tgt_spec`（v18 @125k → E0）| 218 | −0.00108 | 0.00200 | −0.54 |
| `margin`（inj3-aug → v18 @100k）| 40 | −0.00059 | 0.00376 | −0.16 |
| `margin`（v18 @125k → E0）| 40 | +0.01678 | 0.00857 | **+1.96**（E0 更好）|

- 二值命中率（McNemar）：inj3-aug 7/41 → v18 9/41，**只翻了 2 列**，p=0.50。
- **最好 power 的检验只给 t=0.86** → 效应在噪声水平。
- 更专门的 `margin` 甚至倾向 **E0 更好**（t=1.96）。
- 注意 `margin` 均值为**负**（v18 −0.046 / E0 −0.030）：平均而言生成结果离**别的**书家更近。

**根因：`cal_enrich` 的有效列只有 41**（249 列里只有 16.5% 在训练集里有同书家候选），
而留出池总共只有 250 个样本，**扩不出来** → 必须重新划分数据。

## v18 的另一条重要观察：它也在退化

v18 停在 145k，最后一次 eval：
`seen ssim=0.6098 frag=3.236 | strict ssim=0.5537 frag=1.989`

- **seen frag 从 1.885(@100k) 涨到 3.236(@145k)** —— 和 E0 一样会爆，只是晚一些
- seen ssim 0.6098 仍低于 E0@100k 的 0.6612
→ 说明扰动带来的 frag 稳定性**不是永久的**，只是推迟了爆炸。

## v19 计划（用户指定，按顺序执行）

### 第一步：恢复基线
**注意：用户后来澄清"不需要重做 E0"** —— 所以第一步改为直接做 **GQ+E0**（见下），
用现有 E0 曲线作参照。

### 第一步（修正版）：GQ + E0
配置 `src/train/configs/v19_gq_e0_100k.json`：
**严格 E0 + GlyphQuery@2,6，其余全部关闭**（含扰动、concat、style-rank）。
- 数据沿用 E0 的 `train_50k_v2.csv` —— 让"与 E0 曲线的差异"唯一归因于 GlyphQuery
- `max_steps=100k`（E0 的 40k 看不到 frag 的 60k–100k 判别区）
- 2026-09-25 16:52 起跑，实测 `local_ca=2@[2,6]`、新增参数 1383.9K、GPU 100%/370W

### 第二步：扰动消融（待第一步确认 seen 后）
按顺序，每步都看 **seen / strict / frag / ink_ssim**：
```
baseline  ->  +glyph_drop  ->  +geometry aug  ->  +both
```

### 第三步：建立新的风格评测集
- 现有 `cal_enrich` 只有 41 有效列 → 需要 **900+ 有效列**的新评测集
- **不应阻塞第一步**：
  - seen 恢复 → 用现有 seen 曲线诊断
  - 风格是否有效 → 用新评测集
  - frag 是否改善 → 用 60k–100k 区间判断
- 约束：留出池只有 250 个样本 → 必须**重新划分**（从训练集再拿出 ~1500），
  代价是从今往后的 run 都要在缩小的训练集上训

## 代码清理

`src/loss/style_rank_module.py` 删掉两处死代码（`_enc_force_eval` 空方法、
`with torch.no_grad(): pass` 空块），2532 → 2355 字节。
⚠ **未删除任何模块级代码** —— 删模块会破坏旧 ckpt 加载（`local_ca_impl` 那次已经踩过）。
