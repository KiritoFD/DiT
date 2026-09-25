# 63. 下一步实验计划（v12+）：探针阶梯 + 开关矩阵

> 日期：2026-09-16
> 相关：[62](62_param_budget_derivation.md)（参数量推演）、[60](60_what_is_actually_useless.md)、[59](59_12ch_and_white_zero_retrospective.md)
>
> 本文是"下一批实验怎么做"的可执行清单：**已落地的配置** + **可用开关矩阵** + **判据**。

---

## 0. 当前状态

- **v12 在跑**（S/2 + `factorized_cat` + `glyph_vec_cond`，从零训练，step ~25k/400k）
- v11（M/2）已停，best strict ckpt 存于 `assets/results/v11_M432_best_strict_152500.pt`
- **过拟合已确认**：gap(seen−strict) 从 40k 的 −0.0126 涨到 157.5k 的 +0.2089，
  47 个相邻点 42 个单调上升，斜率 +0.00217/1k（t=+23.0, R²=0.920）
- **LPIPS 已接入**（此前是"声明了但从没算过"的空列）

---

## 1. 已落地的配置（src/train/configs/）

全部与 v12 **逐字段一致，只改 `model` 一个字段** —— 干净的容量单变量。

| 配置 | model | depth | h | params | FLOPs vs M/2 | 预期步速 | 50k 步耗时 |
|---|---|---|---|---|---|---|---|
| `v12_pretrain_S_cat_fame_kxl_tj_px60` | S/2 | 12 | 384 | 37.24M | 0.790× | 5.25 | ~2.6h |
| **`v13_pretrain_XS_cat_...`** (P1) | XS/2 | **8** | 384 | 26.61M | **0.527×** | **7.88** | **~1.8h** |
| **`v14_pretrain_S320_cat_...`** (P2) | S320/2 | 12 | **320** | 26.45M | 0.549× | 7.56 | ~1.8h |
| **`v15_pretrain_XS6_cat_...`** (P3) | XS6/2 | **6** | 384 | 21.29M | 0.395× | 10.50 | ~1.3h |

**注意 XS/2 与 S320/2 是一对天然对照**：参数量几乎相同（26.61M vs 26.45M），
但一个缩深度、一个缩宽度 → 可直接回答"**深度还是宽度更该缩**"。

参考：M/2 = 47.04M / 1.000× / 4.15 step/s；Sp/2 = 66.70M / 1.405× / 2.95 step/s。

**FLOPs 比可直接预测步速比 —— 已验证**：S/2 vs M/2 FLOPs 0.790 → 预期 1.266×，
实测 5.28/4.15 = 1.272×。

---

## 2. 推荐执行顺序

```
P0  补 LPIPS ──────────────── ✅ 已完成 (in_mem_eval 已计算并落 CSV)
    └ 顺带结论: 20k 时 v12 vs v11 的 LPIPS 配对差 +0.0019 (t=+0.67) 不显著
      → 与 ssim 一致, "v12 明显更差" 的警报被两个指标同时否掉

P1  v13 (XS/2, depth 8) @50k ── 1.8h  → 回答"深度是否过度配置"
P2  v14 (S320/2, h=320) @50k ── 1.8h  → 回答"宽度是否过度配置"
P3  v15 (XS6/2, depth 6) @50k ─ 1.3h  → 定位深度下界
```

每个探针**只需 50k 步**（v11 的 gap 发散出现在 ~50–75k，50k 足够看出苗头），
不必跑满 400k。三个探针合计 **~5 小时**。

---

## 3. 判据（重要）

**主判据**：与 v12 在**同 step** 做**逐样本配对检验**（两者用同一批 eval 样本，
SE 仅 ~0.003，比独立比较灵敏约 4 倍）。

**同时看两个指标**：

| 指标 | 角色 |
|---|---|
| **LPIPS** | 对结构细节敏感，能分辨"墨团"与"成形" |
| ssim | 只看趋势，**不能**作为"是否成型"的判据 |

⚠ **doc59 已实测 ssim 会撒谎**：15k 时 v11 能看出字形、v12 是墨团，
而两者 ssim 几乎相同（0.4796 vs 0.4860）。**每 5k 步必须看图。**

**判定**：
- LPIPS 配对差 **不显著** → 该容量维度确实过度配置，继续往下缩
- LPIPS 显著变差（t > +2，LPIPS 越高越差）→ 撞到下界，取上一档

---

## 4. 可用开关矩阵

以下开关**全部已注册为 argparse 参数并冒烟验证**（含 12ch 下 CFG 作用域正确性）。

### 4.1 容量 / 结构

| 开关 | 取值 | 说明 |
|---|---|---|
| `model` | 见 §1 表 | XS6/XS/S320/S/M/Sp/B |
| `rope` | 0/1 | 1 = 纯 RoPE(不加绝对位置嵌入) |
| `norm_type` / `mlp_type` | rms/layer, swiglu/gelu | |
| `qk_norm` | 0/1 | |

### 4.2 条件融合（v12 新增）

| 开关 | 取值 | 说明 |
|---|---|---|
| `condition_fusion` | `factorized_add` / `factorized_cat` | cat = 各向量因子拼接后**一个**联合 Linear（ref/Moyun 式）；add = 各自投影后加权求和 |
| `glyph_vec_cond` | true/false | 把 g 池化成全局向量，作为 **concat 的第二个操作数**。不开则 cat 退化为单层 Linear（与 add 参数量完全相同 36.4571M） |
| `glyph_vec_dim` | int (128) | |
| `glyph_vec_pool` | `mean` / `max` | g_tok 256 token 的池化方式 |

**concat 落在哪**：`c = t_emb + Linear(concat([e_callig, e_glyph_vec]))`。
理由：ref 拼的是"向量因子"，我们第二个因子只能把空间条件 g 向量化；
且原先 `c = t_emb + callig_proj` 让 **adaLN 调制分支从来看不到内容**。

### 4.3 glyph 注入方式

| 开关 | 取值 | 说明 |
|---|---|---|
| `glyph_inject_mode` | `adaln` / `xattn` | adaln = ZeroAdaLNInjection（固定 1:1 位置调制）；xattn = ZeroCrossAttention（内容寻址，GlyphDraw/IP-Adapter 式） |
| `glyph_inject_layers` | int | 注入层数，均匀分布 |
| `glyph_scale_init` | float | 输入层 token-add 强度 |
| `glyph_embedder_depth` | int | g 编码器深度（0=单层 Conv） |
| `glyph_drop_prob` | float | g 训练期随机丢弃 |

⚠ doc60 把 **xattn vs adaLN 判为"不确定"**（只有 80k 短期对照持平，
缺同预算长训对照）。xattn 参数更多，80k 持平**可能只是还没训够**。

### 4.4 12ch 辅助目标（ref 式）

| 开关 | 取值 | 说明 |
|---|---|---|
| `aux_latent_shards_dirs` | 逗号分隔路径 | 每个 dir 一组 (N,4,32,32) latent，与图像 latent 拼接成扩散目标 |
| `aux_loss_weights` | 如 `"0.3,0.8"` | **per-group 权重**，覆盖 `aux_loss_weight` |
| `aux_loss_weight` | float | 单一 aux 权重（默认 1.0 = 等权） |
| `latent_channels` | int (4) | 图像 latent 通道数 |
| **`image_channels`** | int (None→latent_channels) | **CFG 作用域**。12ch 下**必须保持 4** |

⚠ **12ch 的两条纪律**（doc59）：
1. **等权有害** —— 结构通道会吃掉 ~51% final-layer 梯度（`patch_embed` aux 是 img 的 2.6×）
2. **CFG 作用域必须显式设 4** —— aux 分布与 image 不同，被同一 `cfg_scale` 放大
   会让采样轨迹跑飞，这是"墨团"的直接原因。已冒烟验证：12ch 下 CFG 只改前 4 通道，
   aux 通道逐位不变（d_aux ~1e-7 为 fp32 噪声）。

⚠ doc60 把 12ch 判为**"已证有害"**（在我们有 g 条件的前提下）。ref 用 12ch 是因为
它 `use_stroke=False`、**没有结构条件**；我们结构已从条件侧给过 → 重复投入。
保留为开关是为了复评历史 ckpt，不是推荐配方。

### 4.5 条件 dropout

| 开关 | 取值 | 说明 |
|---|---|---|
| `cond_drop_all_prob` | 0.1 | 全丢（uncond 分支） |
| `cond_drop_one_prob` | **0.0** | 单因子丢。**当前为 0 → 4-way mask 退化成 2-way** |
| `cond_drop_which_glyph_prob` | 0.85 | 单因子丢时偏向丢 callig。**因上项为 0 而完全没生效** |
| `no_char_cond` | true | 移除 char 向量因子（字身份由 g 承担） |

⚠ ref 用 `charactor-x 0.08 / font-x 0.08 / calligrapher-x 0.16`（总 0.32）。
我们单因子 drop 是 0。**方向一致（都偏向训练 content score）但强度差 3 倍以上。**

### 4.6 评测（零训练成本，最便宜的实验）

| 开关 | 当前 | 建议 |
|---|---|---|
| `gpu_eval_cfg` | **0.7**（<1 = 在**稀释**条件） | sweep {0.5, 0.7, 1.0, 1.5} |
| `eval_steps` | 50（heun 二阶 = 100 NFE） | sweep {50, 100} |
| `in_mem_eval_lpips` | true（新） | 保持开 |
| `repa_layers` | "8"（**新注册**） | 可试 {8} vs {8,11} |

⚠ **cfg sweep 是最该先做的**：`cfg < 1` 意味着在把输出拉向 uncond（通用）分支，
而"笔画糊成一团"正是 cfg 不当的典型症状。若 `cfg=1.2` 就能让笔画分开，
前面几轮"模型不行"的判断都要重估。**成本几乎为零。**

---

## 5. 我修的两个静默丢弃（doc56/59 同类坑，第 3、4 次）

| 键 | 状态 | 后果 |
|---|---|---|
| `image_channels` | **此前未注册** | config 里写 `image_channels` 被静默丢弃；CFG 作用域只能靠 `latent_channels` 间接设，命名易混 |
| `repa_layers` | **此前未注册** | config 值被静默丢弃，实际永远走硬编码兜底 `or (8,)`。v11/v12 恰好都想要 8 所以没暴露 |

两个都已注册。`repa_layers` 默认 `"8"` 与旧兜底**行为完全一致**（已验证）。

---

## 6. 一条纪律

**不要在饱和的指标上做容量决策。**
"strict 对容量不敏感"既可能是"容量够了"，也可能是"指标看不出差别"。
先补 LPIPS（已完成）再缩容量，否则收益无法与"静默丢质量"区分开。
