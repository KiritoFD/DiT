# 67. 12ch / xattn 实现审计

> 日期：2026-09-17
> 前置：`v12_w320` 已停（ckpt → `assets/results/v12_w320_last.pt`），GPU 空闲。
> 相关：[54](54_results_and_insights_20260913.md)、[59](59_12ch_and_white_zero_retrospective.md)、
> [60](60_what_is_actually_useless.md)、[61](61_ref_moyun_code_audit.md)、[65](65_experiment_plan_v2.md) §2.4

---

## 0. 结论

**12ch 的主干路径（concat / CFG 作用域 / 采样器）实现是正确的**，
但有 **1 个危险默认**（忘了给权重就静默等权）和 **1 个缺断言**；
**px60 只备了 skel 的 aux，没有 canny** → 12ch 在 px60 上只能做 **8ch**。

**xattn 有一个真实的实现缺陷：Q 没有位置嵌入，只有 K/V 有**
→ 所谓"空间寻址"实际退化成"内容寻址"，与 docstring 声称的机制不符。
另有一个已知设计问题（对 CFG 差分贡献为 0，见 doc 65 §2.4）。

---

## 1. 12ch 审计

### 1.1 ✅ 正确的部分

| # | 位置 | 核实结果 |
|---|---|---|
| 1 | `train.py:1189` | `x_latent = cat([x_latent, aux_latents], dim=1)` → `[image(4), aux...]`，与 ref 的 `cat((image, edge, skeleton))` **顺序一致** |
| 2 | `train.py:302` | `image_channels = args.image_channels if 非None else latent_channels` → 12ch 下解析为 **4**（CFG 作用域），且带显式注释说明"不能回退到 in_channels" |
| 3 | `train.py:1199` | `_ch_w[4 + 4*gi : 8 + 4*gi] = w` → per-group 权重索引与 `aux_latent_shards_dirs` 顺序一致 |
| 4 | `dit.py:1239` | `forward_with_cfg` 只对 `model_out[:, :image_channels]` 做引导，`rest` 原样返回 —— **比 ref 的 `[:, :3]` 硬编码强** |
| 5 | `flow_matching.py:307-386` | ODE 循环**与通道数无关**（`C = shape[1]`），12ch 会完整积分，aux 通道不会被漏掉 |
| 6 | `flow_matching.py:278` | `_v` 里 `if v.shape[1] == 2*C: v = v[:, :C]` —— 用 C=12 时正确切掉 learn_sigma 通道 |
| 7 | `latent_dataset.py:351` | `aux_t = cat([... for a in self._aux_latents], 0)` —— 顺序与 `aux_latent_shards_dirs` **严格一致** |
| 8 | `train.py:266` | `glyph_inject_mode` / `glyph_inject_layers` / `aux_*` / `image_channels` / `latent_channels` **全部已注册**（无静默丢弃） |

### 1.2 ⚠ P1：危险默认 —— 忘了给权重就**静默等权**

`train.py:1194-1205`：

```python
_aux_w_list = [float(s) for s in str(getattr(args,'aux_loss_weights','') or '').split(',') if s.strip()]
if _aux_w_list:
    _ch_w = torch.ones(...); _ch_w[4+4*gi : 8+4*gi] = _w
else:
    _w_aux = float(getattr(args, 'aux_loss_weight', 1.0))
    if _w_aux != 1.0:              # ← 1.0 时 _ch_w 保持 None
        _ch_w = torch.ones(...); _ch_w[4:] = _w_aux
```

**配了 `aux_latent_shards_dirs` 但忘了写 `aux_loss_weights` → `_ch_w = None` → 等权。**

而等权正是 doc 54 / 60 判定**"已证有害"**的配置（结构通道吃掉 **~51% final-layer 梯度**，
`patch_embed` 上 aux 梯度是 img 的 **2.6×**）。**一个"忘了写"就掉进已知有害的坑，且不报错。**

→ **修法（二选一）**：
- **(a) 有 aux 但没给权重 → `raise`**（推荐：这是最容易踩的坑，宁可拒绝启动）
- (b) 安全默认（如 `0.1`），并在日志里显式打印生效权重

### 1.3 ⚠ P2：缺 `in_channels` 一致性断言

没有任何地方校验 `in_channels == latent_channels + Σ aux_channels`。
若 aux 目录数与模型 `in_channels` 不匹配，concat 后会在 `x_embedder` 处炸，
或者（更糟）在通道权重索引上静默错位。
→ 建议在模型构造前加断言 + 打印 `in_channels / image_channels / aux groups`。

### 1.4 ✅ 数据已就绪（**我之前的判断错了**）

`data/aux/inst_skel_latents_px60`：
- **447 shards，28,569 ids，shape (64,4,32,32) float16**
- **对 px60 的 28,569 张覆盖率 100.0%，缺失 0**

→ **实例骨架 latent 是齐的。** 我在 doc 66 里说"需要先补实例骨架 shards"是错的 ——
它已经存在（`inst_skel_shards_dir` 也已接进 dataset，见 `latent_dataset.py:110`）。

### 1.5 ⚠ P3：px60 只有 skel，**没有 canny**

`data/aux/` 下 canny 的 aux 有 `base / base_wz / fame_e / sym / v8`，
**没有 px60**。skel 有 `inst_skel_latents_px60` 和 `aux_skel3_latents_*`。

→ **12ch 在 px60 上只能做 8ch**（image 4 + skel 4），
或者先给 px60 补一份 canny aux（从 `final_canny_*` 的思路看，成本可控）。

---

## 2. xattn 审计

### 2.1 ✅ 正确的部分

| # | 位置 | 核实结果 |
|---|---|---|
| 1 | `dit.py:255-256` | `out_proj` **zero-init** → 初始严格恒等，不破坏已有训练 |
| 2 | `dit.py:245-249` | Q=x、K/V=`g_tok + ctx_pos`，`F.scaled_dot_product_attention`，形状处理正确 |
| 3 | `dit.py:832-833` | `glyph_inject_at` 均匀分布：depth=12、n_inj=4 → **`[2,5,8,11]`**（0-indexed，即第 3/6/9/12 层）；n_inj=12 → `[0..11]` 全层 ✓ |
| 4 | `controlnet.py` | `ZeroAdaLNInjection(mode="modulate")`：`x*(1+s)+t`，zero-init → 恒等 ✓ |
| 5 | `dit.py:837` | `glyph_inject_mode="xattn"` 且 `n_style_token==0` → 用 `ZeroCrossAttention`；`>0` → 用 `GlyphStyleCrossAttn` ✓ |

### 2.2 ⚠ P4：**Q 没有位置嵌入，只有 K/V 有** —— 真实实现缺陷

`dit.py:245-249`：

```python
q = self.q_proj(self.norm_x(x))                              # ← 没有 pos
k = self.k_proj(self.norm_c(context + self.ctx_pos[:, :Nc]))  # ← 有 pos
v = self.v_proj(self.norm_c(context + self.ctx_pos[:, :Nc]))  # ← 有 pos
```

而 `rope=True` 时，**x 的残差流完全不加绝对位置**（`dit.py` 的 `if self.rope: pass`，
位置信息只进 attention 内部的 q/k）。

→ **Q 是"无位置"的**，而 K/V 带绝对位置。
这意味着 xattn 的**空间对应关系只能靠内容相似度猜**，
无法可靠地锁定"同网格位置"的骨架 token。

**这与类自己的 docstring 矛盾**：

> "2D 绑定靠 g 网格与 x 网格相同 (16×16) + **固定 sincos 位置嵌入**保证"

—— 位置嵌入只在 K/V 侧，Q 侧没有，所以"2D 绑定"的保证是不成立的。

**这可能是 xattn 收益不明显的直接原因**：如果 Q 无法定位，
xattn 就退化成"每个 token 聚合全部骨架 token 的内容平均"，与全局池化差别不大。

→ **修法**：给 Q 也加同样的 `ctx_pos`：
```python
q = self.q_proj(self.norm_x(x + self.ctx_pos[:, :N]))
```
（`ctx_pos` 的 N_ctx 与 x 的 N 相同，都是 16×16=256，可直接复用。）
**建议作为 xattn 实验的一个独立开关**（`xattn_q_pos`），与现状做 A/B。

### 2.3 ⚠ P5：对 CFG 差分贡献精确为 0（设计问题，已在 doc 65 §2.4 记录）

`forward_with_cfg` 里 `g2 = cat([g, g])` —— **两半都给真实 g**，
所以 xattn 的 Q/K/V 在 cond / uncond 两半完全相同 → 对 `cond − uncond` **贡献精确为 0**。

→ 这解释了 **doc 54 的"xattn strict ≈0"**：不是 xattn 没用，
**是在 cfg 引导的评测口径下它天然被绕过**。修法是内容轴 CFG（doc 65 §2.4.4）。

### 2.4 一个非缺陷但值得知道的事实

K/V 在**每个注入层各自重算**（每层有自己的 `k_proj`/`v_proj`），
所以不是冗余计算。若要省，可以做**跨层共享 K/V 投影**（doc 65 的 B3），
代价是表达力下降 —— 这是设计取舍，不是 bug。

---

## 3. 修复状态（2026-09-17 全部已改并验证）

### 3.0 ★ Fix 1（P0，审计时额外发现）：`factorized_cat` 不在 drop guard 里

`dit.py:1013` 的元组是 `("factorized_add", "xl_highdim")` —— **漏了 `factorized_cat`**，
而 cat 分支在下方**同样复用 `y_callig_in` / `char_drop`**。

→ 整个 drop 块被跳过 → `callig_drop is None` → `y_callig_in = y_callig`
→ **训练全程零条件 dropout**，null token 从未出现，`null_embed` 停在随机初始化
（std=0.02）→ 推理时 `forward_with_cfg` 的 uncond 半用的是**未训练**的向量
→ **`cfg=0.7` 实际是在往一个随机方向插值，不是真正的 CFG**。
不报错、loss 正常下降 —— 又一例静默失效。

**影响**：v12 / v12_12ch / v12_xattn / v12_d8 / v12_w320 / v13~v16 全部（都是 cat）。
v11（`factorized_add`）不受影响。

**修法**：元组加 `"factorized_cat"`，并在代码里写明"新增 fusion 分支必须回来加"。

### 3.1 修复清单与验证

| # | 动作 | 文件 | 验证 |
|---|---|---|---|
| **Fix 1** | cat 加入 drop guard 元组 | `dit.py:1013` | **T1**: cat + drop=1.0 → `\|forward(real) − forward(null)\| = 0`（drop 生效）；对照 drop=0.0 差值 4.87e-2；`factorized_add` 对照通过 |
| **A1** | `aux_loss_weights` 缺失 → **raise** + 提示建议值 | `train.py` | 实测第一个 step 抛出清晰 ValueError（含建议值 `0.3,0.3`） |
| **A2** | `image_channels == latent_channels` 断言 + 启动打印 | `train.py` | 实测 `image_channels=12` 启动即拦；正常配置打印 `[channels] latent=4 aux 组=2 in_channels=12 image_channels(CFG 作用域)=4` |
| **A3** | xattn 的 Q 加 sincos 位置嵌入，开关 `--xattn-q-pos`（默认 False 兼容旧 ckpt） | `dit.py` + `train.py` | **T2** 开关生效（差值 2.94）；**T3** 默认 False → 旧 ckpt 兼容 |

⚠ **写验证脚本时踩的坑**：DiT 的 `final_layer.linear` / `adaLN_modulation[-1]` 是
**zero-init → 输出恒为 0**。不打破它，测出来的全是"0 和 0 的差"，
第一版测试就因此给出了两个假 FAIL。**做这类等价性测试前必须先打破 zero-init。**

### 3.2 仍待做

| # | 动作 | 说明 |
|---|---|---|
| A4 | 给 px60 补 canny aux | 现状 `aux_canny_latents_base` 对 px60 覆盖 100%，可直接用；但若要"px60 自己的 canny"需重编 |
| A5 | 内容轴 CFG（doc 65 §2.4.4） | **xattn 对 CFG 差分贡献仍为 0**（`g2=cat([g,g])`）→ xattn 的收益在 cfg 引导下仍会被绕过。⚠ 这是 xattn 实验的已知偏差 |

---

## 3.5 CFG 到底哪里不对，以及该怎么做

**两个互相独立的问题，只有一个算 bug。**

### 问题 A（bug）：uncond 分支用的是未训练的参数

CFG 需要两个分支：`cond`（真条件）和 `uncond`（把条件换成 null token）。
训练时必须**随机做条件 dropout**，让模型学会"没有条件时该怎么预测"。

`dit.py:1013` 的 guard 元组是 `("factorized_add", "xl_highdim")` —— **漏了 `factorized_cat`**，
而 cat 分支在下方**同样复用 `y_callig_in` / `char_drop`**。

→ 整个 drop 块被跳过 → `callig_drop is None` → `y_callig_in = y_callig`
→ **训练全程零条件 dropout**，null token 从未出现，`null_embed` 停在随机初始化
→ 推理时 `uncond_eps` 是拿**未训练的向量**算出来的 → `cfg=0.7` 是在**往一个随机方向插值**。

**特征**：不报错、loss 正常下降、指标看起来也正常 —— 又一例静默失效。

**影响范围**：所有用 `factorized_cat` 的 run（v12 全系 + v13~v16）。
v11 用 `factorized_add`，不受影响。

**修法**：元组加 `"factorized_cat"`（已做）。**新增 fusion 分支时必须回来加。**

### 问题 B（设计选择，但有副作用）：`g` 在差分中结构性抵消

```python
g2 = torch.cat([g, g], dim=0)   # 两半都用真实 g
```

`eps_cond − eps_uncond = f(真实书家, g) − f(null, g)` —— **g 项在两侧相同，完全抵消**。

注释写明这是**有意**的（"字形内容是正条件, CFG 只强化 callig 风格"）。但它有个必然副作用：

> **任何"只由 g 驱动"的通路，对 CFG 的贡献恒为 0。**

`xattn` 的注入、`glyph_embedder` 都属此类。**这解释了 doc 54 的"xattn strict ≈ 0"** ——
不是 xattn 没用，**是在 cfg 引导的评测口径下它天然被绕过**。

### 该怎么做 —— 三层

**层次 1（必须，已做）**：修 drop guard，让 uncond 分支真的被训练过。

**层次 2（推荐，成本为零）**：**组件结论用 `cfg=1.0`（纯条件）作为默认报告口径**，
把 cfg 引导当成一个**独立的推理期旋钮**单独扫。

理由：`cfg=1.0` 时根本不走 uncond 分支 → 问题 A 和 B 都不影响。
**组件有没有用，不应该依赖一个推理期超参。**

**层次 3（要真做内容轴 CFG）**：把 g 也纳入差分。两种做法：

| 做法 | 机制 | 代价 |
|---|---|---|
| (a) 单轴 g | uncond 半把 g 也置零（空骨架），`eps = eps_uncond + w·(eps_cond − eps_uncond)` | 风格与内容引导**混在一起**，不能分别调 |
| (b) 双轴 | 跑 4 个 pass：`full / callig-only(g=0) / glyph-only(null_callig) / uncond(null_callig,g=0)`，各自独立权重 | 4× NFE |

⚠ **`forward_with_2axis_cfg` 现成实现对我们的架构不适用**：
它把 "glyph 轴" 定义在 **`y_char`**（字符 ID）上，且 `g4 = cat([g,g,g,g])` ——
**g 仍然在 4 个 pass 里完全相同**。
我们的 v12 是 `no_char_cond=True`（char 通路关闭，内容来自 g），所以：
- glyph 轴 = **no-op**（char 被忽略）
- g **仍然抵消**

→ 要用得改：把 "glyph 轴" 从 `y_char` 改到 **`g`**（例如置零或换成空骨架）。

### 为什么现在可以先不做

后面换 HCSU 数据本来就要**从头训**，所以修 drop guard 是**零额外成本**（已做）。
问题 B / 内容轴 CFG 属于"推理期增强"，**不影响任何组件结论**，可以最后再加。
**但前提是：组件结论必须用 `cfg=1.0` 口径报告**，否则会重演 doc 54 的误判。

---

## 4. 已拉起的实验

`_sync_work/run_12ch_xattn.sh`（tmux `v12series2`），串行：

```
v12_12ch  → v12_xattn
```

**配置改动**：

| 配置 | 改动 | 理由 |
|---|---|---|
| `v12_xattn_pretrain.json` | `global_batch_size 260 → 360` | **对齐 v12 基线** —— 否则"同 step"不等于"同数据量"（v12_d8 的教训：576 vs 360，同 step 差 60% 数据） |
| 同上 | `xattn_q_pos: true` | 启用 A3 修复 |
| `v12_12ch_pretrain.json` | 未改 | `aux_loss_weights: 0.3,0.8` 已给、batch 360 已对齐 |

**12ch 数据核验**：`aux_canny_latents_base` 与 `inst_skel_latents_px60`
对 px60 的 28,569 张均 **100% 覆盖**（canny 是超集，54,892 ids）。

⚠ **xattn 实验的已知偏差**：A5 未做，所以 xattn 对 CFG 差分的贡献仍为 0。
若要公平评价 xattn，需要先上内容轴 CFG，或把 eval 的 cfg 设为 1.0（纯条件）。
