# 919 / 20 — 模型架构与条件机制

> 快照日期 2026-09-19。源文件：`src/model/dit.py`（~1400 行）、`src/model/modules.py`。

---

## 1. 骨干：DiT-2Cond（本仓库自建，非官方 DiT）

官方 DiT 只有 class-condition；**本仓库的核心改造是"开集字形条件 g"**。

### 1.1 规模变体（实测参数量）

| variant | depth | h | heads | 参数量 | FLOPs ∝ d·h² | vs M/2 |
|---|---|---|---|---|---|---|
| `DiT-2Cond-XS6/2` | 6 | 384 | 6 | 20.60M | 884,736 | 0.395× |
| `DiT-2Cond-XS/2` | 8 | 384 | 6 | **25.92M** | 1,179,648 | 0.527× |
| `DiT-2Cond-S320/2` | 12 | 320 | 5 | **25.93M** | 1,228,800 | 0.549× |
| `DiT-2Cond-S/2` | 12 | 384 | 6 | **36.55M**（可训 36.45M） | 1,769,472 | 0.790× |
| `DiT-2Cond-M/2` | 12 | 432 | 6 | 47.05M | 2,239,488 | 1.000× |
| `DiT-2Cond-Sp/2` | 12 | 512 | 8 | 66.70M | 3,145,728 | 1.405× |

**FLOPs 比可直接预测步速比 —— 已验证**：
S/2 vs M/2 FLOPs 比 0.790 → 预期 1.266×，实测 5.28/4.15 = **1.272×** ✓

### 1.2 现代化组件（`src/model/modules.py`，全部可开关）

| 组件 | 原版 DiT | 本仓库 | 默认 |
|---|---|---|---|
| 归一化 | LayerNorm(无 affine) | **RMSNorm**(无 affine) | `norm_type=rms` |
| FFN | GELU MLP | **SwiGLU**（参数等量：h=8D/3） | `mlp_type=swiglu` |
| 位置编码 | 固定 2D sin-cos 加到 x | **2D axial RoPE**（作用 q/k） | `rope=1` |
| Attention | timm.Attention | **QK-Norm + SDPA** | `qk_norm=1` |

**约束**：`elementwise_affine=False` —— 尺度/平移全由 adaLN 提供，
所以 adaLN zero-init → 每个 block 初始是恒等映射。

**RoPE 是 `persistent=False` buffer** → 不进 state_dict → 不破坏任何已有 ckpt 的 key 集合。

`attn_impl=sdpa`（torch≥2.0 用 `F.scaled_dot_product_attention`；xformers 只作 fallback，
已在 infra 升级中从 cu121 卸载）。

---

## 2. 条件机制（本项目最关键的设计）

### 2.1 三个条件入口 + 一个身份约定

```
输入: x (noise latent 4×32×32)  t  y_callig (书家 ID)  g (标准字形 latent 4×32×32)

① 书家风格 y_callig  →  y_callig_embedder (52/45/87 行 × 128)
                        → condition_fusion → c = t_emb + y_emb
                        → adaLN 调制每个 block（scale/shift/gate）

② 字形内容 g        →  glyph_embedder (Conv 堆) → g_tok (256, D)
                        → x = x + glyph_scale · g_tok        （输入层 token-add）
                        → glyph_injections（逐层注入，见 §2.2）

③ 字身份 = g        ← 【关键取舍】no_char_cond=true，不用 char ID
                        字身份完全由 g 承担 → 开集（能写没训练过的字）
```

**为什么不用 char ID**（doc 60 §4）：

| | 用 char ID | 用 g（本方案） |
|---|---|---|
| 字身份 | 离散 ID，**闭集** | 标准字 latent，**开集** |
| 没训练过的字 | **写不出来** | 只要字体库能渲染就能写 |
| 代价 | — | 标准字形信息量不足（这就是瓶颈） |

### 2.2 g 的两条注入路径（可选，历史包袱多）

| 模式 | 开关 | 机制 | 状态 |
|---|---|---|---|
| **输入层 token-add** | `glyph_scale`（可学习） | `x = x + glyph_scale · g_tok` | **必需**（置零 ssim −0.28） |
| **逐层 adaLN 注入** | `glyph_inject_mode=adaln` | `x*(1+s)+t`，zero-init | 当前默认（4 层） |
| **逐层 xattn 注入** | `glyph_inject_mode=xattn` | `ZeroCrossAttention`（Q=x(+pos 可选), K/V=g_tok） | 历史最高分用它，但**不确定**；v15c 用它 |
| **风格 token** | `style_token_n>0` + xattn | 书家向量→N 个 style token 参与 K/V | 实测 ≈0（但未公平重训） |
| **多模态风格三式** | `callig_multi_style_k>0` | 池化→adaLN / CA 书家化 / 每层 ctx（见 §2.6） | **v15a/b/c 在跑** |

⚠ **D1 教训（doc 48）**：曾把输入层 token-add 判为"与 xattn 重复、可删"，
置零后 **ssim −0.28 / skel_iou 0.223→0.011（几乎归零）** —— **它是骨架条件的主力通道**。
原因：`learned glyph_scale = 0.1896`（init 0.6），模型主动把它调小 3 倍但**没关掉**。

### 2.3 条件融合（`condition_fusion`）

| 模式 | 机制 | 备注 |
|---|---|---|
| `legacy` | 联合 MLP | 旧 |
| `factorized_add` | 各因子独立投影后相加 | |
| **`factorized_cat`** | 各向量因子**拼接** → 一个联合 Linear | **v12 起默认**（ref/Moyun 式） |
| `xl_highdim` | 高维、保预训练 adaLN | 旧 |

**`factorized_cat` 的细节**：操作数 = `[e_callig]` +（若 `glyph_vec_cond`）`[e_glyph_vec]`。
v15 多模态下 `e_callig` = **K 个 token 的 mean pooling**（384 维），cat 输入 384+128=512：

```
c = t_emb + Linear( LayerNorm( concat([pool(style_tokens)(384), e_glyph_vec(128)]) ) )
```

**为什么 `glyph_vec_cond` 重要**：原先 `c = t_emb + callig_proj(e_callig)`，
**adaLN 调制分支从来看不到"在写哪个字"** —— 每个 block 的 scale/shift/gate
都不知道内容。把 g 池化成向量（256 token → mean/amax）拼进去，第一次让 adaLN 看到内容。

⚠ 若两个操作数都没有，cat 退化成单层 Linear，**与 add 参数量完全相同**（36.4571M）。

### 2.4 条件 dropout（4-way，用于 CFG）

```
cond_drop_all_prob      全丢        → uncond（CFG 基准）
cond_drop_one_prob      单因子丢    → glyph-only / callig-only
cond_drop_which_glyph   偏向丢哪个
```

⚠ **历史静默失效（已修）**：`dit.py` 的 drop guard 元组里**漏了 `factorized_cat`** →
`callig_drop is None` → **训练全程零条件 dropout** → `null_embed` 停在随机初始化 →
推理时 CFG 的 uncond 半用的是**未训练**的 null 向量（`cfg=0.7` 实际是往随机方向插值）。

**该 bug 2026-09-17 才修** → 此前所有 cat 系 run（含 v12_12ch）的 **CFG 评测口径不可信**。

⚠ **第二个面**：`null_embed` 加载时被静默丢弃（`materialize_lazy_params` 必须在
`load_state_dict` **之前**调用）→ 恢复成随机。**两者叠加**：既没训好、又没存对。

**v15 的 drop 语义**：`MultiStyleEmbedder` 的 dropout_prob 恒为 0（**不用**模块内 drop），
null 标签（=num_classes）由 forward 顶部的 4-way mask 统一写入 —— drop-callig 样本的
K 个 token **整组**替换为 `null_embed.view(K,D)`，保证 adaLN 分支与风格注入共享同一份 mask。
另加 `glyph_drop_prob=0.1`（此前全为 0），使 g=0（纯风格分支）成为训练内条件 ——
**双轴 CFG 的 content 轴从此可用**。

### 2.5 CFG 实现（`forward_with_cfg`）

```
经典 2 路:  eps = uncond + cfg · (cond − uncond)
            ⚠ g2 = cat([g, g]) → 两半给同一个 g
            → 任何"只由 g 驱动"的通路（xattn / glyph_embedder）在
              (cond − uncond) 里**完全抵消，对 CFG 贡献恒为 0**

双轴（已实现，需 glyph_drop_prob>0 才可用）:
  pass1 full    : (y_callig, g)     -> eps_full
  pass2 style   : (y_callig, g=0)   -> eps_style
  pass3 content : (null,     g)     -> eps_content
  pass4 uncond  : (null,     g=0)   -> eps_uncond
  eps = eps_uncond
      + cfg_glyph · (eps_content − eps_uncond)                 内容轴
      + cfg_callig · (eps_style   − eps_uncond)                风格轴
      + w_inter   · (eps_full − eps_content − eps_style + eps_uncond)  交互项
```

⚠ **前提**：`glyph_drop_prob > 0`（`g=0` 必须是训练时见过的条件）。
**当前所有 config 的 `glyph_drop_prob` 都是 0 → 内容轴从未被训练 → 双轴 CFG 用不了。**

✅ **CFG 作用域修复（doc 58）**：`forward_with_cfg` 原先对 `[:in_channels]` 全做 CFG，
12ch 下 `in_channels=12` → **canny/skel 结构通道被同一 cfg_scale 放大**（分布不同）→
采样轨迹跑飞（**"墨团"的直接原因**）。已加 `image_channels` 参数（CFG 只作用前 4 通道）。

### 2.6 v15 多模态风格：MultiStyleEmbedder + 注入方式矩阵（当前）

**`MultiStyleEmbedder`**（`src/model/dit.py`）：每个风格类（书家×书体 pair）K=4 个 token
的查表，`embedding_table (87, K×384)` + 独立可学习 `null_embed (K×384)`（不占表行），
`forward(labels) -> (B, K, D)`；null 标签整组替换。表由 **DINO K-Means 质心**初始化
（`tools/build_multistyle_k4.py`，每个 token 对应该 pair 的一个风格模态）——防 K token
训练初期隐式塌缩。动机：单冻结 128 维向量装不下多模态风格（r(样本数, strict)=−0.62）。

**三种注入方式**（`callig_multi_style_k>0` 下的单变量矩阵，全部**从头训练**）：

| 变体 | 开关 | 机制 |
|---|---|---|
| **v15a** | （默认） | K token **mean pooling** → cond_fusion → adaLN（全局向量，无空间通路） |
| **v15b** | `callig_style_ca=true` | + **CalligStyleCrossAttn**（v15 重写版）：Q=骨架 token+2D sincos 位置，K/V=K 个风格 token，zero-init out_proj → step0 恒等。旧版（单向量经内部 style_proj 展开、Q 无位置）已存档 `legacy/dit_core/callig_style_cross_attn_v1.py` |
| **v15c** | `style_ctx_every_layer=true`（需 xattn 注入） | K token 拼进**每层** `GlyphStyleCrossAttn` 的 context（K/V），风格在每层、每个空间位置可直接寻址 |

配套机制：
- **锚定正则** `--style-anchor-mode mean`：K token 的 mean pooling 拉向 pair 级 DINO 质心
  （`pair_mean`），λ=0.01 —— 允许 token 各自分化，只约束均值不漂，防 K 簇重新塌缩。
- **`ZeroCrossAttention` 的 K/V 位置修复**：原实现给整个 context 加 16×16 sincos，
  v15c 的 256+K context 直接崩溃；现改为前 grid² 个 token（骨架）加位置、尾部风格 token
  （抽象语义）不加。
- **`--callig-multi-style-k` 与 `callig_spatial`/`style_token_n>0` 互斥**（后两者读 2-D
  e_callig，构造时直接 raise）。
- **架构演进 resume 的通用护栏** `drop_shape_mismatched`（`src/utils/channel_expand.py`）：
  resume 时形状失配的键剔除并并入"新模块"集合（冻结策略自动放开）——v15 若走 resume
  会剔除 {书家表, null_embed, cond_fusion.*}；但**注入方式变了 resume 仍不公平**
  （adaLN 条件几何断层 + 冻结主干不会读新通路 + 判据不可归因），故 v15 全部从头。

---

## 3. 已证伪组件的**机制**解释（不是"没调好"）

### 3.1 12ch 为什么在 ref 成立、在我们这失败

| | ref (Moyun) | 我们 |
|---|---|---|
| 结构信息入口 | **只有 12ch 目标通道**（`use_stroke=False`，**没有任何结构条件**） | **g 条件**（已覆盖）+ 12ch 目标通道 |

ref 的 5 个生产配置**全部** `use_stroke=False` → 12ch **必需**；
我们结构已从条件侧喂进去 → 12ch 是**第二条结构通路 = 同一个信号喂两遍**。

**三个附加失效机制**：
1. **梯度稀释**：等权下 img 4ch : aux 8ch = 1:2，image 只拿 1/3 损失权重；
   且 aux std 更低更易拟合 → 优化器优先压 aux（实测结构通道吃 **~51% final-layer 梯度**，
   `patch_embed` 上 aux 梯度是 img 的 **2.6×**）
2. **CFG 作用域 bug**：见 §2.5
3. **flow 路径被扭曲**：aux 分布与 image 不同，等权下线性插值路径更弯曲

⚠ **ref 的 CFG 其实更糟（不要学）**：`moyun_2.py:648` 硬编码 `model_out[:, :3]`，
12ch + learn_sigma → 输出 24 通道，只引导 3/24 = 12.5%。

### 3.2 白底归零（wz）为什么放弃

**理论收益成立**：白底区 `v = ε − 0 = ε`，恰好是噪声本身，这部分不用学。

**但代价实测**：

| 指标 | 未减白底 | wz |
|---|---|---|
| img `mean` | +0.28 | **−0.33**（左偏） |
| img 极端值占比（abs(x)>2） | 4.1% | **12.1%**（重尾 3×） |
| aux `std` | 1.21/1.04 | **0.57/0.56**（减半） |

⚠ **VAE 零向量 ≠ 白**：`decode(0) = [129,110,89]`（灰黄棕），推理必须加回 ——
而 `aux_zero_white` **从未注册为 argparse 参数** → config 值被**静默丢弃** →
所有 decode 路径都没加回 → **生成图整体发黄/更负处发黑**（doc 56 事故）。

⚠ **ref 有完整实现（`custom_zero`）但 5 个 .sh 全部为 0** —— 作者有构想但从未启用。

### 3.3 xattn 的 Q 没有位置嵌入（真实实现缺陷，doc 67 §2.2）

`ZeroCrossAttention` 原实现**只给 K/V 加 sincos 位置，Q 没有**。
而 `rope=True` 时 x 的残差流不加绝对位置 → **Q 完全无位置信息** →
所谓"空间寻址"退化成**内容寻址**（只能靠内容相似度猜该看哪个骨架 token）。

**已加 `xattn_q_pos` 开关（默认 False 保 ckpt 兼容）**，但当前 config 都没开。

---

## 4. 参数量预算（doc 62 推演）

三条独立估计收敛到 **N\* ≈ 15–25M**（比 M/2 的 47M 小一半）：

| 依据 | 估计 |
|---|---|
| strict 对容量不敏感（α≈0，且未排除 ssim 饱和） | ≤ 37M，很可能 ≪ |
| gap 减半目标（47M × 0.5） | **~23.5M** |
| params/sample 启发式（100–1000） | **14–23M** |

⚠ **但实测打脸了一半**：
- **缩深度（XS/2, 25.9M）→ strict −0.026**（v12_d8，见 `30_training.md`）
- 缩宽度（S320/2, 25.9M）→ 早期领先，但最终不显著
- → **容量不是"过度配置"，depth 真的有价值**

⚠ **纪律**：不要在饱和的指标上做容量决策。"strict 对容量不敏感"既可能是"容量够了"，
也可能是"指标看不出差别"。

---

## 5. 关键代码位置索引（`src/model/dit.py`）

| 组件 | 行号（约） |
|---|---|
| `LabelEmbedder`（含可学习 null_embed） | 60-135 |
| `get_2d_sincos_pos_embed` | 146-193 |
| `ZeroCrossAttention`（xattn 注入） | 214-269 |
| `GlyphStyleCrossAttn`（风格 token 每层可见） | 272-315 |
| `CalligStyleCrossAttn`（书家化骨架） | 317-365 |
| `DiT_2Cond.__init__`（全部开关） | 380-920 |
| `initialize_weights`（zero-init 恢复） | 919-996 |
| `forward`（条件 dropout + g 注入 + fusion） | 1008-1271 |
| `forward_with_cfg` / `forward_with_2axis_cfg` | 1273-1400 |
| 模型注册表 `DiT_2Cond_models` | 1359+ |

---

## 6. 其他模块

| 模块 | 路径 | 作用 |
|---|---|---|
| Fused RMSNorm | `src/model/fused_rmsnorm.py` | Triton 融合 kernel（可选，`fused_rmsnorm=1`） |
| REPA | `src/model/`（`repa_*` 开关） | DINOv2 特征对齐，w=0.03；实测会劫持梯度（158%） |
| VAE | SD-VAE f8（`stabilityai/sd-vae-ft-ema`） | 4ch×32×32；**在 1px 细线上不保真** |