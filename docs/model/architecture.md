# 模型架构 / Model Architecture

> 本文档描述 **DiT-MCCD** 的网络结构、条件机制与 VAE 集成。
> 所有模型定义在 `models.py`（根目录）与 `src/models.py`（镜像，二者内容完全一致）。
> 训练/扩散流程在 `diffusion/`（或 `src/diffusion/`）；采样在 `samplers.py`。
> 关于训练配置与超参，见 `docs/training/`；关于数据与 label 映射，见 `docs/data/`。

---

## 1. 概述

DiT-MCCD 用 **Diffusion Transformer (DiT)** 在 **VAE 潜空间** 中做条件生成：

```
输入条件： (calligrapher 书家, character 汉字)   ← 2Cond 主线
输出：     256×256 书法字图像（经 VAE 解码）
```

- **骨干**：Transformer 直接对 VAE latent 做去噪（latent diffusion），而非像素空间扩散。
- **条件**：离散类别 ID（书家 / 字 / 书体），经 `LabelEmbedder` + 融合模块注入。
- **调制**：adaLN-Zero（自适应 Layer Norm，零初始化门控）。
- **学习目标**：`learn_sigma=True` 时输出 2× 通道（ε + σ），即预测噪声与方差。
- **引导**：Classifier-Free Guidance (CFG)，推理时同时跑条件 / 无条件两路再外推。

模型尺寸沿用 facebookresearch/DiT 的 S/B/XL 命名与配置，使其 body 能直接加载官方 ImageNet 预训练权重做 LoRA 微调。

---

## 2. 模型变体 / Model Variants

变体构造函数见 `models.py:940` 起（`DiT_2Cond_*`）与 `models.py:621` 起（`DiT_3Cond_*`），
注册表 `DiT_2Cond_models`（`models.py:953`）、`DiT_3Cond_models`（`models.py:636`）。

| 模型 | hidden | depth | heads | patch | params | input_size | tokens | 典型 VAE |
|-------|-------:|------:|------:|------:|-------:|-----------:|-------:|----------|
| DiT-2Cond-S/2  | 384  | 12 | 6  | 2 | ~33M  | 32 (f8 VAE) | 16×16=256 | sd-vae-ft-ema |
| DiT-2Cond-S/4  | 384  | 12 | 6  | 4 | ~42M  | 64 (f4 VAE) | 16×16=256 | kl-f4 |
| DiT-2Cond-B/2  | 768  | 12 | 12 | 2 | ~132M | 32 (f8 VAE) | 16×16=256 | sd-vae-ft-ema |
| DiT-2Cond-XL/2 | 1152 | 28 | 16 | 2 | ~673M | 32 (f8 VAE) | 16×16=256 | sd-vae-ft-ema |
| DiT-3Cond-S/2  | 384  | 12 | 6  | 2 | ~34M  | 32 (f8 VAE) | 16×16=256 | sd-vae-ft-ema |

> **token 数恒为 256**：`tokens = (input_size / patch)²`。各变体刻意保持 token 数一致，
> 让 S/B/XL 仅在“每 token 的宽度与深度”上不同，便于横向对比与权重迁移。

### 关键洞察：S/4 + f4 VAE（同算力，3× 潜信息）

| 配置 | VAE | latent 形状 | latent 数值量 |
|------|------|------------|--------------:|
| DiT-2Cond-S/2 | f8 (sd-vae) | (4, 32, 32) | 4 × 32 × 32 = **4,096** |
| DiT-2Cond-S/4 | f4 (kl-f4)  | (3, 64, 64) | 3 × 64 × 64 = **12,288** |

两者都产生 **256 个 token**（32/2=16，64/4=16），因此 **Transformer 计算量相同**，
但 f4 路径的 latent 携带 **3× 的信息量**（12,288 vs 4,096）。这是用 kl-f4 替换 sd-vae 的核心理由：
不增加 DiT 主干开销，仅通过更“细”的 VAE latent 提升可还原的笔画细节。见 `configs/s7_klf4_top30_diffonly.json`。

---

## 3. 架构类 / Architecture Classes

### 3.1 共享基础件

所有变体共用同一套嵌入与块结构（`models.py`）：

| 组件 | 位置 | 作用 |
|------|------|------|
| `TimestepEmbedder` | `models.py:28`  | 正弦 timestep → 向量（频率编码 + 2 层 MLP/SiLU） |
| `LabelEmbedder`   | `models.py:68`  | 类别 ID → 向量；内建 CFG null token 与 label dropout |
| `DiTBlock`        | `models.py:103` | adaLN-Zero 块：`norm→attn→gate` + `norm→mlp→gate` |
| `FinalLayer`      | `models.py:127` | adaLN 调制 + Linear → `patch²×out_ch`，再 `unpatchify` |
| `get_2d_sincos_pos_embed` | `models.py:279` | 冻结的 2D sin-cos 位置编码（`requires_grad=False`） |

**adaLN-Zero 调制**（`DiTBlock.forward`，`models.py:120`）：
条件向量 `c` 经 `adaLN_modulation` 切成 6 段
`(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)`，
对 attention 与 MLP 各做 `modulate`（scale+shift）并以门控残差相加：

```
x = x + gate_msa * Attn(modulate(LN(x), shift_msa, scale_msa))
x = x + gate_mlp * MLP(modulate(LN(x), shift_mlp, scale_mlp))
```

门控权重初始化为 **0**（`initialize_weights` 中 `adaLN_modulation` 权重/偏置清零），
即训练初期每个块近似恒等，保证稳定。

### 3.2 `DiT`（基类，单条件） — `models.py:147`

源自 facebookresearch/DiT，改造为 latent diffusion：

- 单条件：`y_embedder = LabelEmbedder(num_classes, ...)`（ImageNet 分类式）。
- 条件融合：`c = t_emb + y_emb`（ timestep 与类别向量直接相加）。
- `learn_sigma=True` → `out_channels = in_channels × 2`（输出 ε 与 σ）。
- 提供标准 `DiT_models`（S/B/L/XL × /2/4/8，`models.py:370`）。
- 该类是 2Cond/3Cond 的结构蓝本，训练主线不直接使用它。

### 3.3 `DiT_2Cond` — `models.py:648`

**两条件**：`calligrapher`（风格）+ `character`（内容）。这是当前推荐主线。
Body 尺寸与官方 DiT-XL 对齐，便于加载 `DiT-XL-2-256x256.pt` 预训练 body 后做 LoRA。

#### 三种条件融合模式（`condition_fusion`）

| 模式 | 结构 | 是否支持组合泛化 | 说明 |
|------|------|:---:|------|
| **`factorized_add`**（推荐） | 独立低维 embedding，各自 `LayerNorm+Linear` 投影到 hidden 后 **相加**（除以 √2 归一） | ✅ | 见 `models.py:700` |
| `xl_highdim` | 高维 concat `(callig_d + char_d)` → MLP → hidden；乘可学习 `y_scale`（init 0.05） | 部分 | 见 `models.py:715` |
| `legacy` | 联合 MLP `hidden×2 → hidden`（原始方案） | ❌ | 见 `models.py:737` |

**`factorized_add` 详解**（推荐，配置见 `configs/exp_s6_top6_diffonly.json`）：

```
e_callig = LabelEmbedder(num_calligraphers, callig_embed_dim=128)   # 128-D
e_char    = LabelEmbedder(num_characters,   char_embed_dim=256)     # 256-D
y_emb = ( LayerNorm+Linear(e_callig) + LayerNorm+Linear(e_char) ) / sqrt(2)
```

- 两个因子各自独立查表 + 独立投影，**相加**后才进入 DiT。
- 组合泛化：未见过的 `(callig, char)` 对可由各自训练充分的边际 score 组合推断，
  而非依赖一张完整的联合表做 memorization。
- 配套 **可控 4-way 条件 dropout**（`models.py:842`）：
  - `drop_all`（prob `cond_drop_all_prob`）→ 全无条件（CFG 基准）
  - `drop_one & which`（prob `cond_drop_one_prob`）→ 仅丢一个因子，学单因子边际 score
  - 其余 → full（双因子联合）
- 推荐配比 `cond_drop_all_prob=0.05, cond_drop_one_prob=0.25`
  → 约 full 70% / 单因子 25% / 无条件 5%。

**`xl_highdim`**：高维拼接后过 MLP，条件向量由训练目标自行建立语义；
`y_scale`（可学习，init 0.05）让 `y_emb` 初始幅度接近 `t_emb`，保证 adaLN 早期稳定。
ImageNet 预训练的 adaLN/final_layer 是“分类→调制”耦合、与书法正交，训练时会重置从头学。

**`legacy`**：原始联合 MLP，向后兼容旧 checkpoint；不支持组合泛化。

#### 可选组件

| 组件 | 开关 | 位置 | 作用 |
|------|------|------|------|
| **glyph_cond**（标准字形条件） | `use_glyph_cond=True` | `models.py:768` | 标准字形 latent 经独立 `Conv2d` 编码成 token，**加到** patch token 上（见 §4.2） |
| **skel_head**（骨架辅助头） | `skel_head_enabled=True` | `models.py:756` | 从 `final_layer` 前的 block 特征并行解出 1×32×32 latent 骨架预测，仅训练时用 |

### 3.4 `DiT_3Cond` — `models.py:382`

**三条件**：`calligrapher` + `script`（书体）+ `character`。
与 `DiT_2Cond` 同构，把“两因子相加”换成“三因子相加”（除以 √3 归一，`models.py:549`）。

- 融合模式：`legacy` / `factorized_add`（3 因子版）。
- **已弃用**：实测 `script` 与 `calligrapher` 严重混杂，
  互信息 `I(calligrapher; script) = 1.527 bits`（约解释 68.3% 的 script 熵），
  script 作为独立第三因子提供的是重复而非新增信息。
  详见 `docs/design/2026-08-15-sparse-compositional-calligraphy-dit.md` 与
  `docs/experiments/2026-08-15-factorized-3cond.md`。
- 当前主线已从 3Cond 收敛到 **2Cond**（把 `script×character` 合并为单一 glyph 类）。

---

## 4. 条件细节 / Conditioning

### 4.1 `LabelEmbedder` — `models.py:68`

- **结构**：`nn.Embedding(num_classes + 1, hidden)`，多出的一行是 **CFG null token**（`use_cfg_embedding=True` 时）。
- **训练时 dropout**（`token_drop`，`models.py:80`）：
  - 以 `cond_drop_all_prob` 概率把整批样本的**全部**因子置为 null token → 无条件（uncond）。
  - 以 `cond_drop_one_prob` 概率随机丢**一个**因子 → 部分条件（学单因子边际 score）。
  - 2Cond 中这两种概率由外层 `DiT_2Cond.forward` 统一控制（`models.py:842`），避免意外的“同时丢两因子”。
- **推理时 CFG**（`forward_with_cfg`，`models.py:915`）：每个样本复制两份，
  前半用真实条件 ID，后半用 null token（`num_classes`），各跑一次 forward，
  在 ε 子空间外推：

  ```
  eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
  ```

  `cfg_scale` 默认 4.0（见各 config 的 `eval_cfg`）。σ 通道不做 CFG（直接用条件路输出）。

### 4.2 标准字形条件（可选，`w_glyph_cond`）

当 `use_glyph_cond=True` 时，向 patch token 注入“标准字形”作为内容先验（`models.py:828`）：

```
x = x_embedder(x) + glyph_scale * glyph_embedder(g) + pos_embed
```

| 元素 | 说明 |
|------|------|
| 标准字形 latent `g` | 标准字体（楷/隶）经 sd-vae 编码的 `(4,32,32)` latent；库在 `std_glyph_latent/{kai,li}/U+XXXXX.npy` |
| `glyph_embedder` | 独立 `Conv2d(in_channels→hidden, kernel=patch, stride=patch)` → 展平为 token 序列（`models.py:771`） |
| `glyph_scale` | 可学习标量，init `glyph_scale_init=0.4`（`models.py:763`） |
| 查询 | `src/glyph_latent.py` 的 `GlyphLatentLookup`，按 `(script_id, char)` 返回 latent |

- **独立投影而非复用 `x_embedder`**：保证 `g` 的编码可学习、norm 可控；
  XL LoRA 模式下 `x_embedder` 被冻结，复用会锁死字形信号。
- CFG 时 `g` 始终全给（��半都用真实 `g`）：字形内容是正条件，CFG 只强化 calligrapher 风格（`models.py:927`）。

---

## 5. VAE 集成 / VAE Integration

DiT 在 **VAE 潜空间** 运行，不直接处理像素。图像经 VAE 编码为 latent（已乘 `scaling_factor` 归一到 std≈1），
DiT 去噪后再除以 `scaling_factor` 解码回像素。

| VAE | 下采样 | latent_channels | latent_size (256² 输入) | latent 数值量 | scaling_factor | 配置示例 |
|-----|:------:|:--------------:|:----------------------:|:-------------:|:--------------:|----------|
| sd-vae-ft-ema (f8) | 8 | 4 | (4, 32, 32) | 4,096 | 0.18215 | `exp_s6_top6_diffonly.json` |
| kl-f4 (f4)         | 4 | 3 | (3, 64, 64) | 12,288 | 0.102079 | `s7_klf4_top30_diffonly.json` |

**尺寸推导**（`src/train.py:174`）：

```
input_size      = image_size // vae_downscale        # 256/8=32 或 256/4=64
in_channels     = latent_channels                     # 4 或 3  (传入 x_embedder)
out_channels    = latent_channels × 2  (learn_sigma)  # 8 或 6
                 或 latent_channels     (不学 sigma)
```

- **编码**：`z = vae.encode(x).latent_dist.sample() * scaling_factor`（见 `tools/vae/encode_latents_klf4.py`、`src/eval_auto.py:115`）。
  latent 预编码后缓存为 shard（`final_latents` / `final_latents_f4`），训练直接读缓存。
- **解码**：`x = vae.decode(z / scaling_factor).sample`（见 `sample.py:65`、`src/eval_auto.py:279`）。
- `scaling_factor` = `1 / std(latent_samples)`（见 `tools/vae/estimate_scaling_factor.py`），
  sd-vae 对应 std≈5.49，故 0.18215。

> VAE 选择与 patch 联动：f8 → `patch=2`、`input_size=32`；f4 → `patch=4`、`input_size=64`。
> 二者 token 数同为 256（见 §2 关键洞察）。

---

## 6. 前向与采样流程

### 6.1 训练前向（`DiT_2Cond.forward`，`models.py:818`）

```
x = x_embedder(x) + [glyph_scale * glyph_embedder(g)] + pos_embed   # (N, 256, D)
t_emb = t_embedder(t)                                                 # (N, D)
y_emb = fuse(callig, char)                                            # 见 §3.3 融合模式
c = t_emb + y_emb
for block in blocks: x = block(x, c)                                  # adaLN-Zero × depth
[可选] skel_pred = skel_head(x)                                       # 辅助骨架预测
x = final_layer(x, c); x = unpatchify(x)                              # (N, out_ch, H, W)
```

- 支持 `use_checkpoint`（梯度检查点，省显存）。
- `return_intermediate_layer=i` 可取第 `i` 个 block 的 patch 特征，用于 REPA 表示对齐损失。
- `skel_head` 启用时返回 `(主输出, skel_pred)`，CFG 只取主输出。

### 6.2 CFG 采样（`forward_with_cfg`，`models.py:915`）

1. 复制 batch：前半条件、后半无条件（null token）。
2. 一次 forward 得到 `(2B, out_ch, H, W)`。
3. 在 ε 子空间（前 `in_channels` 通道）做 CFG 外推，σ 通道不变。
4. 返回前 `B` 个样本，交由扩散模块（`diffusion/`）的 DDIM/DDPM 采样循环逐步去噪；推理脚本见 `sample.py`。

> 注意：`samplers.py` 中的 `DistributedFactorBalancedSampler` 是训练期的**长尾数据采样器**（按字符/书家频率做温度逆采样），并非 DDIM 采样器。

### 6.3 扩散过程

- DDPM/DDIM 实现在 `diffusion/`（`gaussian_diffusion.py`、`respace.py`、`timestep_sampler.py`）与 `src/diffusion/`，提供 `p_sample_loop` / `ddim_sample_loop` 等采样循环。
- `create_diffusion`（`src/diffusion/__init__.py`）默认 `noise_schedule="linear"`、1000 步、`learn_sigma=True`、`ModelVarType.LEARNED_RANGE`。
- 推理入口 `sample.py` 调用扩散模块的采样循环生成 latent，再经 VAE 解码为像素。
- 结构性辅助损失（Canny / skeleton，可选）在 latent 空间施加，见 `latent_structure.py`。

---

## 7. 关键文件 / Key Files

| 文件 | 作用 |
|------|------|
| `models.py` / `src/models.py` | 全部模型定义（`DiT` / `DiT_2Cond` / `DiT_3Cond` + 嵌入件 + 配置表） |
| `diffusion/` / `src/diffusion/` | DDPM/DDIM 扩散过程（`gaussian_diffusion.py` 的 `p_sample_loop`/`ddim_sample_loop`、`respace.py`、`timestep_sampler.py`） |
| `lora.py` / `src/lora.py` | LoRA 注入（可选，用于微调；`inject_lora` 替换 qkv/proj/fc1/fc2） |
| `latent_structure.py` / `src/latent_structure.py` | 结构性损失：`LatentStructureProbe`（冻结探针）+ `LatentStructureLoss`（Canny 梯度 + skeleton BCE-Dice） |
| `samplers.py` / `src/samplers.py` | `DistributedFactorBalancedSampler`（训练期长尾温度采样，非 DDIM 采样器） |
| `sample.py` | 推理采样入口：加载模型 + VAE，跑扩散采样循环并解码出图 |
| `src/glyph_latent.py` | 标准字形 latent 查询（`GlyphLatentLookup`，楷/隶 → `(4,32,32)`） |
| `configs/*.json` | 训练/实验配置（含 VAE、融合模式、dropout、LoRA 等参数） |

---

## 附：相关文档

- 设计与决策：`docs/design/2026-08-15-sparse-compositional-calligraphy-dit.md`
- 因子化实验：`docs/experiments/2026-08-15-factorized-3cond.md`、`docs/experiments/2026-08-15-v3a-2factor-glyph.md`
- latent vs pixel 结构损失：`docs/experiments/2026-08-17-latent-vs-pixel-struct.md`
- 历史文档：`docs/legacy/`（DOCUMENTATION / TRAINING / INFERENCE 等，部分已被本文取代）
