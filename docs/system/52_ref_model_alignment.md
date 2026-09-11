# 52. ref（moyi / Moyun）主模型设计精读 · 哪里好 · 我们要对齐什么

> 对应实现：`ref/moyi/moyun/moyun_2.py`（模型）、`ref/moyi/train_moyun2.py`（训练）。
> 本文只讨论**主模型设计**；aux 目标通道机制见 51 号。
> 起点核对：ref 12ch run = `--model test-models-nofeature-12channel`
> = depth24 / hidden1024 / heads16 / patch2 / in_channels12 / learn_sigma=True / no RoPE。

---

## 1. ref 模型逐块精读

### 1.1 整体：标准 DiT + adaLN-Zero（无任何附加注入）

```
x(12ch latent) ─ PatchEmbed(p=2) ─ +sincos pos ─┐
t ─ TimestepEmbedder(sinusoid+2×Linear) ────────┤ c = t_emb + y_emb
y=(callig,font,char) ─ LabelEmbedder(3×Embedding→concat→Linear) ─┘
                                                │
  24 × MoyunBlock(adaLN-Zero):  x = x + gate·Attn(modulate(norm1(x)))   ← shift/scale/gate 额外 ×1.2
                                x = x + gate·MLP (modulate(norm2(x)))
                                                │
                        FinalLayer: modulate(norm_final(x), shift, scale) → Linear → unpatchify
```

### 1.2 Block（`MoyunBlock`）

- **adaLN-Zero**：`adaLN_modulation = SiLU → Linear(h, 6h)`，chunk 成
  `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp`（标准 DiT）。
- Transformer 路径：
  ```python
  shift/scale/gate 各 *= 1.2                         # ← ref 的调参 trick (L404-409)
  x = x + gate_msa * Attention(modulate(norm1(x)))
  x = x + gate_mlp * MLP(modulate(norm2(x)))
  ```
- 可选 `use_mamba=True`（VisionMamba2 替 Attention）/`use_kan=True`（FasterKAN 替 MLP）/
  `use_stroke=True`（多一个 stroke cross-attn 子层）——**本 run 全部关闭**。
- `norm = LayerNorm(elementwise_affine=False)`（调制只来自 c，不带自己的仿射）。

### 1.3 Embedding 与条件

- `TimestepEmbedder`：sinusoid(256) → Linear→SiLU→Linear，标准 DiT。
- `LabelEmbedder`：**三张 Embedding 表（callig/font/char，各 4793 类）→ concat(3h) → Linear(h)**，
  带 token_drop（CFG 用）。
- `c = t_emb + y_emb`，**仅此一个条件向量**，无骨架/无风格 token/无 ControlNet。

### 1.4 FinalLayer / 初始化

- `FinalLayer = adaLN_modulation(SiLU+Linear(h,2h)) + modulate(norm_final) + Linear(h, p²·C)`。
- `initialize_weights`：`adaLN_modulation[-1]` 与 `final_layer.*` **全部置零**（标准 DiT）；
  x_embedder 用 xavier；label/t 表 N(0,0.02)。

### 1.5 扩散/推理

- DDPM（1000 步线性），eps 预测，`learn_sigma=True` → 输出 24ch。
- `forward_with_cfg`：**CFG 只作用在前 3 个通道**（RGB 遗留写法），其余 passthrough。
- 训练：AdamW lr 1e-4 恒定、batch 256、无 EMA? (有 EMA)、no early stop。

---

## 2. 哪里好（值得学的点）

| # | ref 的设计 | 为什么好 | 我们的证据 |
|---|---|---|---|
| 1 | **极简主干**：所有条件只进 `c`（adaLN-Zero），没有 ControlNet/xattn/风格 token/骨架注入 | 参数全给主干；不存在"外挂被门控忽略"问题 | 48 号：xattn 推理消融 −0.24（被用），但**训练期增量 ≈0**（doc 50：Sp adaLN4 0.518 vs xattn12 0.518@80k）；外挂 ±0.002 死重 |
| 2 | **adaLN-Zero 全零初始化** | 训练初期恒等，深网（24 层）稳定 | 我们同款做法（block/final 置零）已验证 |
| 3 | **结构进"目标"而非"条件"**：edge/skel 作为额外 latent 通道共同去噪 | 结构成为必须重建的对象（不可忽略），且推理不需要它 | 48 号 D1：输入层 token-add 是骨架主通路；条件在深网易被稀释 |
| 4 | **字体（font/script）label** | 楷/行/隶 的结构风格先验由独立 embedding 提供 | 我们只有书家 label，csv 里其实有 `script_id`（未用） |
| 5 | **门控 ×1.2** | 轻微放大 adaLN 调制幅度（便宜、可试） | 我们未试 |
| 6 | **3 表拼接的单类条件** | 把"人×字体×字"的笛卡尔积压成 3 个边际 embedding 再融合 | 与我们 factorized_add 思想一致（我们=书家+骨架） |

---

## 3. 我们（DiT_2Cond）与 ref 的差异

| 维度 | ref Moyun | 我们 DiT_2Cond |
|---|---|---|
| 宽度/深度 | h1024 / d24（~500M） | h384/d12（33M，S/2）；h512/d12（59M，Sp） |
| Block | LayerNorm + 标准 Attention/Mlp + adaLN-Zero | **RMSNorm + SwiGLU + QK-Norm + RoPE**（v2 现代化） |
| 位置编码 | sincos 加到 x | RoPE（可选 sincos） |
| 条件 | `c = t + y`（3 label） | `c = t + callig(41 冻结表)`，外加 **std-skel-g 注入**（输入 token-add + 逐层注入） |
| 骨架角色 | **只在目标通道**（不作为条件） | **既作条件**（std skel g）**又作目标**（aux，新加） |
| 注入方式 | 无（只有 adaLN-Zero 本身） | 输入 token-add + `ZeroAdaLNInjection` 逐层（xattn 已删） |
| 扩散 | DDPM eps，learn_sigma=True | flow velocity，learn_sigma=False |
| CFG | 仅前 3 通道 | 前 in_channels（4 或 12） |
| 数据/规模 | 1.9M 图 / batch256 / lr1e-4 | 28k(113k aug) / batch192 / lr5e-5 |

---

## 4. 对齐清单（按优先级）

1. ✅ **删除 xattn 与风格 token**（已做）：主干回到 `输入 token-add + ZeroAdaLNInjection`，
   注入与 adaLN-Zero 同族（`out = x*(1+s)+t`，zero-init）。
2. **g 条件的"ref 化"**：ref 没有 g；我们要保留 g 时，最贴近 ref 的形式是
   **把 g 编码成一个向量加进 `c`**（`c = t + y_callig + y_skel`），而不是 256 个 token 逐层注入。
   → 待定：保留逐层 adaLN 注入（现方案）还是 g→c（更 ref）。
3. **aux 目标对齐**（见 51 号）：12ch latent 等权 MSE；**CFG 只作用于图像通道**（ref 只对前 3ch CFG）。
4. **加 script/font label**：csv 有 `script_id`，加一张 Embedding（3 表变 2 表）即可复刻 ref 的字体先验。
5. **门控 ×1.2**：一行改动，可作小消融。
6. **DDPM vs flow**：严格对齐需切 `diffusion_type=ddpm`；否则在结论里注明口径。
7. **规模**：ref ~500M/1.9M 图 vs 我们 33M/113k。若 aux 机制的收益依赖容量，需要放大/延长。

---

## 5. 下次实验的最小对齐配方（建议）

```
DiT-2Cond-S/2 (33M) 或 Sp/2 (59M)
条件: c = t_emb + callig_emb(41冻结)            # 无 char 向量
骨架: 输入层 token-add (glyph_scale) + ZeroAdaLNInjection ×4 (glyph_inject_layers=4)
目标: 12ch = cat(image(4), skel(4), canny(4))    # aux=GT 实例结构 latent, 等权 MSE
扩散: flow (注明与 ref DDPM 的差异)
数据: fame_e 113,540 (或 fame3 28k)
batch 192, lr 5e-5 cosine, 200k/133k, 从零
```

**执行顺序**：① 先 300~500 步 **梯度 debug**（图像 4ch vs aux 8ch 的 loss/梯度占比），
据此定 `aux_loss_weight`；② 再跑正式预训练。
