# 模型设计与预训练方案（2026-08-31）

> 回答两个问题：
> 1. 「条件信息量差距」这个诊断，解决了吗？
> 2. 模型设计与预训练方法，有更好的方案吗？

---

## 0. 先修正我之前的说法

我在文档 19 里写：

> base 0.50 vs ControlNet 0.80 是**条件信息量**的差距（~162 有效维 vs ~4258 维，约 120 倍）

**这个说法只对了一半，而且误导了重点。**

正确的表述是：

> **不是「信息量不够」，而是「信息进不去」。**
> 全局向量（AdaLN）在架构上就无法传递空间结构——
> 给它 4 万维也没用，因为 AdaLN 对每个 token 用**同样的**调制参数。

证据：

| | 条件维度 | 注入方式 | SSIM |
|---|---:|---|---:|
| base | ~162（全局向量） | AdaLN（空间无关） | 0.47–0.50 |
| **glyph_cond（当前实现）** | **4096（空间图）** | **输入层 token-add，仅 1 次** | **待测（s23 在跑）** |
| ControlNet | 4096（空间图） | **ZeroAdaLN 逐层注入 ×12** | 0.73–0.80 |

**glyph_cond 和 ControlNet 拥有相同的信息量（都是 4×32×32 空间图），
但注入方式差 12 倍。** 这才是关键变量。

---

## 1. 当前实现的三个设计问题

### 问题 1：空间条件只在输入层注入一次（**最严重**）

`src/model/dit.py:950-953`：

```python
if self.use_glyph_cond and self.glyph_embedder is not None and g is not None:
    g_tok = self.glyph_embedder(g).flatten(2).transpose(1, 2)  # (N,256,D)
    x = x + self.glyph_scale * g_tok        # ← 只在输入层，一次
```

之后 x 经过 **12 层 Transformer**。单层注入的信号在残差流中被反复稀释：
每层 block 的输出 `x + block(x)` 都会把 g 的贡献相对削弱。

对比 ControlNet（`controlnet.py:70`）：

```python
for i, block in enumerate(self.blocks):
    x = block(x, c)
    x = self.injections[i](x, ctrl_feats[i])   # 每层都注入
```

且用的是 **adaLN 式调制** `x*(1+s) + t`，比加法 `x + feat` 更强——
既能增强也能抑制残差流。

### 问题 2：条件职责混淆

当前 `factorized_add`：

```python
c = t_embedder(t) + callig_proj(y_callig) + char_proj(y_char)
```

书家（风格）和字（结构）都挤在同一个全局向量里，一起喂给 AdaLN。

**AdaLN 是空间无关的**：它对 1024 个 token 施加**相同**的 scale/shift/gate。
所以它只能控制「整体风格」，**无法指定「第 37 个 token 该画一横」**。

把字形结构压进 AdaLN 向量，本质上是要求模型做一件架构上做不到的事。

### 问题 3：字的表示方式丢掉了空间信息

```
字 ID → DINO CLS（384 维，有效秩 34）→ char_proj → AdaLN
```

DINO CLS 是**全局摘要**——VIT 把 16×16 个 patch 压缩成一个向量，
空间信息在这一步就没了。之后再怎么投影都找不回来。

而「字」最本质的信息恰恰是**空间结构**。

---

## 2. 方案

### 方案 A：glyph 条件逐层注入（**优先级最高，改动小**）

把 `w_glyph_cond` 从单层 token-add 改成逐层 ZeroAdaLNInjection，
**与 ControlNet 完全对齐**。

```python
# __init__
if self.use_glyph_cond:
    self.glyph_embedder = nn.Conv2d(in_channels, hidden_size,
                                    kernel_size=ps_, stride=ps_, bias=False)
    # 逐层注入（复用 controlnet.ZeroAdaLNInjection）
    self.glyph_injections = nn.ModuleList([
        ZeroAdaLNInjection(hidden_size, mode="modulate")
        for _ in range(len(self.blocks))
    ])

# forward
g_tok = self.glyph_embedder(g).flatten(2).transpose(1, 2)   # (N, 256, D)
for i, block in enumerate(self.blocks):
    x = block(x, c)
    x = self.glyph_injections[i](x, g_tok)                   # 每层调制
```

**参数增量**：12 × (384 × 768) ≈ **3.5M**（总参数 46.5M → 50M，+7.5%）

**优点**：
- 零初始化 → 恒等初始化，不会破坏已有训练
- 与 ControlNet 同一套机制，正确性已被验证（0.80）
- 可直接复用 `controlnet.py` 的 `ZeroAdaLNInjection`

**风险**：显存。s23 当前已 19.63G/19.63G 占满，
加 12 层注入需减小 batch（192 → 128）或开 gradient checkpointing。

### 方案 B：条件职责分离（**架构正统做法**）

```
书家 ID  ──→ callig_proj ──→ AdaLN 调制      （风格，空间无关，保持现状）
字 ID    ──→ 标准字形 latent ──→ 逐层空间注入  （结构，空间对齐，改这里）
时间步   ──→ t_embedder ────→ AdaLN 调制      （保持现状）
```

即：**风格走全局标量通路，结构走空间图通路。**

这是扩散模型的通行设计（文本走 cross-attention /空间条件，风格走 AdaLN）。
当前把两者混在一个向量里，是历史遗留。

**注意**：方案 A 是方案 B 的子集——先把注入改对，职责自然就分开了。

### 方案 C：预训练策略——渐进式条件 dropout

**问题**：若 glyph 条件太强，模型会「复制」标准字形，失去书法风格；
若太弱，又学不到结构。

**方案**：训练时以概率 p 随机 drop 标准字形条件：

```
p = 0.0  → 纯条件生成（复刻标准字形）
p = 0.15 → 大部分时候有条件，偶尔没有（推荐）
p = 0.5  → 平衡
```

好处：
1. 模型既学会用条件，也保留无条件生成能力
2. 推理时可通过条件强度插值，控制「标准字形 vs 书法风格」的权衡
3. 与 CFG 天然配合（drop 的样本就是 uncond 分支）

**实现**：已有 `cond_drop_all_prob` / `cond_drop_one_prob` 机制，
加一个 `glyph_drop_prob` 即可。

### 方案 D：字形作为输入通道（备选）

```python
x = torch.cat([x, g], dim=1)      # (N, 4, 32, 32) → (N, 8, 32, 32)
x_embedder = PatchEmbed(in_channels=8, ...)
```

**优点**：信息在每一层都通过残差流自然传播，无需额外注入模块。

**缺点**：
- 需要改 `x_embedder` 的输入通道，与已有 ckpt 不兼容（需从头训）
- CFG 时 g 无法 drop（因为它是输入的一部分）
- 增加 4 通道 → patchify 后 token 数不变但每 token 维度不变，只是 embed 层变宽

**评价**：干净但破坏兼容性，优先级低于 A。

---

## 3. 预期效果与判读

s23（单层注入）正在跑，第一个 eval 点在 step 2500。

| 结果 | 判读 | 下一步 |
|---|---|---|
| SSIM 明显 > 0.467（如 ≥0.52） | 单层注入就够，架构问题没那么严重 | 加方案 C（dropout 调优） |
| SSIM 略升（0.48–0.52） | 信息进得去但被稀释 | **实施方案 A（逐层注入）** |
| SSIM 基本不动（≈0.467） | 单层注入几乎无效 | 立即实施方案 A |

**理论预期**：标准字形 latent 提供「字形结构」，书家条件提供「风格」，
二者解耦。因此即使 SSIM 不如 ControlNet 的 0.80（后者是复刻），
**字形正确率应有明显提升**——这才是我们真正要的。

注意用 **SkelIoU / 字形正确率** 判读，而不是只看 SSIM：
SSIM 对「结构对但风格不同」惩罚很重，会低估该方案的价值。

---

## 4. 预训练方法的改进建议

### 4.1 当前做法

从零训 s21：随机初始化 → 51k fame 数据 → flow matching → early-stop @40k

**问题**：从零开始，模型要同时学①字形结构 ②书法风格 ③扩散去噪，三者耦合。

### 4.2 建议：两阶段预训练

**阶段 1（结构学习）**：
- 用标准字形 latent 作为强空间条件（drop_prob = 0）
- 目标：让模型快速学会「字的空间结构」
- 数据：可用大规模标准字体渲染数据（不限于 fame 的 51k）

**阶段 2（风格学习 + 解耦）**：
- 渐进提高 glyph_drop_prob（0 → 0.15 → 0.3）
- 目标：让模型在条件缺失时也能生成，并学会书家风格
- 数据：fame 51k（44 书家）

这与 classifier-free guidance 的训练范式一致，且已被 ControlNet 验证。

### 4.3 数据层面

- **扩大书家数**：fame 44 书家 vs 全量 1853 书家。
  注意书家数增加会让 SSIM 下降（见 `difficulty_summary.csv`），
  但**泛化能力提升**——需要明确取舍。
- **覆盖缺口**：草 10,910 / 篆 2,977 样本无标准字体（v2 字典 0% 覆盖）。
  对这些样本，glyph 条件会给零张量。
  建议：**显式标记缺失样本**，或用「同字其他书体」借代。

---

## 5. 关于 std-skel 失败的新解释

之前归因于「冗余条件被门控」。补充一层：

**std-skel 用的是标准字体的骨架（1px 细线），与书法 GT 的笔画差异大。**
- 条件：楷书骨架（均匀细线）
- 目标：草书 GT（笔势、粗细、连带）

即使逐层注入再好，**条件本身与目标的匹配度**就决定了上限 ~0.52。

而 GT 骨架是从目标图抽的，匹配度 100% → 上限 0.80。

**标准字形 latent（g）介于两者之间**：
- 比骨架信息完整（含笔画粗细，不只是拓扑）
- 但风格仍是标准字体，不是书法

**所以 g 的预期是「结构正确 + 风格不匹配」**，
字形正确率应显著提升，SSIM 提升幅度中等。

---

## 6. 实施清单

| 优先级 | 改动 | 文件 | 成本 |
|---|---|---|---|
| **1** | 等 s23 首个 eval（step 2500） | — | 等待 |
| **2** | 方案 A：glyph 逐层注入 | `src/model/dit.py` | ~40 行 + 减 batch |
| **3** | 方案 C：glyph_drop_prob | `src/train/train.py` + `dit.py` | ~15 行 |
| 4 | 方案 B：职责分离（A 之后自然达成） | — | — |
| 5 | 方案 D：输入通道拼接 | `dit.py` | 破坏 ckpt 兼容 |

**下一步**：等 s23 的 step 2500 结果 → 按上表判读 → 实施方案 A。

---

## 7. 一句话总结

> **信息量从来不是瓶颈，注入方式才是。**
> 给模型 4096 维的空间条件，却只在输入层加一次然后让它穿过 12 层 Transformer，
> 等于把答案写在第一页然后让人翻完整本书再回答。
> ControlNet 做对的不是「提供了更多信息」，而是「每层都提醒一遍」。
