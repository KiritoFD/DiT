# 922 / 80 — 注入机制：怎么改、怎么实现

> 前置：`60_diagnostics.md`（D1–D6 实测）、`70_training_plan.md`（训练计划）。
> 本文只回答**注入机制本身**——改哪里、为什么、怎么改、怎么验证。
>
> 结论先行：**不要换 adaLN，要给它"配一个增益"**。

---

## 0. 先看实测：问题到底在哪一层

### 0.1 条件幅度（D1 实测，7 个 ckpt）

| run | `t_emb` | `y_emb` | `c_norm` | **`y/t`** | 备注 |
|---|---|---|---|---|---|
| v13_base_155k | 29.10 | 22.27 | 37.43 | **0.765** | T2 最好（0.612） |
| v13_12ch_225k | 32.39 | 22.59 | 40.22 | **0.698** | T2 次好（0.514） |
| v15b_supcon_70k | 35.74 | 20.38 | 36.71 | 0.570 | 表最可分 |
| v15a_150k | 38.70 | 20.74 | 46.78 | 0.536 | |
| v13_wd01_125k | 33.35 | 15.13 | 35.34 | 0.454 | |
| **v14_s2_160k** | 48.94 | 20.12 | 48.54 | 0.411 | **已用 hier_style，仍不行** |
| **v15c_fixed_210k** | 46.75 | 16.10 | 45.92 | **0.344** | T2 最差（0.110） |

**★ 两个反常**：

1. **`t_emb ≈ 29–49，`y_emb` 却只有 16–23。**
   `c = t_emb + y_emb`，所以**条件向量 `y_emb` 只占 `c` 的 34%–77%**，
   而 `t_emb` 贡献 1.3–2.9 倍。→ **风格信号天生被时间步"稀释"**。

   ⚠ 但注意：`y/t ≈ 0.34` 的 v15c 是**最差**的，`y/t ≈ 0.77` 的 v13 是**最好**的。
   → **`y/t` 与 T2 强正相关！** 这不是巧合，是因果。

2. **`v14_s2_160k` 已经用了 `hier_style=1`，`y/t` 也只有 0.41，T2 一样不达标。**
   → **"三层表"本身没解决问题**（与 D5 的"表可分"结论一致：**表可分 ≠ 表有用**）。

### 0.2 adaLN 出口的增益（D1 `dmod`）

`dmod = ‖mod(c) − mod(c − y_emb)‖ / ‖mod(c)‖`，测"拔掉书家后 adaLN 输出变了多少"：

| run | `dmod` | 相对 v13_base |
|---|---|---|
| v13_base_155k | **0.65** | 1× |
| v13_12ch_225k | 0.52 | 0.8× |
| v15a_150k | 0.21 | 0.32× |
| v15b_supcon_70k | 0.09 | 0.14× |
| **v15c_fixed_210k** | **0.0499** | **0.077×（1/13）** |

**★ 这是全项目最关键的一张表。**

`dmod` 与 T2 DINO 线性可分性的 **Pearson r = +0.719**（两个独立方法同一排序）。

---

## 1. 诊断结论：问题是"增益"，不是"结构"

### 1.1 ★ 三层漏斗（本轮实测的完整链路）

```
书家 id
  │  ① 查表                 D5: effR90 = 57–74, cosμ ≈ 0     ✅ 表可分，正常
  ▼
e_callig (128/384 维)
  │  ② 融合进 c = t_emb + y_emb   y_emb/c ≈ 0.34–0.77        ⚠ 已被 t_emb 稀释
  ▼
c
  │  ③ adaLN → shift/scale/gate   dmod = 0.05–0.65           ❌❌ 最大瓶颈（差 13 倍）
  ▼
每层调制量
  │  ④ 逐层累加进 x
  ▼
像素                            D3 下界 < 0, D6 ratio_null 1.46   ❌ 淹没在噪声里
```

**瓶颈在 ③ 与 ②，不在 ①。**

### 1.2 为什么 ③ 会塌

adaLN 的调制量是 `shift, scale = chunk(Linear(c))`，**而 `c = t_emb + y_emb`**：

1. **加性混合 → 无隔离**：`t_emb` 幅度是 `y_emb` 的 1.3–2.9 倍，
   经同一 Linear 后，**梯度被 `t_emb` 主导**，`y_emb` 只贡献一个小的方向扰动。
2. **共享投影 → 无区分**：同一个 `adaLN_modulation` 要同时编码"t 决定加噪程度"
   和"y 决定风格"，**两个语义在同一条 384 维通路上正交挣扎**。
   `cos(t_emb, y_emb) = −0.22`（D1）→ 它们**确实近似正交**，所以能各自存活，
   但**谁幅度大谁说话**。
3. **zero-init 的初始死锁**：`adaLN_modulation[-1]` 零初始化（DiT 标准），
   step 0 时 `shift=scale=gate=0`。此时 `∂L/∂(y_emb) = ∂L/∂mod · ∂mod/∂c`，
   而 `∂mod/∂c` 在零点**只由 `c` 的幅度决定** → **`y_emb` 越小，起步梯度越小**
   → 越训越弱 → 自我锁定。**v15c 的 `dmod=0.05` 就是这个死锁的终点。**

> **⇒ 结论：不动 adaLN 的结构，要动的是它的"配比"与"起步条件"。**

---

## 2. 改造方案：4 个改动，按性价比排序

### ★ 改动 1（最高优先）：给风格向量独立投影 + 可学习幅度门控

> **状态：已实现并验收（2026-09-23）** —— 见 §7 实施记录。

**问题**：现在 `c = t_emb + y_emb`，`y_emb = callig_scale · callig_proj(e_callig)`，
`callig_scale` **初值 1.0**（`dit.py:851`），完全靠训练自己找平衡。

**改法**：把"风格对 c 的贡献"显式拆出来，并给它一个 **更大的初值 + 独立 LN**：

```python
# __init__（新增，默认关闭以保持旧 ckpt 逐位兼容）
style_gain_init=2.5,        # 风格分支的初始增益（显式放大）
style_ln=True,              # 风格分支独立 LayerNorm，切断幅度与 t_emb 的耦合

# forward
_s = self.callig_proj(e_callig)
if self.style_ln:
    _s = self.style_ln_mod(_s)          # ← 新增 nn.LayerNorm(D)
y_emb_callig = self.style_gain * _s     # style_gain = nn.Parameter(init=2.5)
```

**为什么有效**：

- 独立 LN 让 `‖y_emb‖` **不受 `e_callig` 原始幅度影响**（D5 显示 v13 系表范数失衡 63 倍！
  `0.21…13.07`，独立 LN 直接把这个失衡抹平）。
- `style_gain` 初值 2.5 让 `y_emb` 起步就与 `t_emb` 同量级（`y/t` 从 0.34 → ~0.85），
  **打破 zero-init 死锁**——起步梯度 ∝ `c` 幅度，`c` 大了梯度就大。
- ⚠ **不能设太大**（如 5.0）：会压过 `t_emb`，让模型分不清加噪程度 → 字形崩。
  **用 `strict SSIM ≥ 0.56` 守门。**

**验证**：训练 5k 步后测 D1，看 `ratio_y_over_t` 是否 ≥ 0.7 且 `dmod` ≥ 0.3。
若 `dmod` 没起来 → 说明瓶颈不在配比，直接跳到改动 3。

---

### ★ 改动 2：adaLN 的"风格专用分支"（双流调制）

> **状态：已实现并验收（2026-09-23）** —— 见 §7 实施记录。

**问题**：改动 1 只是调幅度，`t` 与 `y` 仍**共享同一条 `adaLN_modulation`**。

**改法**：给风格一条**独立的调制支路**，与 t 支路**相加**：

```python
# 现在
c = t_emb + y_emb
shift, scale, gate = self.adaLN_modulation(c).chunk(3, dim=-1)

# 改成（style_adaln_stream=True 时）
c_t = t_emb
c_s = y_emb
shift = self.adaLN_modulation(c_t)      # t 支路（保持零初始化，旧行为）
shift_s = self.adaLN_style(c_s)         # ★ 风格支路（独立 Linear，也零初始化）
shift = shift + shift_s                  # 相加
```

**为什么有效**：

- **梯度隔离**：`∂L/∂(风格支路)` 不再经过 `t_emb` 的投影，
  两者**各有一条 Linear**，规模对等 → 风格不用跟 `t_emb` 抢容量。
- **仍可回退**：`adaLN_style` 零初始化 → step 0 输出恒等于旧行为，
  **可安全从任何 ckpt 续跑**（实测 resume 后第一个 `dmod` 变化即可判定是否"点着"）。
- **可诊断**：直接对比两个支路的输出范数：
  `‖adaLN_style(c_s)‖ / ‖adaLN_modulation(c_t)‖` —— **这就是改动是否生效的直接读数**。

**成本**：每 block 一个 `Linear(D, 6D)` = `384×2304 ≈ 0.88M` 参数 × 12 层
= **10.6M（+29% 模型参数）**。
⚠ 若嫌贵，可用 **low-rank**：`Linear(D, r) → SiLU → Linear(r, 6D)`，`r=64`
→ 每层 `384*64 + 64*2304 = 172K`，12 层 = **2.1M（+5.7%）**。**推荐 low-rank。**

**与改动 1 的关系**：**可以先做 1，再叠加 2**。若 1 单独就够，2 就不用上。

---

### 改动 3：把"局部通路"的目标从 `g_tok` 换成**调制 x 本身**

**现状**：B1/B2/B3 三条通路**全部作用在 `g_tok` 上**，最后才 `x = x + glyph_scale * g_tok`。

**问题**：`glyph_scale` 初值 **0.4**（`dit.py:815`）—— 风格经过 `g_tok` 后
**又被乘了 0.4**。相当于：`风格 → g_tok → ×0.4 → x`。
**这是第二道衰减**（第一道是 adaLN 的 `dmod`）。

**改法**：新增一条**直接作用于 x** 的通路（`local_ca_q="x"` 已有雏形，
但仍是零初始化 + 在 block 之间），改为**同时给 x 一个显式的风格缩放**：

```python
# 在 x = x + glyph_scale * g_tok 之后，新增
if self.x_style_add is not None:
    x = x + self.x_style_scale * self.x_style_add(_e_cond())   # (N,D) -> (N,1,D) 广播
```

**为什么有效**：

- 绕开 `glyph_scale=0.4` 的第二道衰减。
- 与 adaLN 正交：adaLN 改的是**每层的 shift/scale**，
  这是**直接加到残差流**，两条路的梯度不冲突。
- **与改动 1 共用同一个 `_style_out()` 前处理**（同一份 LN + 同一套维度懒适配）
  → 不会再引入第二个维度坑。

**★ 实现清单**（触发条件：改动 1+2 后 `dmod` 仍 < 0.30）：

| 项 | 落点 |
|---|---|
| 构造参数 | `x_style_scale_init=0.0`、`x_style_rank=0`（0=关） |
| 参数 | `dit.py` `__init__`：`self.x_style_add = nn.Linear(hidden, hidden)`、`self.x_style_scale = nn.Parameter(tensor(init))`；**放在 `initialize_weights` 末尾再设一次**（§5 坑 1） |
| 输入 | 复用 `_e_cond()`（= `[e_style ; e_script]`）→ 过 `_style_out()` 同款 LN |
| 接线 | `dit.py` `forward` 中 `x = self.x_embedder(x)` 之后、`g_tok` 注入之前，**只接一次** |
| 广播 | `(N,D) → (N,1,D)` 加到残差流，**不加位置编码**（位置由 RoPE 管） |
| 门控 | 全部 4 条 fusion 共用（**不在 fusion 分支内**，避免又漏接） |
| 验收 | `test_style_wiring.py` 加一项：`x_style_scale.grad != 0` 且 `x_style_add.weight.grad != 0` |

⚠ **风险**：直接加到 x 上最容易破坏字形（因为没有"按位置调制"的约束）。
→ 必须配 `strict SSIM` / `ink_ssim` 守门，且初值设小（`1e-3`）；
→ 若 `strict_ssim` 掉超过 0.02，立即回退到 `x_style_scale_init=0`。

---

### 改动 4：`t_emb` 的幅度驯服（可选，治本）

**问题根源**：`t_emb ≈ 29–49` 而 `y_emb ≈ 16–23`。**为什么 `t_emb` 这么大？**

`TimestepEmbedder` 是 `Linear(256, D) + SiLU + Linear(D, D)`，无归一化。
而 `t ∈ [0, 1000]`，sinusoidal 编码幅度很大 → 输出 `t_emb` 幅度失控。

⚠ **注意与 §7.1 的实测数字对照**：`gl.` 阶段 `‖t_emb‖` 实测只有
**0.886**（不是 29–49）。差异来自"未训练 vs 已训练"—— 统计量以
**已训练 ckpt（D1 那 7 个）** 为准，改造本身要以 `_init_style_gain`
的**运行期实测**为准（它每次都重新测，不依赖文档里的魔法数）。

**改法**：给 `t_emb` 加一层 **RMSNorm 或 LayerNorm**：

```python
t_emb = self.t_embedder(t)
if self.t_norm is not None:
    t_emb = self.t_norm(t_emb)     # nn.LayerNorm(D)，可学习
```

**★ 实现清单**（触发条件：改动 1+2+3 都无效）：

| 项 | 落点 |
|---|---|
| 构造参数 | `t_norm=False`（默认关，保住历史复现） |
| 参数 | `dit.py` `__init__`：`self.t_norm = nn.LayerNorm(hidden_size) if t_norm else None` |
| 接线 | `dit.py` `forward`：`t_emb = self.t_embedder(t)` 之后立即过 `t_norm`，**`c = t_emb + y_emb` 之前** |
| 副作用 | `style_gain` 的自动标定**会自动跟随**（它读的是实测 `‖t_emb‖`），无需手改 |
| 兼容 | 开启后**无法**直接 resume 旧 ckpt（`t_norm` 是新增 key，但旧权重全部保留 → 只多不少，实际可 resume） |
| 验收 | `dmod` 上升 **且** `ratio_y_over_t` 在**不开** `style_ln` 时也 ≥ 0.5 |

**为什么有效**：把 `t_emb` 的尺度固定住，`y_emb` 的**相对影响力自然上升**
（不需要靠 `style_gain` 硬拉）。这是**最干净的做法**，但也是**改动最深的**
（影响所有历史 ckpt 的复现）。

⚠ **建议放到最后做**，且**只在改动 1/2/3 都失败时**才上。

---

## 3. 实施顺序与验收

```
第 1 步  改动 1（style_gain + 独立 LN）
         └─ 5k 步看 D1：ratio_y_over_t ≥ 0.7?  dmod ≥ 0.3?
            ✅ 继续 15k，看 D3 下界是否转正
            ❌ 直接上第 2 步

第 2 步  改动 2（adaLN 风格专用 low-rank 支路）
         └─ 5k 步看：‖adaLN_style‖ / ‖adaLN_modulation‖ 是否 > 0.1?

第 3 步  改动 3（x 的直接风格加法）
         └─ 只在 1+2 有效但 D3 下界仍为负时（说明"幅度够了但没到像素"）

第 4 步  改动 4（t_norm）
         └─ 只在 1+2+3 都无效时（说明根因是 t/y 尺度失衡本身）
```

### 每步的**硬判据**（都在 5k 步内可判）

| 检查点 | 指标 | 通过线 | 不通过的含义 |
|---|---|---|---|
| 通路是否"点着" | **`dmod`** | ≥ 0.30 | zero-init 没被激活 → 调大 gain / 换改动 2 |
| 幅度是否够 | `ratio_y_over_t` | ≥ 0.70 | 仍在被 t 稀释 → 调大 `style_gain` |
| 是否进像素 | **D3 严格下界** | **> 0** | 幅度够但没进 → 上改动 3 |
| 字形是否守住 | `strict SSIM` | ≥ 0.56 | 风格压过结构 → 调小 gain |
| 书家是否可分 | **D6 `ratio_null` 中位** | ≥ 1.5（16 字） | 通路无效 → 回退 |

⚠ **样本量铁律**：D6 必须 **≥16 字**（9 字会把 1.46 误报成 2.68）。

---

## 4. 明确不做的事（避免重蹈覆辙）

| 不做 | 原因 |
|---|---|
| ❌ **不用 `CalligStyleCrossAttn`（K 个风格 token 当 K/V）** | K token 间 `cos=0.884`（v15a）/ `0.66`（v15b）→ **"多模态"是假的**，无可寻址内容 |
| ❌ **不用 `style_ctx_every_layer`** | 成本 +33%，v15c 独立口径**最差**（T2 0.110） |
| ❌ **不重建条件表** | D5 已证表可分（effR90 57–74）→ 表不是瓶颈 |
| ❌ **不靠加训练步数** | v15c 训 210k 最差，v13_12ch 22.5k 就最好 → **步数不解决问题** |
| ❌ **不用 `ratio_style` 做判据** | 测离散度不是可分性（历史误导来源） |
| ❌ **不用 `in/bg` 判断"该上局部通路"** | 噪声地板 `in/bg` 同样 ≫1.5（D3 坑 1） |
| ⚠ **`LocalStyleGlyphAdapter` 暂缓** | 它作用在 `g_tok` 上，会再吃一道 `glyph_scale=0.4` 衰减 → **先做改动 1/2/3，若仍不够再考虑** |

---

## 5. 代码落点（实际实现，2026-09-23）

| 改动 | 构造参数 | 参数在哪 | `forward` 接线 |
|---|---|---|---|
| 1 | `style_ln`、`style_gain_init`、`style_y_over_t_init` | `dit.py` `__init__` L1157–1184（`factorized_add/cat` 公共分支内）<br>+ L1327（`xl_highdim` 独立分支）<br>+ `_style_branch` / `_style_out` / `_init_style_gain` | L1973 / 1982（add）、**L2017（cat）**、L2062（xl_highdim）、L2068（else） |
| 2 | `style_ada_rank`（0=关） | `modules.py` `DiTBlock` / `FinalLayer` 的 `style_ada_in/down/up`<br>+ `StyleAdaBranchInitMixin` | `dit.py` `forward`：4 处 `block(...)` 调用 + `final_layer(...)` 均传 `c_style` |
| 3 | — | 待做 | — |
| 4 | — | 待做 | — |

**全部默认关闭**（`style_ln=False`、`style_ada_rank=0`）→ 与历史 ckpt **逐位等价**。

### ★ 开工前踩过的 4 个坑（务必记住）

1. **`_basic_init` 会冲掉所有 zero-init。**
   `initialize_weights()` 对**每个** `nn.Linear` 做 `xavier_uniform_`，
   把 `DiTBlock.__init__` 里刚设的初值整片覆盖。
   → 必须在 `initialize_weights` 末尾**再设一次**（`script_film` / `local_ca` 已踩过同样坑）。

2. **★ 改动 2 的 `W_up` 不能用 zero-init。**
   实测：`W_up = 0 → ∂L/∂W_up ≡ 0`，**打开主干 gate 也救不回来**，
   参数永远不动（`test_style_ada.py` T5 交叉验证）。
   → 改用 `STYLE_ADA_INIT_STD = 0.002`（小随机）：
   `LN 后 z_down 的 per-dim std ≈ 1.23` → 支路输出 rms ≈ 0.020，
   与 DiT 自身 `final_layer` 的 `std=0.02` 主初始化同量级。
   ⚠ 不要调到 0.02 以上：主干还全是 0 时支路会反客为主。

3. **★ 零初始化模型做梯度测试会得到"全零"的假象。**
   未训练时 `adaLN[-1]` 与 `final_layer.linear.weight` **全为 0**
   → 模型输出**结构性恒为 0**（实测 `out.std()=0.0000`）
   → 任何参数的梯度都是 0，极易把"接通了"误判成"没接通"。
   → 测梯度前必须先用非零统计量"激活"主干（见 `test_style_wiring.py` 的
   `activate_backbone()`）。

4. **★ 构造期维度 ≠ 运行期维度（第二类坑，详见 §7.5）。**
   `e_callig` 在 `hier_style=1` 下是 `D_style`、在 `xl_highdim` 下是 `d_c`(384)，
   **都不是**构造期看的 `callig_embed_dim`(128)。
   → 一切"按构造期维度建的投影"都要做**懒适配**（首次前向按真实维度重建），
   否则直接 `RuntimeError: mat1 and mat2 shapes cannot be multiplied`。
   ⚠ 注意 `Identity` 分支：当 `callig_embed_dim == hidden_size` 时投影被建为
   `nn.Identity`，此时若**实际维度仍不符**也必须换回真 `Linear`（本地已修）。

### ⚠ 接线铁律：**"参数建出来了" ≠ "通路接上了"**

`condition_fusion` 有 **4 条**互斥分支（`factorized_cat` / `factorized_add` /
`xl_highdim` / `else`），`forward` 里必须**逐条**接。漏接的后果极其隐蔽：

- 参数量正常（参数确实建了）
- 前向不报错（返回值只是被丢弃）
- **指标与 baseline 逐位相同**（因为训练完全无效）

→ 所以**每次加新通路，必须跑 `_sync_work/test_style_wiring.py`**：
用"梯度是否存在"而不是"参数是否存在"做判据。

## 6. 一句话总结

> **adaLN 不是错的，是"饿着的"。**
>
> 现在 `c = t_emb + y_emb` 里 `y_emb` 只占 0.34–0.77，且共享投影、共享初始增益、
> zero-init 起步梯度 ∝ `c` 幅度 → **越弱越锁死**（v15c `dmod=0.05`）。
>
> **修法 = 给风格一条独立、够响、可回退的路**：
> 独立 LN 抹平范数失衡 → 自动标定 `style_gain` 打破死锁 → （不够再加）adaLN 风格专用 low-rank 支路
> → （还不够再加）直接加到 x 绕过 `glyph_scale` 衰减。
>
> ⚠ 标定值由**实测**反解，**不要手猜**：初版设计的 `style_gain=2.5` 实测会过冲 53 倍（见 §7.1）。
>
> **每一步都用 `dmod` / D3 下界 / D6 `ratio_null`（16 字）在 5k 步内判定，不通过就跳下一步。**

---

## 7. 实施记录（2026-09-23，改动 1 + 改动 2 已实现并验收）

### 7.1 改动 1：`style_ln` + `style_gain`（自动标定）

**改动文件**：`src/model/dit.py`、`src/train/train.py`、`src/train/cli.py`、`src/eval/model_io.py`

新增参数：

| CLI | 默认 | 说明 |
|---|---|---|
| `--style-ln` | `False` | 风格分支过独立 `LayerNorm` |
| `--style-gain-init` | `1.0` | 显式初值；**传负数（如 `-1`）走自动标定** |
| `--style-y-over-t-init` | `1.0` | 自动标定的目标 `‖y_emb‖/‖t_emb‖` |

**★ 关键实测：不要手猜 `gain`。**

```
style_ln_mod 输出范数 = 19.13   (≈ sqrt(384)=19.60，LN 的必然结果)
t_embedder   输出范数 =  0.886  (每维 std 0.045)
```

即 **LN 输出是 `t_emb` 的 21.6 倍**。初版设计里写的 `style_gain=2.5`
会让 `y/t` 冲到 **53**（不是"同量级"，是**过冲 53 倍**，直接把 `t` 条件盖掉）。

→ 所以改成**自动标定**：

```
style_gain = style_y_over_t_init * ‖t_emb‖ / ‖LN(e_callig)‖
```

实测标定结果 **`style_gain = 0.0462`**（不是 2.5！），标定后 **`y/t = 1.000`**（恰好命中目标）。

**验收**（`_sync_work/test_style_branch.py`，ALL PASS）：

| 项 | 结果 |
|---|---|
| 关闭时新增参数 | **0**（与旧 ckpt 逐位等价） |
| 开启时新增 key | `style_gain`、`style_ln_mod.{weight,bias}`、`style_in_proj.{weight,bias}` |
| 参数量 | 32,800,161 → 32,850,466（**+50,305，+0.15%**） |
| 旧 key 丢失 | **0**（可安全 resume） |
| `style_ln=False` 时 `_style_branch` | **恒等**（前向完全不变） |
| 开启后 `y/t` | 0.258 → **1.01**（目标 1.0，未过冲） |

### 7.2 改动 2：adaLN 风格专用 low-rank 支路

**改动文件**：`src/model/modules.py`（`DiTBlock` / `FinalLayer` / `StyleAdaBranchInitMixin`）、
`src/model/dit.py`、`src/train/train.py`、`src/train/cli.py`、`src/eval/model_io.py`

新增参数：`--style-ada-rank`（默认 `0` = 关闭）

结构：

```
mod = adaLN(c) + W_up · LN( W_in · e_callig )
      └─ 主干（t 主导，旧语义全保留）  └─ 新增，独立参数、独立梯度
```

两条路**相 `+`**（不是替换）→ 主干梯度路径完全不变。

| 项 | 实测 |
|---|---|
| 新增 key | **91** = 13 模块 × 7（`style_ada_in` 2 + `down` LN 2 + `down` Linear 1 + `up` 2） |
| 参数量（rank=64） | 32,800,161 → 35,620,641（**+2,820,480，+8.60%**） |
| rank=32 | 约 +1.1M（+3.2%） |
| 旧 key 丢失 | **0**（可安全 resume） |
| `c_style` 送达 | **13/13 模块**（12 blocks + final_layer），`None` 次数 = 0 |
| 支路起步贡献 rms | **0.0204**（与 DiT 自身 `std=0.02` 主初始化同量级，不喧宾夺主） |

**★ `W_up` 初值 = `STYLE_ADA_INIT_STD = 0.002`（小随机，不是 zero-init）** —— 理由见 §5 坑 2。

### 7.3 真实训练路径冒烟（39.3M 参数，40 步）

改动 1 + 改动 2 **同开**，跑 `v17_s2_s2z_baseline`：

```
[style_branch] gain 自动标定: ‖t_emb‖=0.928 ‖style_out‖=19.197
                             target_y/t=1.00 -> style_gain=0.0484   ✅
Trainable Parameters: 39,316,242        (baseline 36.5M -> +2.82M)  ✅
step=0000005  Diff: 1.9634   (与 baseline 同量级，未过冲)           ✅
step=0000020  Diff: 1.0368                                          ✅
step=0000040  Diff: 0.8252   (单调下降，无 NaN)                      ✅
[in-mem-eval] strict: ssim=0.4057  ink_ssim=0.3020                  ✅
Reached max_steps=40; stopping cleanly.
```

### 7.4 验收脚本（都在 `_sync_work/`）

| 脚本 | 作用 |
|---|---|
| `test_style_branch.py` | 改动 1：向后兼容 / 恒等 / 自动标定 |
| `test_style_ada.py` | 改动 2：兼容 / 小随机初值 / 接通 / 梯度可达 |
| **`test_style_wiring.py`** | **★ 4 种 `condition_fusion` 通路的接线回归（必跑）** |
| `smoke_style_branch.sh` | 真实训练路径 40 步冒烟（改动 1+2） |

### 7.5 ★ 第二类坑：**运行期维度 ≠ 构造期维度**（2026-09-23 二次事故）

§5 的铁律说"必须逐条接"，但**接通了仍可能崩溃** —— 因为风格向量的维度
**在构造期无法可靠预知**。

`c_style = e_callig`，而 `e_callig` 的维度由运行时开关决定：

| 路径 | `e_callig` 维度 |
|---|---|
| 普通（无 hier / 无多模态） | `callig_embed_dim`（128） |
| **`hier_style=1`（S2 主线！）** | **`D_style`**（`e_style` 的输出，非 128） |
| 多模态 `style_tokens` | `D` |
| **`xl_highdim`** | **`d_c`（= `hidden_size`，384）** |

而构造期只知道 `callig_embed_dim`，于是 `style_ada_in = Linear(128, 384)`
在 `xl_highdim` 下直接：

```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x384 and 128x384)
```

**教训**：任何以"构造期维度"建的投影，只要输入来自条件向量，就必须做
**懒适配**（首次前向时按真实维度校正）。已修三处：

| 位置 | 修法 |
|---|---|
| `modules.StyleAdaBranchInitMixin._style_ada_apply_in` | `in_features != c_style.shape[-1]` → 就地重建 `Linear` |
| `modules.DiTBlock._mod6` / `FinalLayer._mod2` | 改走 `_style_ada_apply_in(c_style)` |
| `dit.DiT_2Cond._style_out` | `Identity` 也要检查：`callig_embed_dim == hidden` 时它建 `Identity`，但实际 `d_in != hidden` → 换成真投影 |

**修后 4 种 fusion 全绿**（`test_style_wiring.py`）：

```
factorized_cat   style_gain=12.26  style_ln=0.384  ada_up=0.800  ada_down=0.0242  ✓
factorized_add   style_gain=14.18  style_ln=0.288  ada_up=0.538  ada_down=0.0187  ✓
xl_highdim       style_gain= 1.64  style_ln=0.026  ada_up=0.066  ada_down=0.00145 ✓  ← 本轮修复
default(else)                                       ada_up=0.035  ada_down=0.00080 ✓
```

### 7.6 起训配置（已建，2026-09-23）

两份配置都以 `v17_s2_s2z_baseline.json`（纯基线，`hier_style=0`，
`factorized_cat`，`lr=5e-4`，40k 步）为模板**逐字段复制**，
**唯一差别只在 `style_*` 四个字段**，从而保证单变量可比：

| 配置文件 | 改动 | `style_ln` | `style_ada_rank` | 参数量 |
|---|---|---|---|---|
| `v17_s2_s2z_baseline.json` | 无（对照） | — | — | 36,445,457 |
| **`v17_s2z_ada1_ln.json`** | 改动 1 | `true` | `0` | **36,495,762**（+50,305） |
| **`v17_s2z_ada12_ln_rank64.json`** | 改动 1+2 | `true` | `64` | **39,316,242**（+2,870,785） |

**★ 两条路径都跑过真实训练冒烟，参数量精确对上设计值**：

```
改动 1 单开（20 步）：
  [style_branch] gain 自动标定: ‖t_emb‖=0.875 ‖style_out‖=19.131 -> style_gain=0.0457
  Trainable Parameters: 36,495,762
  step=5 Diff 1.8896 -> step=20 Diff 1.0360      (与 baseline 轨迹重合，无过冲)

改动 1+2 同开（40 步）：
  [style_branch] gain 自动标定: ‖t_emb‖=0.928 ‖style_out‖=19.197 -> style_gain=0.0484
  Trainable Parameters: 39,316,242
  step=5 Diff 1.8777 -> step=40 Diff 0.8042      (单调下降，无 NaN)
```

即：**两个改动的开关是正交的、可分别关闭的**，参数量差值 2,820,480
精确等于 rank=64 支路的理论参数量。

### 7.7 下一步

改动 1 / 2 已就绪，**可以直接起训**：

1. **先跑 `v17_s2z_ada1_ln`**（改动 1 单独版）作为最干净的单变量对照；
2. 同时跑 **`v17_s2z_ada12_ln_rank64`**（改动 1+2 同开）；
3. 5k 步内看 §3 的硬判据（`dmod ≥ 0.30`、`ratio_y_over_t ≥ 0.70`）；
4. 若 `dmod` 仍不达标 → 上**改动 3**（`x` 的直接风格加法，绕开 `glyph_scale=0.4` 的二次衰减）；
5. 再不行 → 上**改动 4**（`t_emb` 的 LN/RMSNorm 驯服）。
