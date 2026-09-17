# 61. ref (Moyun) 代码精读：12ch / 白底归零 为什么在我们这失败，以及真正该学什么

> 日期：2026-09-16
> 相关：[51](51_moyi_aux_alignment.md)、[52](52_ref_model_alignment.md)、[54](54_results_and_insights_20260913.md)、
> [59](59_12ch_and_white_zero_retrospective.md)、[60](60_what_is_actually_useless.md)
>
> **结论先行**：
> 1. **12ch 在 ref 里是"唯一的结构通路"，在我们里是"第二条结构通路"** —— ref 的所有生产配置
>    `use_stroke=False`，**根本没有任何结构条件**，结构只能从目标通道进。我们有 g 骨架条件，
>    12ch 就是重复投入。这不是"调参没调好"，是**前提不同**。
> 2. **白底归零 ref 自己从没开过** —— 5 个 `.sh` 全部 `--custom_zero 0`。我们照搬了一个
>    作者自己都没启用、因而从未验证的开关，还为此付了一次事故代价。
> 3. **规模差 1~2 个数量级** —— ref 生产模型 ~245M 参数 × ~1.93M 图；我们 42M × 28,569 图。
>    在 68× 的数据差下，任何"用多任务换表征"的配方都不成立。

---

## 0. 规模对照（一切结论的前提）

| | ref (Moyun) | 我们 (DiT_2Cond) |
|---|---|---|
| 生产模型 | `moyun-12channel-B`：depth=12, h=1024, 8 头 | `DiT-2Cond-M/2`：depth=12, h=432, 6 头 |
| 参数量 | **~245M**（估；`moyun_12channel` 24 层版 ~470M） | **~42M** |
| 训练数据 | **~1,929,396 图/epoch**（`train_moyun2.sh` 注释），cmp1 路径 3,312,345 | **28,569** |
| 扩散类型 | DDPM（`create_diffusion`）/ Rectified Flow | Flow matching (logit_normal + heun) |
| `learn_sigma` | **True** → out_channels = in×2 | 不适用（flow 无 sigma 头） |
| batch / lr | 100–384 / 1e-4 | 240 / 1e-4 |
| 条件 | 3 个离散 ID：calligrapher × font × charactor | callig ID + **g 骨架 latent**（开集） |

**参数量核算口径**：DiT block ≈ `18·h²`（qkv+proj 4h² + adaLN 6h² + MLP 8h²）。
校验：DiT-XL/2（28×1152）→ 18·1152²·28 ≈ 669M，与官方 675M 吻合。

---

## 1. 12ch 为什么在 ref 成立、在我们这失败

### 1.1 ref 的 12ch 构造（`train_moyun2_RF.py:305-320`）

```python
image    = vae.encode(image).latent_dist.sample().mul_(0.18215)     # (B,4,32,32)
edge     = vae.encode(edge).latent_dist.sample().mul_(0.18215)      # (B,4,32,32)
skeleton = vae.encode(skeleton).latent_dist.sample().mul_(0.18215)  # (B,4,32,32)
x = torch.cat((image, edge, skeleton), dim=1)                       # (B,12,32,32)
```

三张 RGB 图**各自独立**编码成 4ch latent 再拼接。等权 MSE 监督全部 12 通道。

### 1.2 关键：`use_stroke=False` —— ref 完全没有结构条件

`moyun_2.py` 里 `MoyunBlock` 确实写了 stroke cross-attention（`self.cross_attention`，
骨架 token 作 K/V）。但**注册表里每一个变体都是 `use_stroke=False`**：

```python
moyun_12channel:     use_stroke=False
moyun_12channel_B:   use_stroke=False   # ← 生产配置（_ful.sh）
moyun_4channel:      use_stroke=False
test_models*:        use_stroke=False
moyun_4channel_moyun:use_stroke=False
```

**所以 ref 的结构信息只有一条入口：12ch 的目标通道。** 没有它，模型就只剩
(calligrapher, font, charactor) 三个离散 ID，写不出任何字形。

### 1.3 我们：结构已经从条件侧给了

我们的 `g`（标准字骨架 latent）经 `glyph_embedder` + 4 层 `ZeroAdaLNInjection`
逐层注入，且输入层还有 `glyph_scale·g_tok` 直通。doc 54 实测：char-only 0.3469 →
std-skel **0.5680**（+0.22），g 已经是字身份与结构的主要载体。

→ **12ch 对我们是"同一个信号喂两遍"**：一遍当条件（无损、每层可见），
一遍当监督目标（要分走梯度）。doc 54 实测等权下结构通道吃掉 **~51% final-layer 梯度**
（`patch_embed` 上 aux 梯度是 img 的 **2.6×**）。

### 1.4 三个附加的失效机制（我们自己的实证）

| 机制 | 证据 |
|---|---|
| 等权稀释 image 通道 | img 4ch : aux 8ch = 1:2，image 只拿 1/3 损失权重；且 aux std 更低更易拟合 |
| CFG 作用域 bug | `forward_with_cfg` 由 `image_channels` 控制，12ch 配置里是 `None` → 回退 `in_channels`=12 → CFG 对**全部 12 通道**做引导，aux 分布不同却被同一 scale 放大 → 轨迹跑飞（"墨团"的直接原因） |
| flow 路径被 aux 扭曲 | aux 通道分布与 image 不同，等权下线性插值路径更弯曲（与白底归零同机制，见 §2.3） |

### 1.5 ref 的 CFG 其实比我们更糟（不要学）

`moyun_2.py:648`：

```python
# eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]   # 被注释掉
eps, rest = model_out[:, :3], model_out[:, 3:]
```

12ch + `learn_sigma=True` → 输出 **24 通道**。这里硬编码切 3：
- 被 CFG 引导：通道 0:3（**3/24 = 12.5%**）
- 未引导：通道 3:23（image 第 4 通道 + edge 4 + skeleton 4 + 全部 12 个 sigma）

且 `rest` 取自**联合前向的前半（cond 支）**，所以最终解码的 4 个通道里
**前 3 个有引导，第 4 个是无引导的 cond 输出**。这是从 ADM/DiT 原版沿袭的
legacy 切分（原代码是 RGB 3 通道），在 12ch 下是明显失配。

**我们已用 `image_channels` 显式修好（[58]），这块我们比 ref 强，不要回退。**

### 1.6 一个内部不一致（读代码时注意）

`train_moyun2_diffusion_repa.sh`：`--model moyun-12channel`（in_channels=12）
但 `--use_12channel 0` → 数据分支 `x = vae.encode(image)` 产出 **4ch**，与模型 12ch 不匹配。
同一脚本还在 `use_12channel==0` 时把三个 feature 全换成全零 `pseudo_feature`。
而真正带 `--resume` 的生产脚本 `_ful.sh` 是 `moyun-12channel-B` + `--use_12channel 1`（自洽）。
→ 读 ref 时以 **`_ful.sh` + `train_moyun2.sh` + `train_moyun2_RF.sh`** 为准，repa 那份是 eval/半成品。

---

## 2. 白底归零（wz）为什么失败

### 2.1 决定性证据：ref 自己从没开过

```
train_moyun2.sh                    --custom_zero 0
train_moyun2_RF.sh                 --custom_zero 0
train_moyun2_diffusion_repa.sh     --custom_zero 0
train_moyun2_diffusion_repa_ful.sh --custom_zero 0   # ← 生产
train_moyun2_diffusion_cmp1.sh     --custom_zero 0
```

5 个脚本，5 个 0。而代码里 `get_white_img_embedding()`、`white.pt`、
减白底三行都写好了（`train_moyun2_RF.py:44-47, 281-319`）。

**作者有构想、写了完整实现、但一次都没启用** —— 这就是"大概率没收益"的最强证据。
我们等于把别人抽屉里的草稿当成了已验证配方。

### 2.2 推理侧也从未配套

`generate.py:59` 直接 `vae.decode(samples / 0.18215)`，**没有任何加回白底的步骤**。
与 `custom_zero 0` 自洽 —— 说明这条路径根本没被端到端跑通过。

### 2.3 我们实测的代价（[59] §2.3）

| 指标 | 未减白底 | 减白底 |
|---|---|---|
| img `mean` | +0.28 | **−0.33**（左偏） |
| `\|x\|>2` 占比 | 4.1% | **12.1%**（尾部 3×） |
| canny/skel `std` | 1.21/1.04 | **0.57/0.56**（减半） |

分布变稀疏 + 左偏 + 重尾，flow 的线性插值路径更弯曲。理论收益（白底区 `v = ε`）
成立，但代价是实的，且**从未有过干净对照**。

### 2.4 事故根因

SD VAE 的零向量不是白：`decode(0) = [129,110,89]`（灰黄棕）。减白底后推理
**必须**加回，而 `aux_zero_white` 从未在 `train.py` 的 argparse 注册 →
config 加载器静默丢弃未注册键 → 全线 decode 没加回 → 生成图发黄/发黑（[56]）。

---

## 3. 真正该学的（按性价比排序）

### ★★★ 1. 分维度条件 dropout —— 我们有旋钮但**设成了 0**

ref（`_ful.sh`，生产）：

```
--charactor-x 0.08 --font-x 0.08 --calligrapher-x 0.16
```

三个因子**各自独立**的 drop 概率，且 **calligrapher 的丢弃率是内容的 2 倍**。

我们 `v11_pretrain_Sp2_fame_kxl_tj_px60.json`：

```json
"cond_drop_all_prob": 0.1,
"cond_drop_one_prob": 0.0,        // ← 0！单因子 drop 完全没启用
"cond_drop_which_glyph_prob": 0.85
```

`dit.py:897` 的 4-way mask 只在 `cond_drop_all_prob > 0 or cond_drop_one_prob > 0` 时才生效，
而 `cond_drop_one_prob = 0` → **4-way 退化成 2-way（full / uncond）**，
`cond_drop_which_glyph_prob=0.85` 这个旋钮目前**完全没被用到**。

| | ref | 我们 |
|---|---|---|
| 单因子 drop 总量 | 0.08+0.08+0.16 = **0.32** | **0.0** |
| 偏向 | callig 丢得更多（0.16 vs 0.08）→ 保内容 score | `which_glyph=0.85` → 同向，更极端 |
| 实际生效分支 | 4-way 全开 | 只有 full / uncond |

**方向我们和 ref 一致（都偏向训练 content score），但强度差 3 倍以上。**
CFG 的质量取决于 uncond/单因子分支的训练量 —— 这可能部分解释我们的 CFG 收益有限。

**建议**：`cond_drop_one_prob` 0.0 → **0.20**，保留 `which_glyph=0.85`（或收到 0.67 对齐 ref）。
成本：只改配置，不用重做数据；但严格验证需要重训一段。

### ★★★ 2. 条件融合用 **concat + project**，不要用 add + scale

ref `LabelEmbedder.forward`：

```python
cat_label = torch.cat((calligraphy_embedding, font_embedding, char_embedding), dim=1)
res = self.linear(cat_label)          # Linear(3h → h)，联合非线性
```

我们 `factorized_add`（`dit.py:960-968`）：

```python
y_emb = (self.callig_scale * self.callig_proj(e_callig)
         + self.char_scale   * self.char_proj(e_char)) / math.sqrt(2.0)
```

**concat + Linear 的表达力严格更强**：Linear 一次性看到所有因子，能建模
"书家 × 书体"的交互项；add 是各自投影后固定 1:1 相加，交互只能靠后续 adaLN 间接产生。

⚠ 注意这条与 doc 60 §4 不冲突：ref 的因子分解是**闭集**（字是离散 ID），
我们是**开集**（字身份由 g 承担）。但 **concat vs add 的选择与开闭集无关**，
我们可以只 concat 向量因子 `(callig, script)` 而内容继续由 g 承担。

**建议**：新增 `condition_fusion="factorized_cat"`：
`concat([e_callig, e_script]) → LayerNorm → Linear(256, h)`。参数增量 `+128·h`，可忽略。

### ★★☆ 3. 辅助目标是"表征 shaping"，不是第二个主任务

`generate.py:57-58`：

```python
samples, _ = samples.chunk(2, dim=0)   # 去掉 null 分支
samples = samples[:, 0:4, :, :]        # ← 只取 image latent，8 个 aux 通道直接丢弃
```

**ref 训练了 12 个通道，推理只用前 4 个。** aux 通道生成了但扔掉 ——
说明 12ch 的价值是"逼主干建立结构感知的表征"，**不是"输出结构图"**。

这正是 doc 60 §6 那条纪律的代码来源：
> 新增组件时先问"这个信号是不是已经从别处给了"。

如果将来还要加辅助目标：权重必须低（0.05–0.2，绝不等权），且先确认
该信号没从条件侧给过。

### ★★☆ 4. 位置编码：ref 用绝对 sin-cos，我们用纯 RoPE —— 值得一次消融

ref 所有脚本 `--if-rope 0`，走 `x = x + self.pos_embed`（固定 2D sin-cos）。

但**不能因此认为 RoPE 不好**：ref 的 RoPE 是
`hidden_states = self.rope(hidden_states)` —— 加在 **token 序列上**，在每个 block 之前，
**不是加在 q/k 上**。这不是标准 RoPE 用法（标准版作用于 attention 的 q/k），
实现不对才被关掉。我们的 `modules.py` 是标准的 q/k RoPE + QK-Norm。

**但有一个真实差异**：我们 `rope=1` 时**完全不加绝对位置嵌入**（`dit.py:916-921` 的
`if self.rope: pass`）。对"在画框里写这个字"的任务，绝对位置可能有用。
不过 g 本身是空间对齐的 latent，已经携带了布局信息 —— 大概率不需要。

**建议**：低成本消融 `rope=1` vs `rope=1 + pos_embed`。不急，当前结果已经很好。

### ★☆☆ 5. REPA 层位：ref 取最后一层

`repa_dep` 默认 12：`moyun-12channel-B`（depth=12）→ **最后一层**；
`moyun_12channel`（depth=24）→ 中间层。两者不一致，可信度低。
我们用 layer 8 / depth 12（0.67 位置），落在两者之间。**低优先级，不动。**

### ★☆☆ 6. 数据规模才是主矛盾

ref 245M × 1.93M vs 我们 42M × 28.5k。数据差 **68×**，参数差 **6×**。
在这个比例下，瓶颈是**数据量**不是表征能力。
→ 优先级应该是**加数据**（对称增强已备 83,909、更多书家），而不是加辅助目标。

---

## 4. 不该学的（ref 自己都关掉了）

| 组件 | ref 状态 | 结论 |
|---|---|---|
| `custom_zero`（白底归零） | 5 个脚本全 0，推理侧无配套 | **已弃，勿回** |
| CFG 的 `[:, :3]` 切分 | 24 通道里只引导 3 个，legacy bug | 我们已用 `image_channels` 修好，**比 ref 强** |
| `learn_sigma=True` | DDPM 配套 | flow matching 下无意义，勿搬 |
| `use_mamba` / `use_kan` | 全部 `False` | 他们自己废弃的研究分支 |
| `use_stroke`（stroke cross-attn） | 全部 `False` | 同上，死代码 |
| `--if-rope 0` | 全关 | 他们的 RoPE 实现不对（加 token 而非 q/k）；我们是标准版，**不构成反证** |
| 12ch 本身 | ref 因为没有结构条件才需要 | 我们有 g，**勿搬** |
| `FeatureEmbedder`（DINO 特征表） | `use_feature=False`，走 LabelEmbedder | 与我们的 IDS/DINO 字表结论一致（库外 4%） |
| notebook `run_moyun2.ipynb` | import `moyun_12c`（已不存在） | **已过期，勿作依据** |

---

## 5. 可执行清单

| 优先级 | 动作 | 成本 | 验证方式 |
|---|---|---|---|
| P0 | `cond_drop_one_prob` 0.0 → 0.20（4-way mask 真正生效） | 改配置 | 重训 30–50k vs 当前线 strict |
| P0 | 新增 `factorized_cat`：concat([callig, script]) → LN → Linear(256,h) | ~20 行 | 同上，做 A/B |
| P1 | 加数据：对称增强 83,909（已备好）或扩充书家 | 数据侧 | strict 是否破 0.57 |
| P2 | `rope=1` vs `rope=1 + pos_embed` 消融 | 改配置 | seen/strict |
| P2 | `cond_drop_which_glyph_prob` 0.85 → 0.67（对齐 ref 比例） | 改配置 | 与 P0 一起测 |
| — | ~~12ch / 白底归零 / learn_sigma / Mamba / KAN / stroke~~ | — | **不再投入** |

---

## 6. 纪律（从这次精读里固化）

1. **照搬配置前先问：参考实现有没有我们已有的替代通路？** 12ch 的失败根因不是
   "没调好权重"，而是 ref 没有 g 而我们已有 g —— 前提不同，结论不可迁移。
2. **`x=0` 的开关要看它是否真的被启用过。** ref 的 `custom_zero` 代码完整、
   5 个脚本全 0。写好的代码 ≠ 验证过的配方。
   **检查方法：翻所有 `.sh`/config 的实际取值，不要只看代码有没有这条分支。**
3. **规模不匹配时，配方不可迁移。** ref 的 12ch 等权是在 245M 参数 / 1.93M 图 /
   DDPM+learn_sigma 下成立的；我们是 42M / 28.5k / flow。三者全不同。
4. **新增 config 项后确认它出现在 `resolved_config.json`**（第三次栽在上面：
   `aux_zero_white` 未注册被静默丢弃 → 发黄事故）。
5. **参考实现的"关掉的开关"往往比"打开的开关"信息量更大** ——
   `use_stroke=False` 告诉我们 ref 没有结构条件，这一条推理出 12ch 的全部必要性。
