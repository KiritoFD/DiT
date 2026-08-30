# 模型架构全景（2026-08-30）

> 本文是对 `src/model/` 的静态梳理 + 实测参数统计，数据来自
> `_scan_model.py`（CPU 实例化 DiT-2Cond-S/2 逐模块统计）。
> 配套 CSV：`5script/model_components.csv`、`5script/condition_dims.csv`。

## 0. 一句话概括

**DiT-2Cond-S/2，46.5M 参数，flow matching，双全局条件（书家 + 字）+ 可选空间条件（骨架/标准字形）。
其中 29% 的参数是冻结的 DINO 字形表。**

---

## 1. 主干：DiT-2Cond-S/2

### 1.1 参数分布（实测）

| 组件 | 参数量 | 占比 | 默认冻结 | 说明 |
|---|---:|---:|---|---|
| `blocks`（Transformer） | 31,898,112 | **68.57%** | 否 | 12 层 DiT block |
| `y_char_embedder` | 13,490,688 | **29.00%** | **是** | 35,130 × 384 字形表 |
| `final_layer` | 301,840 | 0.65% | 否 | 输出投影 |
| `char_proj` | 296,448 | 0.64% | 否 | 字条件 MLP 投影 |
| `t_embedder` | 246,528 | 0.53% | 否 | 时间步嵌入 |
| `y_callig_embedder` | 129,792 | 0.28% | 否 | 1,013 × 128 书家表 |
| `pos_embed` | 98,304 | 0.21% | 否 | 32×32=1,024 token |
| `callig_proj` | 49,792 | 0.11% | 否 | 书家条件投影 |
| `x_embedder` | 6,528 | 0.01% | 否 | PatchEmbed (patch=2) |
| `glyph_scale` | 1 | ~0% | 否 | 字形条件可学习缩放 |
| **合计** | **46,518,033** | 100% | | |

**值得注意**：字形表占了近三成参数（13.5M），却默认冻结。它是模型里第二大的组件，
比 Transformer 之外的所有模块加起来还大 20 倍。

### 1.2 前向结构

```
输入 latent x: (N, 4, 32, 32)
  ↓ x_embedder (PatchEmbed, patch=2)
token 序列: (N, 1024, 384)
  ↓ + pos_embed
  ↓ × 12 个 DiT block（每块接收 c = t_emb + y_emb）
  ↓ final_layer
  ↓ unpatchify
输出: (N, 4, 32, 32)  [flow: velocity; ddpm: eps]
```

- **norm**: RMSNorm
- **MLP**: SwiGLU
- **QK-Norm**: 开
- **RoPE**: 开（θ=100）
- **attn**: sdpa（torch 1.13 无 F.scaled_dot_product_attention 时回退 xformers；
  远程环境实际走 xformers 分支，见启动日志）

### 1.3 条件融合：`factorized_add`

```python
c = t_embedder(t) + callig_proj(y_callig_emb) + char_proj(y_char_emb)
```

书家与字条件各自投影后**相加**，再与时间步嵌入相加，得到 AdaLN 的调制向量。
这是最简洁的融合方式，代价是：两个条件之间没有交互建模（无交叉注意力），
只有加法层面的叠加。

---

## 2. 条件机制

### 2.1 四类条件信号（维度对比是理解性能的关键）

| 信号 | 来源 | 注入维度 | 空间性 | 状态 |
|---|---|---:|---|---|
| 书家 ID | `y_callig_embedder` (1,013 类) | 128 | 全局标量 | 已启用 |
| 字 ID | `y_char_embedder` (35,130 类, DINO 初始化) | 384 | 全局标量 | 已启用（**冻结**） |
| 标准字形 latent `g` | `std_glyph_latent_v2` | 4,096 | **4×32×32 空间图** | 代码完整但**接线失效**（见 §2.3） |
| 骨架 latent | `final_skel_latents_fame` | 4,096 | **4×32×32 空间图** | ControlNet 专用，可用 |

**核心量化对比**（`condition_dims.csv`）：

```
全局条件有效维度  ≈ 128 + 34.1(字表有效秩) ≈ 162
空间条件维度      = 4,096
比值              ≈ 120x
```

这个 120 倍的差距，是理解「base 0.50 vs ControlNet 0.80」的关键（详见 §5）。

### 2.2 字条件：DINO glyph embedding

构造方式（`train.py:270-361`）：

```
glyph_id = script_id × 7026 + character_id
DINO CLS = 该 glyph 所有书写样本的 CLS token 取平均 → L2 归一化
→ 写入 y_char_embedder.embedding_table
```

`freeze_char_table=true` 时整表冻结，只有 `char_proj`（MLP）可训练；
`null_embed`（CFG 的 uncond token）保持可训练。

已知问题（本项目实测，见 `14_glyph_condition_probe.md`）：

- 有效秩仅 **34.1 / 384**（PC1 占 26.3% 能量）
- 库内检索 top-1 **84%**，库外（同字不同书家）**4%**
- 即：**特征编码的是「这张图长什么样」，不是「这个字是什么字」**

`dino_per_script_center` 可去书体均值（有效秩 34.1→57.0），
`dino_fill_unknown` 用均值填充未覆盖行（20,468/35,130 有真实 DINO 值）。

### 2.3 标准字形条件 `w_glyph_cond`（**接线失效，未启用**）

设计是合理的：

```python
# dit.py
self.glyph_embedder = nn.Conv2d(4, hidden, kernel_size=p, stride=p)  # 独立投影
self.glyph_scale = nn.Parameter(torch.tensor(0.4))                   # 可学习缩放
...
g_tok = self.glyph_embedder(g)          # 空间图 → token
x = x + self.glyph_scale * g_tok        # token-add 注入
```

用**独立投影**而非复用 `x_embedder`，保证 LoRA 模式下 `g` 信号不被锁死——这个设计是对的。

**但接线是坏的**（详见 `15_codebase_review_20260830.md` P0）：

```
latent_dataset.py:49   from src.utils import get_glyph_lookup      # v1
   ↓
glyph_latent.py:14     LIB_DIR = .../src/utils/std_glyph_latent    # 目录不存在
   ↓
glyph_latent.py:41     if not os.path.isdir(d): continue           # 静默跳过
   ↓
latent_dataset.py:286  g_t = torch.zeros(...)                      # 缺失 → 零
```

fame 数据集实测：v1 命中率 **0.0%**，v2 覆盖 **53.1%**。
即启用该开关后条件从头到尾是零张量，而**不报错、loss 正常下降**。

---

## 3. 扩散 / 流匹配

| 项 | 值 | 备注 |
|---|---|---|
| `diffusion_type` | `flow` | 主流选择；早期系列用 ddpm |
| `t_sampler` | `logit_normal` | |
| `t_mean` / `t_std` | 0.0 / 1.0 | |
| `flow_sampler` | `heun` | 二阶 |
| `shift` | 1.0 | |
| VAE | `sd-vae-ft-ema` (kl-f4) | |
| `vae_scaling_factor` | 0.18215 | |
| latent shape | 4 × 32 × 32 | 256px / 8 |

**历史**：s2–s14 系列用 DDPM，s15 起转向 flow（见 `experiments_enriched.csv`
的 `diffusion_cfg` 列）。flow 在同等步数下质量更好，已是当前默认。

---

## 4. ControlNet

### 4.1 结构

```
骨架 latent (N,4,32,32)
  ↓ ControlConditionEncoder（独立 Transformer，depth=12，hidden=384）
  ↓ 逐层输出 ctrl_feats[12]: (N, 1024, 384)
  ↓
主模型 blocks[i] ──→ x
  ↓ x = x*(1+s) + t     ← ZeroAdaLNInjection（zero-init）
  ↓
final_layer → 输出
```

### 4.2 关键设计

**① ZeroAdaLNInjection（`controlnet.py:70`）**

```python
s, t = self.proj(feat).chunk(2, dim=-1)
return x * (1.0 + s) + t
```

比加法注入（`x = x + feat`）表达力更强：既能增强也能抑制主残差流，
对骨架这种「稀疏强结构」条件更合适。

**zero-init 的梯度含义**：init 时 `s=t=0`，注入严格恒等。
且 `d(out)/d(x) = W = 0` → **ctrl blocks 在 W 变非零前收不到梯度**。
这是 ControlNet 的正确行为（先学注入权重，再学控制特征）。

**② null condition 用高斯噪声而非零（`null_cond="gaussian"`）**

源码注释说明：零 latent 经 VAE 解码不是「空白」而是某种特定灰色块，
与真实骨架分布差距大，会让 CFG 的 uncond 分支落在分布外。

**③ 主模型冻结，只训 ctrl_encoder + injections**（`train_ctrl_only=true`）

### 4.3 CFG 采样

```python
def forward_with_cfg(self, x, t, y_callig, y_char, cfg_scale, cond=None):
    # 骨架始终提供给两半；callig/char 有/无各跑一遍
```

即：**CFG 只作用于全局条件（书家/字），骨架条件不参与 CFG 缩放**。
这解释了为什么骨架条件下 `cfg<1` 反而更好（见 §6）。

### 4.4 一个隐蔽的失效模式（源码已警示）

`controlnet.py:282-304` 的注释：

> 不映射的后果很隐蔽：`load_state_dict(strict=False)` 不报错，
> 但 injections 停留在 zero-init，cond 完全短路 —— 四臂评估结果
> 会完全相同，看上去像「ControlNet 无效」而不是「权重没加载」。

这与 `w_glyph_cond` 的零张量失效是**同一种模式**：
**zero-init / 零填充 + strict=False = 静默失效**。
仓库里这两处都已通过注释告警，但排查新问题时值得优先检查。

---

## 5. 为什么 base 0.50 而 ControlNet 0.80

这是本项目最重要的结构性问题。

### 5.1 信息量差距

| | base（预训练） | ControlNet |
|---|---|---|
| 全局条件 | 书家 128 + 字 384（有效 ~34） | 同左 |
| **空间条件** | **无** | **骨架 latent 4×32×32 = 4,096 维** |
| 总条件维度 | ~162（有效） | ~4,258 |
| 实测 SSIM | 0.467–0.50 | 0.73–0.80 |

**骨架提供的不是「提示」，而是答案的空间骨架。** 模型不需要再从
一个 384 维（有效秩 34）的模糊向量里猜字形——结构已经摆在面前。

### 5.2 这解释了什么

- **为什么解冻字 embedding 无效**（s22 实测 0.4622 < 基线 0.4664）：
  瓶颈不是「表被冻结」，而是「这个特征本身不含字符身份信息」。
  解冻一个信息量不足的表，只引入过拟合自由度（每字仅 10.8 样本 × 384 维）。

- **为什么「字→骨架网络」价值有限**：预测出的骨架属于「有结构但不是目标结构」，
  实测他人书法骨架只值 +0.007（0.5007→0.5074）。

- **数据难度对指标的巨大影响**：见 `17_experiments_registry.md` §3。

---

## 6. 评测体系

| 指标 | 含义 | 计算方 |
|---|---|---|
| SSIM | 结构相似性（↑） | CPU daemon |
| MSE | 像素均方误差（↓） | CPU daemon |
| LPIPS | 感知距离（↓） | CPU daemon（VGG） |
| SkelIoU | **骨架跟随度**（↑） | CPU daemon，新指标 |

**SkelIoU 是本项目最重要的新增指标**：从生成图提取骨架，与给定骨架算 IoU，
直接衡量「模型是否真的在跟随条件」。它戳破了 std-skel 实验的假象——
目检以为成功，但 IoU 只有 0.015。

**评测流程**：GPU 侧采样并写 `eval_pending_*.json` → CPU daemon 计算指标。
注意 daemon 需**按实验目录单独启动**，换新实验名时容易漏配（见 15 文档 P4）。

**CFG 设置**：当前统一 `cfg=0.7`。历史扫描过 >1 区间（1.7 最优，4.0 崩），
<1 区间也扫过但未进文档。注意该 0.7 是为**骨架条件**调的，
而骨架在 CFG 中不参与缩放（§4.3），因此 base 与 ctrl 共享该值是否合理，
尚未系统验证。

---

## 7. 附：架构相关的历史决策

| 决策 | 现状 | 出处 |
|---|---|---|
| 注入方式 add → adaLN 调制 | 已采用 `modulate` | `controlnet.py:28-31` |
| null cond: zeros → gaussian | 已采用 `gaussian` | `controlnet.py:33-36` |
| `char_proj_mode`: ln_only → mlp | 已采用 `mlp`（ln_only 容量不足） | 配置 |
| ckpt key 布局迁移映射 | 已实现 `_remap_ctrl_keys` | `controlnet.py:282` |
| DINO ckpt 路径少一级 | **已修**（commit 9844b8f） | `losses.py` |
