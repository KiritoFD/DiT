# DINO 条件诊断（2026-08-29）

> 目的：搞清楚 `y_char_embedder`（冻结 DINO CLS 字表）到底给模型提供了多少
> **字符身份**信息，以及 s19 unseen 字 SSIM 只有 0.476 是不是条件信号的锅。
>
> 复现脚本：`tools/dino_diagnose.py`（由本次的 `_probe_dino_fast.py` / `_probe_dino_fix.py` /
> `_probe_rank_fix.py` 合并而来）。

## 0. 结论速览

| 指标 | 数值 | 判读 |
|---|---|---|
| 训练集 DINO 命中率 | **100.0%**（5461/5461） | ✅ 用户的判断正确，训练侧全命中 |
| `eval_strict_midclean` 命中率 | 100.0% | ✅ |
| `eval_unseen_top6` 命中率 | 93.4%（228/244） | ⚠ 有 16 个 glyph 落到未初始化行 |
| **DINO 有效秩** | **34.1 / 384**（PC1 占 26.3%） | ⚠ 384 维里只有约 34 维真正独立 |
| **最近邻是同一书体** | **83.0%**（chance 20%） | ⚠ char 分支被书体主导 |
| 跨书体字符检索 top-1 | 1.9%（随机基线 0.014%） | ⚠ 有信号但绝对量很弱 |
| 跨书体字符检索 top-5 | 4.2% | ⚠ |

**核心判断**：DINO CLS 不是"没有"字符信息（是随机基线的 130 倍），但它的
**大部分容量被书体占用**。而书体信息本该由 `y_callig_embedder` 提供 ——
两个条件分支冗余，字符身份信号被稀释。

## 1. 覆盖情况

```
train_mid_clean                  uniq_glyph=  5461  dino_hit=  5461 (100.0%)  miss=0
train_mid_common                 uniq_glyph=  5461  dino_hit=  5461 (100.0%)  miss=0
eval_strict_top6.csv             uniq_glyph=   271  dino_hit=   271 (100.0%)  miss=0
eval_strict_midclean.csv         uniq_glyph=   501  dino_hit=   501 (100.0%)  miss=0
eval500_3top30_common.csv        uniq_glyph=   305  dino_hit=   302 ( 99.0%)  miss=3
eval_unseen_top6.csv             uniq_glyph=   244  dino_hit=   228 ( 93.4%)  miss=16
```

训练集里 seen / unseen 的划分（`eval_strict_top6`）：seen 178 (65.7%)、unseen 93 (34.3%)。

**注意**：`y_char_embedder` 有 `num_characters=35130` 行，而 DINO 只覆盖 20468 个
glyph。剩下的 **14662 行**停留在 `nn.Embedding` 默认的 `N(0, 0.02)` 且被冻结。

## 2. 有效秩（重要修正）

> ⚠ 最初用随机 SVD 估出的 "有效秩 3.1" 是**错的**：代码把已经是 σ² 的谱又平方了一次
> （`p = S**2` 而 `S` 已经是 σ²），导致 σ⁴，严重高估集中度。
> 下表是用**完整 SVD**（`torch.linalg.svdvals`）重测的正确值。

| 变体 | 有效秩 | PC1 | PC10 | PC64 |
|---|---|---|---|---|
| raw DINO（s19 现状） | **34.1** | 0.263 | 0.664 | 0.890 |
| **A: per-script centering** | **57.0** | 0.167 | 0.553 | 0.851 |
| B: cross-script 平均 | 35.0 | 0.271 | 0.656 | 0.889 |

方案 A（减去每个书体的均值再 L2 归一化）把有效秩提升 67%，且不增加任何参数、
不需要改模型 —— 只是 DINO 注入前的一步离线预处理。

## 3. 检索能力

跨书体字符检索（用 A 书体的某字去检索 B 书体的同一个字，6952 个候选）：

| 变体 | top-1 | top-5 | top-10 | 书体泄漏 |
|---|---|---|---|---|
| raw | 1.9% | 4.2% | 5.7% | 83.0% |
| per-script centering | 2.6% | 6.8% | 9.0% | 77.9% |

（随机基线 top-1 = 1/6952 = 0.014%）

书体泄漏 = 最近邻落在同一书体的比例。83% vs chance 20% 说明 DINO 向量的
主要组织原则就是书体，而不是字符。

## 4. 已实施的改动

| # | 改动 | 依据 |
|---|---|---|
| 1 | `char_proj_mode: ln_only -> mlp` | `ln_only` 只给字符分支 768 个可学习参数（一个 LayerNorm）。面对有效秩 34 的冻结输入，没有任何容量去放大/重组有用方向。改为 `LayerNorm+Linear+SiLU+Linear`，296,448 参数。 |
| 2 | `--dino-per-script-center` | 有效秩 34.1 → 57.0，top-5 4.2% → 6.8%，书体泄漏 83.0% → 77.9%。 |
| 3 | `--dino-fill-unknown`（默认开） | 14662 个未命中行原本是冻结的 `N(0,0.02)`（范数 0.39），与真实 glyph 的余弦≈0 = 完全无关的方向。改为填 **L2 归一化的 DINO 均值**（范数 1.0，与已知行余弦 +0.315，已知行两两之间平均 +0.115）→ 一个"居中的典型字形"。 |
| 4 | 修复 CFG null token 从未可训练 | 见下 |

### 关于 LayerNorm 是否"抹掉"信息

我在初版分析里推测 `char_proj=LayerNorm` 会把"已知行范数 1.0 / 未知行范数 0.39"
这个唯一可辨线索抹掉。**实测并不完全成立**：

```
LayerNorm 后: known norm=19.5583  unknown norm=19.3538  (差 1%)
```

两个范数都被拉到 ≈√384，但仍有约 1% 的残余差异，不是完全不可分。
所以这不是主因，真正的区别在**余弦方向**（噪声行与所有真实 glyph 余弦≈0）。
结论从"LayerNorm 有害"修正为"LayerNorm 让两者量级接近，问题主要在方向"。

### 修复：CFG null token 从未可训练（既有 bug）

`src/model/dit.py` 原写法：

```python
w.requires_grad_(False)
w[-1].requires_grad_(True)     # ← 静默 no-op
```

`w[-1]` 是索引产生的**非叶子**张量，对它的 `requires_grad_(True)` 不会报错也不会生效
（实测 `w[-1].requires_grad` 仍为 `False`）。因此 null token 一直被冻结在
`N(0, 0.02)`，而它承担了 4-way dropout 中约 30% 的样本（cond_drop_one 25% +
cond_drop_all 5%），是 CFG uncond 分支的核心。

修法：在 `LabelEmbedder` 里把 null token 拆成独立的 `nn.Parameter`，
forward 用 `torch.where(null_mask, null_embed, out)` 覆盖 —— 零额外拷贝开销，
且冻结表仍然不产生梯度（省 13.5M 参数的梯度 + Adam 状态）。

## 5. 未采用 / 待验证

- **cross-script 平均**（方案 B）：有效秩只从 34.1 到 35.0，无收益；
  但覆盖率能从 93.4% 提到 99.6%。若将来 unseen 覆盖率成为瓶颈再考虑。
- **unknown flag**（额外一维 0/1 标记）：需要把 `char_embed_dim` 从 384 改成 385，
  改动面较大。当前 `dino_fill_unknown` 已把最严重的"随机方向"问题解决掉。
- **换掉 DINO CLS**：DINO 的 patch tokens（而非 CLS）含更多空间/结构信息。
  这是"正确利用 DINO"的下一步，但要重新抽取特征。

## 6. 真正的零样本瓶颈（仍未解决）

以上改动都是"把现有 DINO 用得更好"，但**没有改变 DINO CLS 本身字符信息薄弱**
这个事实（跨书体 top-1 只有 1.9%~2.6%）。

零样本的根本解法仍然是用 **标准字形 latent（`w_glyph_cond` + `std_glyph_latent_v2/`）**
作为字符结构条件 —— 那里编码的是字的形状本身，而不是一个全局图像描述子。
基础设施已就绪（`src/utils/glyph_latent_v2.py`、`dit.py` 的 `glyph_embedder`），
只差接线。建议在 s20 收敛后单独做 A/B。
