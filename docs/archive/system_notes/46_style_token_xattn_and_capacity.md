# 46. 风格 token 每层注入：设计、实现、resume 兼容性与容量扫描

日期: 2026-09-10
状态: 三实验串行训练中（sty16 → sty32 → sty64，各 30k 从零）
前置: 44（stdskel_sp）、45（c41 系列：干净 41 词表 + SupCon 书家表 + xattn 骨架注入）

---

## 1. 问题：原有注入缺"局部性"与"每层可见"

45 号的 c41x 只启用了 `ZeroCrossAttention`（Q=x 画布, K/V=g_tok 骨架），
书家风格仍走**全局 adaLN**。45 号自己诊断的根因 (b)"callig 只能全局 adaLN
调制，表达不了结体差异"**并未解决**。

用户的要求：在**图片上每一处**（空间位置）综合
**局部字形 + 空间信息 + 书家风格**做调制，且要求"可以从头设计"，
并质疑"两层 attn 是否都必要"。

## 2. 设计结论：两层职能不同，都不能简单去掉

| | Q | K/V | 职能 | 必要性 |
|---|---|---|---|---|
| 层1 风格×字形 `CalligStyleCrossAttn` | g_tok（骨架） | style token | **让骨架本身变形**（结体差异：颜体宽博 / 欧体紧收） | 书法风格核心含结体 → 思想必须保留 |
| 层2 注入 `ZeroCrossAttention` | x（画布） | g_tok | 把条件送进每一层去噪 | 必需 |

单层 context 拼接（K/V=[g_tok; style]）能让每个 x 位置同时看到字形和风格，
但**骨架结构仍是标准印刷体**，产生不了结体差异。

**原实现真正的两个缺陷**：
1. 层1 只在输入层做一次 → 风格调制后的骨架进入深层被稀释；
2. 层2 的 K/V 只有 g_tok → 风格只能"搭骨架的车"间接进入。

**另一个纠正**：`CalligStyleCrossAttn` 的 docstring 断言"固定空间模板与字无关
→ 纯死重（实测 ±0.002）"——该实测是在书家表**塌缩**（pairwise cos 0.323）下做的，
书家向量本身无区分度，**任何依赖它的模块都必然无效**。现在书家表已用 SupCon
修复（cos 0.024），旧结论不宜直接沿用。

## 3. 实现（`src/model/dit.py`）

新增 `GlyphStyleCrossAttn`：每层注入的
**context = [书家化骨架 + 2D sincos 位置 ; 风格 token + 可学习 role]**。

- 每个 x 位置（query）依据自身内容 + 空间位置，同时寻址：
  - 局部字形（骨架 token，与 x 同 16×16 网格，空间对应）
  - 书家风格（style token，attention 权重随 query 位置变化 → **风格局部化**）
- 风格 token 由**共享投影**生成（各层复用），配可学习 `role` 促 N 个 token 分化
- `out_proj` 零初始化 → 初始恒等
- 开关：`--style-token-n`（0 = 关闭并回退旧 `ZeroCrossAttention`，完全兼容）、`--style-role-init`
- 参数量（S/2 实测）：42.39M → 43.99M（n=32，+1.6M）；加层1 书家化骨架 44.97M（+2.6M）

## 4. 冒烟验证（1000 步，fame3，从零）

**① 通路真的在学**（回答此前"梯度为 0"的疑问）
- 训练日志 `[diag] grad-norm: inj_out_proj=0.119`（非 0）
- ckpt 中 12 层 `glyph_injections.*.out_proj.weight` **全部从 0 变成非 0**
  （层0 \|mean\|=1.16e-3，层11=5.80e-4）→ 每层注入都在工作
- 此前冒烟脚本显示 0 是**测试脚本用法问题**，非模型缺陷

**② 风格容量有效（关键诊断，`_chk_style_divergence.py`）**

| 指标 | 实测 | 判读 |
|---|---|---|
| style token 两两 cos（同书家内） | **0.064**（max 0.25） | < 0.3 = 不塌缩 ✓ |
| **有效秩** | **27 / 32** | 接近 n_style = 自由度真实可用 ✓ |
| 书家间 cos | 0.149（max 0.40） | 高于 intra 0.064 = 书家可分 ✓ |

**③ 训练健康**：Diff 0.628 → 0.480；显存 19.4G / 24G。

**超参结论**：`n_style=32` 的有效秩 27 → 容量充足且不浪费；
`role_init=0.02` 已足够促分化；12 层全注入显存可控。

## 5. ⚠ 结构改动后能否 resume 旧 ckpt（实测，`_chk_resume_compat.py`）

用户疑问："结构改了之后居然还能 resume 之前的模型吗"——**是表面兼容，语义不安全**。

| 检查 | 结果 |
|---|---|
| 形状不匹配 | **0** |
| missing（新模型有、ckpt 无） | 仅 5 个新参数：`style_proj.0/1.weight/bias`、`style_role` |
| unexpected | 1 个 `y_callig_embedder.null_embed`（构建参数差异，非结构问题） |
| 构建后 `out_proj` \|mean\| | 0.000e+00（zero-init ✓） |
| **load 后 `out_proj` \|mean\|** | **3.398e-02** ⚠ |

**原因**：`GlyphStyleCrossAttn` 与 `ZeroCrossAttention` 的子模块命名与形状
**完全一致**（`norm_x/norm_c/q_proj/k_proj/v_proj/out_proj`），且旧类的 `ctx_pos`
是 `persistent=False` 的 buffer（不进 ckpt）。因此 `strict=False` 加载**静默成功**，
旧权重直接覆盖进新类。

**风险**：新加的 style token 是随机初始化，而 `out_proj` 拿到了**已训练的非零权重**
→ 随机风格向量经 attention 聚合后直接注入残差流 = **给已训模型叠加随机噪声**。
`initialize_weights()` 里的 zero-init 发生在**构建时**，load 之后不会再执行。

**结论与处置**：
- 若要 resume，必须在 load 后**显式重新 zero-init `glyph_injections.*.out_proj`**；
- 本次容量扫描**全部从零训练**：既保证三组公平可比，也规避污染。

## 6. 容量扫描实验（已启动）

| 实验 | `style_token_n` | 步数 | 起点 |
|---|---|---|---|
| `c41x_sty16` | 16 | 30k | 从零 |
| `c41x_sty32` | 32 | 30k | 从零 |
| `c41x_sty64` | 64 | 30k | 从零 |

其余配方全同 c41x（`DiT-2Cond-Sp/2`、41 词表、冻结 SupCon 书家表、deep2、
repa 0.03、callig drop 0.1、glyph_drop 0.1、batch 128）。
基线：c41x（`style_token_n=0`，无风格 token）。

串行链：`_sync_work/run_sty_scan_chain.sh`（含每实验 `cpu_eval_daemon` 切 watch-root）。
config：`src/train/configs/c41x_sty{16,32,64}.json`。

**判据**：
- 结构不退化：follow-IoU3 / strict SSIM 不低于 c41x
- **风格分化**：不同书家生成图的 LPIPS 距离矩阵 / 聚类可分层度（无分类器方案）
- **有效秩**：随 n 增长的饱和点 = "足够容量"的拐点

## 7. 待验证

- [ ] 三组 Diff / strict / IoU3 曲线对比
- [ ] 风格 token 有效秩随 n 的变化（16/32/64 是否饱和）
- [ ] 风格分化是否真的体现为生成图的书家差异（当前 30k 仍需更长训练确认）
- [ ] 若 resume 旧 ckpt 被需要：在 train.py 的 load 后加 `out_proj` 显式 zero-init 钩子
