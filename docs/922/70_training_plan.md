# 922 / 70 — 训练实验计划（S2 风格注入）

> 前置：`60_diagnostics.md` 的 D1/D2/D3/D5/D6。
> 本文只写**要训练什么、怎么训、多少步、看什么指标**。

---

## 0. 诊断结论 → 训练决策映射

| 诊断 | 结论 | 对训练的约束 |
|---|---|---|
| **D5** 表可分性 | 表**本身可分**（effR90 57–74，cosμ≈0） | **不用换表结构**；SupCon 保留但降级 |
| **D1** `dmod` | v15c 表最可分但 `dmod=0.0499`（v13 的 1/13） | **瓶颈在 adaLN 读不动** → 需要**新出口** |
| **D2** 三种注入 | `cfg=1.0` 下 `adaln/ca/xattn` 差 <1% | **SSIM 测不出注入方式差异**，换口径 |
| **D3** 空间分布 | 6/6 run **严格下界为负**（−0.12 ~ −0.36） | **风格没进像素**；`in/bg` 判据**作废** |
| **D6** 因果性 | 通路**活着**（t=0.5–0.9 作用 5–7.8%）；但 `ratio_null` 中位 ≈1.9、方差极大 | 不是"没学会"，是**幅度太小 / 不稳定** |

### ★ 核心判断

> **A 类（表不可分）已排除，B 类（adaLN 读不动）已确认。**
> 但 D6 补充了关键一点：**通路不完全是死的**（t 扫描有 5–7.8% 作用），
> 所以真问题是 **C 类：信号在注入出口被衰减到噪声量级以下**。

对应措施：**新增独立的高增益出口（B1/B2/B3），并强制监控出口范数。**

---

## 1. 实验矩阵（单变量）

三条新通路（代码已实现、默认关闭、zero-init）：

| 通路 | 参数 | 做什么 | 层 |
|---|---|---|---|
| **B1** | `script_film=1` | 书体 FiLM（结构轴） | 全局 |
| **B2** | `spatial_film=1` | 逐位置局部 FiLM | Phase 1 |
| **B3** | `local_ca=1` | `LocalStyleGlyphAdapter` 局部 cross-attn | Phase 2 |

### 阶梯（★ 已核对实际配置文件，2026-09-22）

**已有 7 个配置**（`src/train/configs/v17_s2_s2*.json`），实测内容如下：

| 配置 | `script_film` | `pair_residual` | `spatial_film_rank` | `local_ca` | 变量 |
|---|---|---|---|---|---|
| `s2z_baseline` ★新 | 0 | 0 | — | — | **纯基线**（`hier_style=0`） |
| `s2a_scriptfilm` | 1 | 0 | — | — | B1 书体 FiLM |
| `s2b_pairres` | 0 | 1 | — | — | pair 残差（无 B1） |
| `s2c_full` | 1 | 1 | — | — | 三层表全开（**无局部通路**） |
| `s2d_spfilm` | 1 | 1 | 64 | — | + B2 逐位置 FiLM |
| `s2e_lca_g` | 1 | 1 | — | `at=2,6 q=g` | + B3 局部 CA（query=骨架） |
| `s2f_lca_x` | 1 | 1 | — | `at=2,6 q=x win=5` | + B3 局部 CA（query=特征） |
| `s2g_all` | 1 | 1 | 64 | `at=2,6 q=g` | **全通路** |

**★ 重要事实**：7 个原配置**全部**带 `hier_style=1` + `pair_init=zero`
→ 它们不是"从零开始的阶梯"，而是**"在 S2 三层表基础上加通路"的消融**。
原先**缺一个纯基线**，本轮已补：`v17_s2_s2z_baseline.json`（`hier_style=0`，全通路关）。

**关键差异点（已核对一致）**：
`lr=5e-4`、`max_steps=40000`、`ckpt_every=5000`、`freeze_callig_table=True`、
`callig_emb_pretrained=assets/callig_emb_pretrained_50k.pt`（**45 人表，非 87-pair**）、
`data_csv=assets/train_50k_v2.csv`、`global_batch_size=360`、
`cond_drop_all_prob=0.1`、`cond_drop_one_prob=0.0`、`in_mem_eval=True`。

⚠ **全部从头训练**（zero-init 通路**不在旧 ckpt 里**，拿旧 ckpt"测通路"是无效的）。

### 推荐起跑顺序（成本受限时）

```
1. s2z_baseline   ← 必须最先，验证历史失效可复现
2. s2a_scriptfilm ← B1 单独贡献（最便宜的新通路）
3. s2c_full       ← 三层表（hier_style=1）是否有效
   若 s2a / s2c 的 D3 下界转正 -> 继续 4/5；否则停在 1–3 并复查
4. s2d_spfilm     ← 局部 FiLM
5. s2e_lca_g      ← 局部 cross-attn（query=骨架，最符合"风格作用于笔画"）
6. s2g_all        ← 全通路（天花板）
```

---

## 2. 训练配置

### 2.1 共享底座

**不共享预训练权重**（因为新通路从零学），但**共享数据/超参**：

```json
{
  "model": "DiT-2Cond-S/2",
  "condition_fusion": "factorized_cat",
  "callig_embed_dim": 384,
  "callig_multi_style_k": 4,
  "callig_emb_pretrained": "assets/multistyle_k4_supcon.pt",
  "callig_script_map": "assets/callig_script_id_map.json",
  "cond_drop_all_prob": 0.1,
  "cond_drop_one_prob": 0.0,
  "cond_drop_which_glyph_prob": 0.85,
  "t_sampler": "logit_normal",
  "shift": 1.0,
  "weight_decay": 0.1,
  "w_repa": 0.03
}
```

⚠ **保留 `cond_drop_one_prob=0.0`**（与 v13/v15 一致，保证单变量可比）。
⚠ `cond_drop_which_glyph_prob=0.85` 因 `one_prob=0` **实际不生效** —— 不要再基于它做推断。

### 2.2 步数：**30k–60k**

依据：
- `v13_12ch_post` **只用 22.5k** 就达到 T2 最好（0.514）
- `v13_base_50k` 155k / `v15c_fixed` **210k** → 训最久结果**最差**
- ⇒ **更多步数不解决问题**，问题在通路不在收敛

**建议：主跑 40k，ckpt_every=5000（存 8 个点）。**
若 30k 时 D1 的 `dmod` 没起来 → **直接判该通路无效，不必跑满**。

### 2.3 学习率

- **实测：代码只支持单一 `lr`**（`src/train/train.py` line 1348，无 param_groups 分组），
  所以**无法给新通路单独设 lr**。
- 折中方案：把全局 `lr` 从 `1e-4` 提到 **`5e-4`**（已写入 8 个配置），
  让 zero-init 的新通路更容易在 warmup 期被"点着"。
- ⚠ 代价：主干也同步被抬到 5e-4，可能影响字形稳定性 → **用 `strict SSIM ≥ 0.56` 守门**。
  若 SSIM 掉到 0.56 以下，退回 `2e-4` 并改用"s2a 单独跑长一点"的策略。

---

## 3. 评测口径（★ 已改）

### 3.1 训练侧 in-mem eval —— 已加 ink 指标

`src/eval/in_mem_eval.py` **已改**（4 处）：

| 位置 | 新增 |
|---|---|
| summary 表头 | `ink_ssim_mean, ink_iou_mean, skel_iou_mean` |
| raw 表头 | `ink_ssim, ink_iou, skel_iou` |
| 计算段 | `ink_ssim` / `ink_iou` / `skel_iou(thresh=0.5)` |
| 写入段 | 对应列 + 日志行 |

签名已核对：`ink_ssim(pred, gt, thresh=0.5, win=11, sigma=1.5, pad=2)`、
`ink_iou(pred, gt, thresh=0.5)`、`skel_iou(pred, gt, thresh=0.5)`。

> 注：`tools/batch_eval.py`（离线）**本来就在算** ink_ssim/ink_iou/skel_iou（line 263-273），
> 所以**离线口径一直是对的**，缺的只是训练侧。现已补齐。

### 3.2 主指标（按优先级）

| 优先级 | 指标 | 阈值 | 作用 |
|---|---|---|---|
| **P0** | `strict SSIM` | ≥ 0.56 | **守门**：防字形崩（不追求提升） |
| **P1** | `ink_ssim` / `ink_iou` | 相对基线**有提升** | 主指标（墨迹域） |
| **P2** | **`D3 严格下界`** | **> 0** | **风格是否进像素**（本轮核心验收） |
| **P3** | `D6 ratio_null`（中位） | > 1.5 | 通路是否在区分书家 |
| **P4** | `D1 dmod`（新通路） | 与 v13 基线同量级 | 出口增益是否被激活 |
| ~~弃~~ | ~~`ratio_style`~~ | — | 测离散度不是可分性，**误导来源** |
| ~~降级~~ | `in/bg` | — | **噪声地板同样 ≫1.5，无区分力** |

### 3.3 ★ 必看：新通路出口范数

**把 D1 的 hook 扩展到 `script_film` / `spatial_film` / `local_ca` 的输出**，
监控 `‖out‖` 随 step 的变化。
- 若 `‖out‖` 长期 ≈0 → zero-init 没被"点着"（重演 v15c 的 dmod=0.05）
- **这是 30k 早停判据**

---

## 4. 执行顺序

```
1. ✅ 主 eval 换 ink_ssim          —— 完成（in_mem_eval.py 已改）
2. ✅ D3 全量结论                 —— 完成（6 run，下界全负）
3. 🔄 D6 因果性（16 字 × 4 配置）  —— 进行中
4. ▶  写 v17_s2 配置 + 启动阶梯 0（基线复现）
5. ▶  跑满阶梯 1–4（各 40k，可并行 2 条）
6. ▶  D3 重跑作为验收
```

**闸门**：阶梯 0（基线）必须复现出 **D3 下界为负**。
若阶梯 0 的下界已经为正 → 说明与历史不可比，先查配置差异。


