# 跨血统书家风格 poster + 定量图（2026-11-01）

产物：

| 文件 | 内容 |
|---|---|
| [poster_crosslineage_21.png](../../assets/posters/poster_crosslineage_21.png) | 主交付：9 列 × 21 行（0.62× 压缩版） |
| `poster_crosslineage_21_raw.png` | 原始分辨率 1387×3149 |
| [fig_p125_content_metrics.png](../../assets/posters/fig_p125_content_metrics.png) | 内容/画质指标（按面板分组） |
| [fig_p125_writer_diagnostics.png](../../assets/posters/fig_p125_writer_diagnostics.png) | 书家条件诊断四联图 |

复现：`python scratch/panels/pick_cells.py && python scratch/panels/build_poster.py &&
python scratch/panels/build_charts.py`（取图脚本见 `scratch/panels/*.sh`）。

---

## 1. 为什么是 21 行

需要把**两族模型放进同一张 poster**：近期 P125/abc 链用 100 行 `validation_val100.csv`，
历史 CalliVAE 血统与 Moyi 用 187 行 `eval200_fixed.csv`（`in_mem_eval_sets="eval200fix:...:187"`）。
两族面板不同，若按行号硬拼就是错的。

因此按 **(书家, 字)** 求交集，得到 **21 个两面板共有的单元**：

| 书家 | 格数 | 字 |
|---|---|---|
| 赵孟頫 | 5 | 兜 化 火 穡 纨 |
| 文徵明 | 4 | 㩜 公 墨 記 |
| 王羲之 | 4 | 咽 好 羲 閒 |
| 苏轼 | 3 | 冒 巫 桑 |
| 何绍基 | 2 | 寵 豈 |
| 米芾 | 2 | 已 慈 |
| 褚遂良 | 1 | 怏 |

关键校验：这 21 格在两个面板里的 **GT 源图完全相同（21/21）**，所以整张表只需要一列 GT，
且同一行跨面板可直接对比。行标签里同时印出 `#idx200/idx100`，便于任何人回查原图。

> 索引映射依据：`g{i}.png` ↔ 面板 CSV 第 i 行（0 基），由
> `in_mem_eval_sets="eval200fix:<csv>:<N>"` 的 N 截断保证，`N=187` 与 `N=100` 分别对应两族。
> `build_gt.py` 还会对每格断言 `rows[idx100]` 的 (书家, 字) 与 `cells.json` 完全一致——
> 21/21 通过，这是对索引映射的独立复核。

## 1.1 GT 列的一个坑（已修）

第一版 poster 的 GT 取自 `docs/106moyi/.../eval_ours200fix/gt{i}.png`，**那是减细过的版本，不是真迹**。
同一格（王羲之/羲）三方比对：

| 来源 | ink% | 笔宽 p50 | 笔宽 p90 |
|---|---|---|---|
| 原始真迹 `data/top10_style23/imgs/008161.png` | **28.65** | **11.66** | **24.33** |
| 我们 eval 写的 `gt{i}.png` | **28.65** | **11.66** | **24.33** |
| moyi 的 `gt{i}.png` | 11.12 | 4.00 | 6.32 |

我们 eval 的 gt 与原始真迹**逐项完全一致**；moyi 那份墨量只有真迹的 39%、笔宽只有 1/3，
是骨架化/减细产物（moyi 那套 12ch 流程的 gt 口径不同）。

现在 GT 列直接由 `tools/poster_20261101/build_gt.py` 从面板 CSV 的 `image_path` 取**原始真迹**，
21/21 齐备，整列 ink% 均值 19.7%，与其他列（15.6–20.5%）同量级——这也是发现该问题的信号：
修复前 GT 列只有 9.7%，明显比其他列淡。

## 1.2 第 3 行（赵孟頫/火）看着像「大」——不是取错图，是模型写坏了

对照图：[row03_huo_vs_da.png](../../assets/posters/row03_huo_vs_da.png)

**取图没拿错**，三重验证：

| 验证 | 结果 |
|---|---|
| poster 的 GT 列 `g43` vs 原始真迹 `data/top10_style23/imgs/037129.png` | 平均绝对差 **0.000000**（逐像素相同） |
| eval 自己写的 `gt43.png` vs 同一原始真迹 | **0.000000** |
| 187 行面板 6 个探针（159/147/171/24/85/121）× 2 个 run | 全部 **0.0000**，最近的其它行也有 0.12–0.19 的距离 |
| val100 面板 2 个探针（2/5） | 全部 **0.0000** |

即：`g{i}` ↔ CSV 第 i 行、`gt{i}` = 该行真迹，两个面板都对得上；「火」这一行的真值确实是火。

**模型这边确实写坏了**，量化如下（同一张 128 网格、同一二值化阈值）：

| 列 | ink% | 笔宽 p50 |
|---|---|---|
| 真迹「火」 | **19.8** | **14.1** |
| v80 | 10.8 | — |
| v85A | 15.1 | — |
| S9 | 28.4 | 20.0 |
| S0 | 12.9 | — |
| Moyi 12ch | 14.1 | — |
| P3A | 11.9 | 8.0 |
| A35k | 17.8 | — |
| **A 43,429 (K)** | **9.9** | **7.2** |

绝大多数列的墨量只有真迹的 **50%–70%**、笔宽只有一半左右：笔画整体变细、火的两点退化，
所以整字读起来像「大」。这是**真实的内容缺陷**（也正是 SSIM 只有 0.59 的来源），
不是索引错位、也不是取错了文件。

> 注：形状 IoU / chamfer 这类指标在此处**不可靠**——它们对笔宽差异极敏感，
> 同一行不同列会给出互相矛盾的判读（v80 判成「大」、S9 判成「火」）。因此这里只报
> 可确证的墨量/笔宽，以及逐像素的真值同一性，不靠 IoU 下「是不是写成大」的结论。

## 2. 列（按血统时间顺序）

| # | 列 | 面板 | 来源 |
|---|---|---|---|
| 1 | GT 真迹 | 100 | `data/top10_style23/imgs/*.png`（面板 CSV 的 `image_path`，原始真迹，LANCZOS 到 256） |
| 2 | v80 physics 200k | 187 | `assets/results/v80_callivae_physics_noOT/.../step0200000` |
| 3 | v85A style_ada 30k | 187 | `assets/results/v85A_style_ada/.../step0030000` |
| 4 | S9 style_matrix 50k | 187 | 4090 `style_matrix_20261028/.../S9/evaluations/0050000/heldout187_seed0` |
| 5 | S0 style_focus 85k | 187 | 4090 `style_focus_85k_20261029/.../S0/evaluations/0085000/heldout187_seed0` |
| 6 | Moyi 12ch 50k | 187 | `docs/106moyi/moyi/results/moyi_top10_rf/eval_ours200fix` |
| 7 | P3A 20k | 100 | 4090 `assets/f_20261101/eval/K_P3A/eval_samples_ctrl/step0020000` |
| 8 | A 35k | 187 | 4090 `assets/abc_20261031/.../A/evaluations/0035000/heldout187_seed0` |
| 9 | **A 43,429（K，最近最佳）** | 100 | 4090 `assets/f_20261101/eval/K_A/.../step0043429` |

**未纳入**：
- **Moyi 4ch@80k**（strict_ssim 0.6085，历史峰值）——该 run 只留了一张 40 字官方海报
  `poster_40.png`，没有逐行 `g{i}` 面板，无法逐字对齐，故不放入行轴（避免拼错）。
- **Top10 70k / DiT L20 200k / Top5 Scale**——参考 poster 的这几列，其 `eval_samples_ctrl`
  目录在两机上均已不存在，无法取回原图。

## 3. 定量图的数据来源

**fig_p125_content_metrics.png**
- 左：187 面板组（S9 / S0 / A@35k 读官方 `eval_stdskel_summary.csv`；Moyi 读
  `eval_auto.json` + moyi summary 的 strict 值）。Moyi 无 frag 字段，图上标 `n/a`。
- 右：100 面板组（P3A@20k / A@43429，读 `eval_arm` 的 `eval.json`）。
- **两组面板不同，禁止跨组直接比较**，故分两栏而非并成一条曲线。
- v80 / v85A 的历史汇总在两机都不再留存，因此未画入——不猜数字。

**fig_p125_writer_diagnostics.png**
- (a) Writer-Swap 方向/幅度：P3A 的两次独立测量（900 张 vs 20 字×10 家×2）与 K 的测量并列，
  体现采样方差；标出 direction 目标带 0.50–0.55。
- (b) **K 上的注入尺度消融**（30 字 × 10 家 × 3 种子 = 900 张，本次新跑）。
- (c) T1 书家条件时间线（ΔL_wrong / ΔL_null vs t）。
- (d) T2 关掉单站点后的 writer retention 与 FM loss 变化。

## 4. 本次新得结论：摘掉 32×32 / 16×16 **不是免费收益**

上一轮基于 P3A@20k 的 T2 得出「关掉 site0/1 会让 FM loss 下降（−0.0154 / −0.0085）」，
据此推断「摘掉它们不损失书家能力」。**在 K（A@43429）上按用户要求跑满
30 字 × 10 家 × 3 种子后，该推断被修正**：

| 变体 | direction | amplitude | OCR top1 |
|---|---|---|---|
| 原样 | **0.5571** | **0.6759** | **96.44** |
| 关 site0（32×32/192ch） | 0.5496 | 0.6640 | 95.44 |
| 关 site1（16×16/384ch） | 0.5285 | 0.6538 | 95.78 |
| 关 site0+1 | **0.5223** | 0.6388 | 95.11 |

- 单调恶化，无交互反转：direction **−0.035（−6.3%）**、amplitude −0.037、OCR **−1.3pt**。
- 与 T2 不矛盾：它们**同时**往速度场加噪（T2：关掉 FM loss 降）**和**承载一小部分书家信号
  （T3：关掉生成期 direction 降）。两件事可以同时成立。
- 因此「摘掉 0/1 = 免费」的说法**不成立**，只能说「代价很小但非零」。

**对 F 计划的影响**：F0/F1 两臂按原计划在训练期关闭并冻结 site0/1，这是为了清速度场；
但最终验收必须**同时报告把 gain 恢复为 1.0 后的指标**，否则会把「训练目标」和
「推理能力」混为一谈。为此：
- `tools/diag/t3_writer_swap.py` 已支持 `--gain-scale i:factor`（推理期恢复 gain）；
- `tools/f_20261101/eval_arm.py` 新增 `--off-sites`（推理期关闭，不动权重），
  用于在**同一 checkpoint** 上测「关 / 不关」两种结构的内容指标。

> 注：s1 阶段原本那次 `K_abl_off01` 内容复核**是无效的**——当时 `eval_arm` 还没有
> `--off-sites`，它实际测的还是未消融的 K（指标与 K_A 逐位相同：ssim 0.59457450…）。
> 该缺口现已修复，后续以 `--off-sites 0,1` 重测为准。
