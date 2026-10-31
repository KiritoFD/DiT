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

## 2. 列（按血统时间顺序）

| # | 列 | 面板 | 来源 |
|---|---|---|---|
| 1 | GT 真迹 | 187 | `docs/106moyi/.../moyi_top10_rf/eval_ours200fix/gt*.png` |
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
