# 数据清洗现状调查与方案（2026-09-01）

> 全量扫描 fame 数据集（51322 训练 + 500 评测），量化三类污染真实分布，
> 建立可靠评估标准，给出几种 CV 方案与初步实测对比。

## 0. 一句话总结

**之前的"脏污"判定（foreign_area_ratio / main_frac）严重误判**——把正常的
草书/飞白多连通域（"雍"字 main_frac=0.08 干净）误判为污染。
**真正需要修复的是反相图 / 黑边框 / 边界墨块 / 散落噪点 / 严重墨块覆盖字形**。

## 1. 现状（全量扫描 train=51322, eval=500）

工具: `tools/scan_image_pollution.py`（连通域 + 边缘环带）。
输出: `5script/scan_train_pollution.csv` / `5script/scan_eval_pollution.csv`。
训练集与评测集分布几乎一致（数据同分布）。

### 1.1 基础形态
| 指标 | 训练 mean / p95 / max | 评测 mean / p95 / max |
|---|---|---|
| 墨量 ink_ratio | 0.235 / 0.406 / 0.866 | 0.235 / 0.418 / 0.758 |
| 连通域数 n_cc | 6.8 / 21 / 446 | 6.9 / 18 / 146 |
| 主连通域占比 main_frac | 0.711 / — | 0.724 / — |

### 1.2 视觉确认的污染类型与占比
| 污染类型 | 判定 | 占比 (train) | 视觉确认 |
|---|---|---|---|
| **反相图** | ink>0.5 + n_cc=1 | 0.24% (122) | 白字黑底（如 8668）→ 反转 |
| **黑边/实心条** | border_bar=1 | 1.69% (866) | 装裱/扫描黑边（4063, 238302）→ crop+pad |
| **边界墨** | edge_ink_ratio≥0.20 | 2.84% (1455) | 边缘大片墨（197305 鸡字黑框）→ crop |
| **严重脏污** | foreign≥0.30 + main_frac<0.3 | ~0.5% | 大片黑块覆盖字形（280062 隔字）→ 难修复 |
| **散落噪点** | small_a≥0.005 | 1.59% (816) | 背景洒金/扫描噪点（148960 隅字）→ 删小CC |
| **碎片严重** | main_frac<0.5 | 23.48% | **多為草書/飛白（99595 雍字干净）—— 非污染** |
| **非主连通域多** | foreign≥0.10 | 22.89% | **多為多筆畫字（正常）—— 非污染** |

### 1.3 关键认知修正
- **main_frac 低 ≠ 污染**：草书/飞白天然多连通域。"雍"字 main_frac=0.08 但完全干净。
- **foreign 高 ≠ 脏污**：正常多笔画字（如"清"三点水、"时"左右结构）天然 high foreign。
- **真正的"超出字形"墨才是脏污**，需要**参考字形（标准字形 kai/li）**做对比。

## 2. 评估标准（可量化）

| 指标 | 定义 | 用途 |
|---|---|---|
| **主连通域保留率** | clean_main_frac / orig_main_frac | 笔画误伤检查（<1.0 表示误删了主字） |
| **字形 IoU** | clean 墨 ∩ ref 墨 / (clean 墨 ∪ ref 墨) | 修复后与标准字形的相似度 |
| **墨量变化率** | (clean_ink - orig_ink) / orig_ink | 不应过大（<5%），过大=误删 |
| **外缘墨率** | bbox 外 1px 环带内墨/环带面积 | 修复后黑边残留率（<1%） |
| **碎片度** | n_cc_after / n_cc_orig | 不应大幅增加（连通域数增加=误删） |
| **小CC数** | area<0.05% 的CC数 | 噪点残留率（应→0） |
| **SSIM** | clean vs ref (若有) | 整体相似度 |

## 3. CV 算法方案

### 方案 A：保守（沿用现有 clean_gt_images.py）
- 1) 删 `area<0.05%全图` 且与主字不连通的非主 CC
- 2) 边缘 8px 环带内、非主字、且**接触边界**的墨→填白
- 3) 反相：ink>0.5 + n_cc≤2 → 反转
- 4) 黑边：边缘 4 边有连续黑条 → crop + 白边 pad
- ✅ 已有实现（`tools/clean_gt_images.py`），用户之前跑过修了 1430 张
- ⚠ 误伤风险：飞白/草书的小笔画可能被当噪点删

### 方案 B：基于参考字形 bbox（已实现 `tools/clean_v2.py --scheme B`，推荐）
- 用标准字形 kai 渲染图的**墨 bbox** 作参考（预计算于 `5script/..._sync_work/std_glyph_bbox.json`）
- **bbox 外**（膨胀 12px）的非主墨 → 删（明确脏污/边界污染）
- **bbox 内**的非主墨 且面积很小(<0.05%) → 也删（字内噪点）
- **bbox 内**的中等/大面积非主墨 → 保留（正常飞白/连笔/分离笔画）
- ✅ 避免误删字内飞白（飞白在字内，与参考字形重叠）
- ✅ 边界大片黑（超出字形 bbox）在边缘，被清掉
- 实现: `tools/clean_v2.py --scheme B` → `5script/clean_report_train_B.csv` / `final_imgs_256_clean_v2/`

### 方案 C：连通域 + 空间拓扑（中等保守）
- 主连通域 = 最大 CC
- **空间距离判定**：非主 CC 中心 → 主 CC 中心的归一化距离 > 阈值 → 噪点
- 边界环带（8px）内的非主墨 → 边界污染
- 避免单纯按面积（飞白/草书小笔画可能面积<阈值但不该删）
- 比 A 多一个"距主字远"判定
- 待实现

### 方案 D：深度学习（远期，不在本轮）
- 用 inpainting / restoration 网络
- 成本高、过拟合风险大、需训练数据
- 不推荐本轮

## 4. 实测对比（A / B 改进 / C 三方案）

### 4.1 小样本(200张) 三方案对比
| 指标 | A（基线） | **B（参考字形bbox）** | C（空间拓扑） |
|---|---|---|---|
| 墨量变化 mean | -0.0246 | **-0.0104** | -0.0835 |
| \|ink_change\|>5%（过度删除） | 13.5% | **3.5%** | 33.0% ❌ |
| 主连通域保留 | 0.99729 | **0.99729** | 0.99729 |
| 小噪点残留 | 0.085 | **0.010** ✅ | 0.040 |

**结论：方案 B 最优**——噪点清除最干净（残留 0.010，比 A 好 8×），过度删除最少（3.5%，比 A 好 4×，比 C 好 10×）。
方案 C 因"距主字远即删"会误删正常分离笔画（如"氵"、左右结构偏旁），最差，**弃用**。

### 4.2 全量方案 B（train=51322, eval=500，`tools/clean_v2.py --scheme B`）
- **实际修复 18785 / 51322 (36.60%)**；评测 192 / 500 (38.40%)
  - 其中反相反转 103 (0.20%)；黑边crop 959 (1.87%)
- **质量**：
  - 主连通域保留 mean=0.9897（train）/ 0.9815（eval）；严重误删(main_keep<0.95) 816 (1.59%)
  - \|ink_change\|>5% 仅 1278 (2.49%)；>10% 仅 1013 (1.97%)
  - 小噪点残留 mean=0.0027（几乎清零，>0 仅 55 张 0.11%）
- 产物：`final_imgs_256_clean_v2/`（仅写被修改图）、`5script/clean_report_train_B.csv`

### 4.3 与用户后续 v8 清洗的关系
- 用户于 9/2 又做了一轮清洗：`5script/train_fame_clean_v8.csv` → `final_imgs_fame_v8/`（51322 行）。
- 这是与 clean_v2(B) 不同的另一尝试，二者方法/阈值可能不同；**数据清洗是持续迭代项**，
  本轮(28)的扫描+方案B提供了一套**可量化、基于参考字形的评估标准**与基线工具。

## 5. 复现

- 扫描: `python tools/scan_image_pollution.py --csv 5script/train_fame.csv --out 5script/scan_train_pollution.csv`
- 方案B清洗: `python tools/clean_v2.py --csv 5script/train_fame.csv --img-root final_imgs_256 --out-root final_imgs_256_clean_v2 --scheme B --report 5script/clean_report_train_B.csv`
- 现有清洗(旧): `python tools/clean_gt_images.py --audit ... --blacklist ... --img-root final_imgs_256 --out-root final_imgs_256_clean`
- 样本图: `_sync_work/pollution_samples/`（各类 8 张，验证用）
- dashboard: `tools/pretrain_eval_dashboard.html`（base/skel/repa 分组 eval 对比）
- grid: `tools/make_base_model_grid.py --group all --step 30000`（生成改进链路视觉对比）
