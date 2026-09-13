# 44 — fame3 std-skel 诊断结论与开关矩阵

> 2026-09-07。上承 13/18/19/20（std-skel 两阶段失败）、35/36（v10a/v10b 单阶段）、
> 41（梯度诊断）、43（五实验链）。本文档记录 **v10b-stdskel-fame3 的根因诊断**
> 与 **全部可消融开关清单 + 实验矩阵**。

## 1. 问题

v10b-stdskel-fame3（std 骨架走 g 通路，glyph_drop 0.25）训练 57500 步：
- SSIM 0.53→0.61（看似正常）
- skel_iou ~0.0077（vs GT，**测错标的**——std-g 下天然 ~0.01）
- follow IoU3 = **0.112**（vs 输入 std 骨架，**极差**）
- 对照：v10b GT-g follow IoU3 = 0.57，v8e 两阶段 = 0.212

## 2. 诊断结论

### 2.1 skel_iou 测错标的

`src/eval/inference.py:278` 的 `_skel_iou(pred, gt)` 是 **vs GT 图**。
std-g 场景下模型被要求写 std 骨架的字形，与 GT 图天然不同 → skel_iou ~0.01 不代表模型坏。
**正确指标 = follow-IoU3 vs 输入 std 骨架**（模型输出骨架是否跟随输入骨架）。

### 2.2 推翻"条件域不匹配"假说

`tools/debug_stdskel_gap.py`（3000 样本 latent 域量化）：

| 对 | cos 相似度 |
|---|---|
| data/skel/std_skel ↔ GT_1px_skel | **0.9017** |
| data/skel/std_skel ↔ GT_target | 0.593 |
| GT_skel ↔ GT_target | 0.626 |

std 骨架 latent 与 GT 骨架 latent **几乎等价**（0.90）。
历史文档 13/18/19/20 号"std-skel 条件域不匹配、上限 0.52"是 **两阶段 ControlNet** 场景的结论，
单阶段 g 通路下 **不成立**。数据不是瓶颈。

### 2.3 真根因 = g 通路没激活

`tools/debug_fame3_stdskel.py` + `tools/peek_glyph_scale.py`：

| 指标 | fame3 (std-g) | v10b (GT-g) | 判读 |
|---|---|---|---|
| glyph_scale 57500 步 | 0.4→**0.3923** | 0.4→**0.5204** (+30%) | fame3 几乎不动 |
| g 注入作用 | **2.7%** | 10.4% | g 信号被稀释 |
| follow IoU3 | **0.112** | 0.57 | 差 5 倍 |
| 逐层 g 衰减 | 4.7%→1.6% (block00→11) | — | 逐层稀释 |

**结论**：glyph_drop_prob = 0.25（fame3 唯一致命配方差异 vs v10b GT-g 的 0.1）
稀释 g 梯度（25% 训练步 g 置零），glyph_scale 学不动，g 通路形同虚设。

## 3. 开关清单（全部可消融，零破坏）

### 3.1 已有开关（train.py argparse，fame3 未启用）

| 开关 | argparse | 默认 | 机制 | 预期 | 何时用 |
|---|---|---|---|---|---|
| `glyph_drop_prob` | `--glyph-drop-prob` | 0.0 | 训练期随机置零 g 条件概率 | 0.25→0.1 恢复 g 梯度 | **d01 主变量** |
| `w_std_mid` | `--w-std-mid` | 0.0 | 去噪中段 (√α∈[0.35,0.75]) 让 pred_x0 锚定 g，MSE | 直接给 g 通路梯度，最针对根因 | mid 变体 |
| `std_mid_alo/ahi` | `--std-mid-alo/ahi` | 0.35/0.75 | 中段噪声带上下界 | 调节锚定范围 | 配合 w_std_mid |
| `glyph_inject_layers` | `--glyph-inject-layers` | 0 | 深层 ZeroAdaLNInjection 注入层数 | 对齐 ControlNet 逐层调制，抗稀释 | inject 变体 |
| `glyph_scale_init` | `--glyph-scale-init` | 0.4 | g 信号强度初值 | 0.6 增强初始 g 信号 | inject 变体 |
| `glyph_init_mix` | `--glyph-init-mix` | 0.0 | xT 混合初始化 (α·randn+(1-α)·std_latent) | 降低 g 通路学习难度 | 备选 |
| `w_skel_head` | `--w-skel-head` | 0.0 | skel_head 预测 latent 骨架 BCE 回归 | — | **std-g 不适用**（skel_gt 是 GT 骨架） |

### 3.2 新增开关

| 开关 | argparse | 默认 | 机制 | 预期 | 何时用 |
|---|---|---|---|---|---|
| `glyph_embedder_depth` | `--glyph-embedder-depth` | 0 | g 编码器深度：0=单层 Conv（现状）；>0=降采样 Conv + N 层 (SiLU+Conv3x3) | 增强 std 骨架特征提取（单层可能提取不出足够结构） | deep 变体 |

**实现**（`src/model/dit.py:487-499`）：
- depth=0：`nn.Conv2d(in_ch, hidden, ks=ps, stride=ps)` — 等价现状，6144 params
- depth>0：`Sequential(Conv降采样 + [SiLU + Conv3x3保分辨率] × depth)` — 增强编码器
- 冒烟验证（`tools/_smoke_glyph_embedder_depth.py`）：depth 0/1/2/3 全部构建 OK，形状正确

## 4. 实验矩阵

### 4.1 已启动

| 实验 | config | 关键变量 | 假设 | 状态 |
|---|---|---|---|---|
| **d01** | `v10b_stdskel_fame3_d01.json` | glyph_drop 0.25→**0.1** | drop 是主因 → follow IoU3 恢复 | 训练中 (step 5200+, 2.6 sps) |

### 4.2 待启动（按 d01 结论决定是否起）

| 实验 | config | 关键变量 | 假设 | 何时起 |
|---|---|---|---|---|
| **mid** | `v10b_stdskel_fame3_mid.json` | + `w_std_mid`=0.05 | 直接给 g 梯度 > 降 drop | d01 不够 / 加码 |
| **inject** | `v10b_stdskel_fame3_inject.json` | + `glyph_inject_layers`=4, `glyph_scale_init`=0.6 | 逐层注入抗稀释 | d01 不够 |
| **deep** | `v10b_stdskel_fame3_deep.json` | + `glyph_embedder_depth`=2 | 编码器太弱 | d01 不够 |

**原则**：单变量消融，不混改。d01 先验证最弱假设（只降 drop）；
若 follow IoU3 未恢复到 ≥0.3，按 mid > inject > deep 顺序加码。

### 4.3 验收标准

- **主指标**：follow-IoU3 vs std 骨架（`tools/debug_fame3_stdskel.py`）
- **辅指标**：glyph_scale 动幅（应 >0.05）、g 注入作用（应 >5%）
- **不用的指标**：skel_iou vs GT（std-g 下无意义）
- **达标线**：follow IoU3 ≥ 0.3（v8e 两阶段 0.212，v10b GT-g 0.57，0.3 是中间合理目标）

## 5. 工具

| 工具 | 用途 |
|---|---|
| `tools/fame3_status.py [exp]` | 一键查看训练/ckpt/eval/follow 诊断（单次 ssh 批量） |
| `tools/debug_fame3_stdskel.py` | g 注入作用 + glyph_scale + 逐层衰减 + follow IoU3 |
| `tools/debug_stdskel_gap.py` | latent 域 gap 量化（cos 相似度） |
| `tools/peek_glyph_scale.py` | 多 ckpt glyph_scale 对照 |
| `_sync_work/_launch_v10b_stdskel_fame3_d01.sh` | d01 启动脚本（tmux + cpu_eval daemon） |

## 6. 状态（2026-09-07 21:5x）

- **d01 训练中**：step 5200，Total 0.374，2.6 sps，GPU 100%/21.5G，ETA 60k ~5.9h
- eval 已产出 step 1000（SSIM 0.5446，skel_iou 0.0073——后者符合预期，测错标的）
- follow 诊断待 d01 训练到 ≥10k 步后用 `debug_fame3_stdskel.py` 复测
- 判读时机：glyph_scale 应明显升（GT-g 曾到 0.52）、follow IoU3 应明显恢复