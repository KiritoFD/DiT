# 922 / 00 — 项目进展快照（2026-09-22）

---

## 1. 任务定义

| 项 | 定义 |
|---|---|
| 输入 | ① 标准字形骨架 latent `g` (4,32,32) ② 书家/书体 ID ③ 噪声 |
| 输出 | 书法单字 256×256（SD-VAE f8 latent 4×32×32） |
| 条件 | `g`（字身份 + 结构，**开集**）+ `y_callig`（风格，闭集） |
| **无 char ID** | `no_char_cond=true` —— 字身份完全由 `g` 承担，能写没训练过的字 |
| 主模型 | `DiT-2Cond-S/2`：depth 12 / h 384 / 6 heads / **36.55M** |
| 扩散 | Flow matching + logit-normal-t + Heun 2 阶 |
| 评测 | **strict** n=249（三元组从未出现，真泛化）/ **seen** n=20（记忆指标） |

**为什么不用 char ID**：离散字 ID 是闭集，没训练过的字写不出来。代价是标准字形信息量不足 —— 这正是当前瓶颈之一。

---

## 2. 数据资产（全部在远端 `/root/Workspace/xy/DiT`，不入 git）

### 2.1 训练集演进

| 代 | 行数 | 书家 | 字 | 用它的 run |
|---|---|---|---|---|
| fame3（老数据） | 28,385 | 41 | 4,429 | v10 系列 |
| px60（fame-kxl-tj） | 28,569 | 36 | 4,429 | v12 系列 |
| **50k_v2（当前）** | **50,786** | **45** | 5,821（字符串口径） | **v13/v14/v15** |
| `train_50k_v2_fixed.csv` | 50,786 | 45 | — | v15c_fixed（+751 条简繁修正） |
| `train_50k_v2_fixed` + 229 条异体改通行 | 50,786 | 45 | 5,750 | **09-22 新产出，尚未重训** |

**书体只有 3 种**：楷 / 行 / 隶（草、篆因**没有对应字体渲染标准骨架**被排除）。
HCSU 部分分布：楷 41.5% / 行 41.0% / 隶 17.5%。

### 2.2 关键派生资产

| 资产 | 路径 | 说明 |
|---|---|---|
| 标准骨架 latent g | `data/50k/shards_std/` | 51,036 条；**3px 骨架**（1px 的 VAE round-trip IoU 只有 0.861 vs 0.960） |
| 修复版 g | `data/50k/shards_std_fixed/` | 含 751 条简繁 + 229 条异体修正 |
| REPA DINO 缓存 | `data/dino_cache/50k_v1/` | 51,036 patches，9.3 GiB fp16 |
| DINO CLS 特征 | `assets/dino_cls_50k.npz` | (50,786, 384) —— **09-19 重提，此前全零 bug** |
| 风格表 | 见 §4 | 四代 |

### 2.3 数据侧的四个已知缺陷

| # | 缺陷 | 规模 | 是否已修 |
|---|---|---|---|
| 1 | 简繁错配（csv 标简体 / GT 写繁体） | **751 条（1.48%）** 强确认 | ✅ `train_50k_v2_fixed.csv` |
| 2 | 异体字（如「陞」标成「升」） | 229 条已改通行字 | ✅ 三处同步（csv/std png/shards） |
| 3 | 字标注与 GT 完全错配 | 严格集 6.0% 可疑（四模型交叉验证口径） | ⚠ 部分 |
| 4 | **(书家,字) 对只有 1 张样本** | **28,235 对（75.0%）** | ✗ 数据固有 |

⚠ **75% 的对只有 1 张**是记忆化文献里"记忆最强开关"（Gu et al.：唯一标签使记忆率从 0% 跳到 >65%）。
这也解释了**为什么缩模型没用** —— 问题不在参数量，在条件的信息量。

---

## 3. 时间线（v10 → 现在）

| 日期 | 事件 | 结果 |
|---|---|---|
| 09-10 前 | v10/v11/v12：3 条件 → 去 char 表 → `factorized_cat` | strict 0.5680（n=50） |
| 09-13 | **v13_base**：50k 数据 + 45 书家冻结表 + REPA 0.03 | **0.5703@155k**（n=250）← 主线基线 |
| 09-14 | v13_wd01（wd 0.1） | 110k 峰值 +0.0135，后收窄至 +0.006，**停在 125k 未续** |
| 09-15 | v13_12ch_post（4ch 预训练 + 后训练扩 12ch） | **0.5726 / ink_ssim 0.3898 ← 当前最优** |
| 09-18 | **v14**：87（书家×书体）pair 表 + 层级 SupCon | 三阶段 + fullft 全部 **≈ 0.5703，零增益** |
| 09-19 | ★ 发现 **DINO 提取器全零 bug** → v13/v14 的"DINO 锚定"**从未生效** | 旧表几何是纯标签驱动 |
| 09-19 | **v15a/b/c** 从头训练矩阵（87×K=4 token × 三种注入） | a 0.5699 / b 0.5608@115k / c 未跑完 |
| 09-21 | ★ **v15 的表是坏的**：DINO K-Means 行间余弦 **0.860**（几乎共线） | 多风格零增益的**真根因** |
| 09-21 | **SupCon 重训大表** → 余弦 **0.0019** | `assets/multistyle_k4_supcon.pt` |
| 09-21 | v15b_supcon 起训（tmux `v15bs`，150k） | 中途值 0.5525@70k，**终值未出** |
| 09-21 | **few-shot 管线修好**（12 个静默 bug） | 5/5 主题 `row_pt` > `mean_scaled`；沈周-行 **0.5571** |
| 09-22 | ★ **墨迹域指标**（ink_ssim/ink_iou） | ssim 被白底抬高，排序未变但差距放大 3.6× |
| 09-22 | ★ **OCR 字准率**：生成字可读性只有真迹的 **1/3** | ssim 0.57 掩盖了"字写不对" |
| 09-22 | × 重大 bug：`batch_eval` 把 ckpt args 转 dict → 架构参数静默忽略 | v13 seen 0.5697 → **0.7590**；已修 |
| 09-22 | v15c 跑到 **212,200 步后停**；转跑 moyun 复现 | 之后**服务器 SSH 连不上**（疑似 moyun batch64 搞挂） |

---

## 4. 风格表四代（本轮设计问题的起点）

| 代 | 形状 | pairwise cos | 有效秩 | 信息源 | 结果 |
|---|---|---|---|---|---|
| v13 | (45, 128) 冻结 | 0.0204 | 34.4/45（76%） | SupCon（45 书家标签） | **0.5703**（基线） |
| v14 | (87, 128) 冻结 | 0.0135 | 77.2/87（89%） | 层级 SupCon（87 pair） | 0.5703（零增益） |
| v15 | (87, 4×384) 可训 | **0.860** ✗ | 18.0/87 | DINO K-Means | 0.5699（零增益） |
| v15-SupCon | (87, 4×384) 可训 | **0.0019** ✓ | — | SupCon（87 pair 标签，**关掉 DINO 锚定**） | `v15b_supcon` **未跑完** |

**⚠ 关键断层**：v13/v14 的"健康几何"（cos 0.02）是**标签驱动**的（DINO 锚定从未生效）；
v15 换成真 DINO 特征后反而**共线**（cos 0.860）→ 两次失败指向的不是"表结构"，是**"表里的向量从哪来"**。

---

## 5. 当前卡在哪（2026-09-22 19:00）

| # | 卡点 | 状态 |
|---|---|---|
| 1 | **SSH 连不上远端**（3 次超时） | GPU 上所有实验（v15b_supcon / moyun）状态未知 |
| 2 | **v15b_supcon 终值未出** | 唯一能回答"表里有信息后会不会涨"的实验，卡在 GPU |
| 3 | **两个 eval 口径差 0.126**（v15c_fixed：训练内 0.6961 vs 独立 0.5697） | 未查清；设备差异方向反了 |
| 4 | **训练集改了 229 条异体 + 751 条简繁，尚未重训** | 效果未知 |
| 5 | v13_wd01 停在 125k 未续跑 | 唯一确认有效的增益（+0.006~+0.0135）没吃满 |
| 6 | v15c 停在 212k（未跑满 250k） | — |
| 7 | 古字 OCR 无法可靠识别（「陞」四模型全错） | 数据清洗有上限 |

---

## 6. 代码位置索引

| 组件 | 路径 | 关键行 |
|---|---|---|
| 骨干 + 全部条件开关 | `src/model/dit.py`（1637 行） | `DiT_2Cond.__init__` 446-1055 |
| `LabelEmbedder`（v13 单向量表） | `src/model/dit.py` | 76-135 |
| `MultiStyleEmbedder`（v15 K-token 表） | `src/model/dit.py` | 137-203 |
| `ZeroCrossAttention`（g 的 xattn 注入） | `src/model/dit.py` | 275-341 |
| `GlyphStyleCrossAttn` | `src/model/dit.py` | 343-386 |
| `CalligStyleCrossAttn`（v15b 书家化骨架） | `src/model/dit.py` | 388-435 |
| forward（4-way drop + 条件融合 + 逐层注入） | `src/model/dit.py` | 1144-1428 |
| `DiTBlock`（adaLN-Zero，6 调制参数） | `src/model/modules.py` | 338-368 |
| (书家×书体) 词表构建 | `src/utils/callig_script_map.py` | — |
| 风格表 SupCon 预训练 | `tools/pretrain_multistyle_supcon.py` | — |
| 风格表 DINO K-Means 初始化 | `tools/build_multistyle_k4.py` | — |
| 墨迹域指标 | `src/eval/metrics_ink.py` | — |

---

## 7. 当前最优配方（`v13_base_50k.json`）

```
model            DiT-2Cond-S/2      depth 12 / h 384 / 6 heads / 36.55M
num_calligraphers 45                表 (45,128)，freeze_callig_table=True
condition_fusion  factorized_cat    操作数 = [e_callig(128), e_glyph_vec(128)]
glyph_vec_cond    True              g 池化成向量进条件 c（让 adaLN 看到"在写哪个字"）
glyph_inject      adaln × 4 层      x*(1+s)+t，zero-init
w_repa            0.03              层 8
lr / wd / batch   1e-4 / 0.02 / 360
cond_drop_all     0.1               glyph_drop_prob = 0（内容轴 CFG 用不了）
data              50k_v2（50,786）
```
