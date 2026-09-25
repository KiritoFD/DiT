# char embedding 条件信号 与 base model 评估指标（调研）— 2026-09-04

> 回答两个问题：
> **Q1** char embedding 怎么做才能同时具备**区分度**与**正确性**？
> **Q2** base model 除 SSIM/LPIPS 外，有哪些**可观测指标**能判断它对后训练（ctrl/REPA）是否好？
>
> 前置：`docs/results/char-embedding-dino.md`（DINO 探测与证伪）、
> `24_condition_signal_decision.md`、`23_dino_signal_diagnosis.md`、`25_dino_embed_direct.md`

---

## 0. 一句话结论

1. **Q1**：对比微调方向可行，但**必须把负对换成"同书家异字"（hard negative）**，
   否则模型会用**书家风格**做 shortcut 区分正负对，重蹈当前 DINO 表"被书体主导"的覆辙——
   区分度上去了，正确性（书家不变性）反而更糟。更优方案是**以标准字形特征为锚**做对齐。
2. **Q2**：分类器路线已不可用（本次已删除 glyph/pixel classifier；`eval_classifier` 同样稀疏）。
   替代核心是 **skel IoU（结构正确性）+ 骨架检索 top-1（字身份正确性）+ 条件解耦实验**，
   三者都不依赖分类器，且 skel IoU 现已在每 ckpt 计算。

---

## 1. 现状：v8a 的 char embedding 到底是什么

`src/train/configs/v8a_s30_base.json`：

| 参数 | 值 | 含义 |
|---|---|---|
| `char_embed_dim` | 384 | |
| `char_dino_embeddings` | `glyph_dino_embeddings_384.npy` | **glyph 级** DINO CLS（script×char 平均） |
| `freeze_char_table` | **true** | DINO 表冻结，不训练 |
| `char_proj_mode` | `mlp` | 仅投影层可训练 |
| `dino_per_script_center` | 1 | 减每个书体均值（缓解书体主导） |
| `dino_fill_unknown` | 1 | 未命中行用 DINO 均值填（避免随机噪声） |
| `num_characters` | 35130 | = 5 script × 7026 |

**即：当前就是"直接用 DINO 特征"路线**——正是 `char-embedding-dino.md` §3 证伪的对象。
文档 §5.5 的新发现（同书家正对 0.817 vs 异字 0.600，+0.217）说明**问题不在 DINO 能力，
在于没有被任务目标重排过**。

已知病症（不需重测）：
- 有效秩 34.1/384（per-script center 后 57）→ 信息高度压缩
- 形近字 cos：土/士 0.986、王/玉 0.975 → 形近不可分
- 跨书家字检索 top-1 **1.2%**
- 83% 最近邻是同一书体 → **被书体主导**

---

## 2. Q1：什么是"好的条件信号"——区分度与正确性

### 2.1 两个维度的精确定义

| 维度 | 定义 | 可测指标 |
|---|---|---|
| **区分度** | 不同字的 embedding 在空间中可分（尤形近字） | 字检索 top-1、形近字 cos、有效秩 |
| **正确性①语义正确** | embedding 对应"该字"而非随机/邻近字 | 跨书家检索 top-1 |
| **正确性②不泄漏** | 推理时只有字 ID，不能依赖 GT 图/骨架 | 设计约束（排除样本级 latent） |
| **正确性③书家不变** | 风格由 `y_callig` 承载，char 不含风格 | embedding 预测书家 acc ≈ 1/44；同字跨书家 cos 高 |
| **正确性④覆盖完整** | 35130 行都有合理值，不存在"随机字符"行 | 未命中行比例（现 eval_unseen 6.6% 落均值行） |
| **正确性⑤训练/推理一致** | 训练与推理取 embedding 的方式相同 | 设计约束 |

> **区分度与正确性会互相拉扯**：单纯优化区分度最容易走的路就是"用书家风格做判别"
> （因为风格差异 > 字形差异，见 §1 的 83% 同书体最近邻）。
> 所以**必须同时监控两个指标**，只报区分度会自欺。

### 2.2 用户方案（方向 A）的正确性审查

方案：正对 = 同字同书家不同图；负对 = 异字。InfoNCE 微调投影头。
实测：正对 0.817 / 负对 0.600，分离度 +0.217，95% 正对>负对 —— 可行。

**⚠ 但存在 shortcut 风险（必须补测）**：
- 正对限"同书家" → 正对天然**同风格**
- 负对若随机取"异字" → 大部分是**异书家** → 负对天然**异风格**
- 于是模型只需学会"风格是否相同"就能分开正负对 → **学到风格判别，而非字形判别**
- 后果：区分度指标（同书家检索）会很好看，但**书家不变性更差**，与 `y_callig` 冗余冲突——
  这正是当前冻结 DINO 表的病根（83% 同书体最近邻）。

**修正（核心建议）**：

| 修正 | 做法 | 作用 |
|---|---|---|
| **① Hard negative（必须）** | 负对重点采样**同书家异字**（同风格、不同字） | 强制忽略风格、只靠字形判别 |
| **② 正对跨书家** | 正对加入**同字不同书家**（实测 0.790，也有信号） | 强制书家不变 |
| **③ 书家对抗（可选）** | 书家分类头 + 梯度反转（GRL） | 显式剥离风格分量 |
| **④ 锚点方案（更优，见 §2.3）** | 标准字形特征作对齐目标 | 从源头风格归一化 |

**补测项（决定方案走向，成本很低）**：
| 配对 | 需测 |
|---|---|
| 同字同书家 vs **同书家异字**（hard neg） | 分离度？若仍显著 → 方案成立 |
| 同字跨书家 vs 同书家异字 | 分离度？检验书家不变前提下的判别力 |

> 如果"同书家异字"下分离度大幅缩水（如 <0.1），说明 0.817/0.600 的分离度主要来自风格，
> 必须先做风格剥离（③或④）再谈对比微调。

### 2.3 更优方案：以标准字形为锚（推荐与 A 并行评估）

矛盾：标准字形（kai/li）**判别性好且无风格**，但覆盖有限；
真迹 DINO **覆盖全**但被风格主导。

**锚点方案**：
- 正对 = (真迹图 DINO 特征, **该字标准字形** DINO 特征)
- 负对 = (真迹图, 其他字标准字形)
- 学一个映射 `f: 真迹特征 → 印刷体锚空间`
- **推理时只用字 ID 查标准字形表**，不经真迹编码器 → **不泄漏、训练/推理一致、书家不变** ✅
- 覆盖：有标准字形的字直接查表；无标准字形的字用已学映射从少量真迹特征预测，
  或退化为 DINO 均值兜底（沿用现有 `dino_fill_unknown` 机制）

与 s28 失败（直接用标准字形特征做条件，0.4476，印刷体域差）的区别：
s28 是**用印刷体特征当条件**（域差），锚点方案是**用印刷体当监督目标**（真迹侧仍是输入），
不引入域差。

### 2.4 评估方案：区分度与正确性必须一起报

| 类别 | 指标 | 现状 | 目标 |
|---|---|---|---|
| 区分度 | 同书家字检索 top-1 | — | >95% |
| 区分度 | 形近字 cos（土/士、王/玉） | 0.986 / 0.975 | 显著低于同字 cos |
| 区分度 | 有效秩 | 34.1（center 后 57） | 显著提升 |
| **正确性** | **跨书家字检索 top-1** | **1.2%** | **20%+**（用户预期） |
| **正确性** | **书家泄漏度**（embedding→书家分类 acc） | — | ≈ 1/44（随机） |
| 正确性 | 同字不同书家 cos | 0.790 | 尽量高 |
| 正确性 | 未命中行占比 | 6.6%（eval_unseen） | 降低 |

**验收判据**：跨书家检索 top-1 ≥ 20% **且** 书家泄漏 acc ≈ 随机。
只有前者没有后者 = 靠风格作弊，不可用。

### 2.5 落地步骤

1. 补测 §2.2 的 hard-negative 分离度（1 小时级，决定方向）
2. 实现 InfoNCE：冻结 DINO 主干，训练投影头（+可选 GRL 书家对抗）
3. 产出判别字表（7026×384，按 glyph 展开为 35130 行）
4. 替换 v8a `char_dino_embeddings`，保持 `freeze_char_table`，跑 v9 预训练
5. 与 v8a（base 0.5061）对比，并按 §2.4 双指标验收

---

## 3. Q2：base model 对后训练有益的**可观测指标**

### 3.1 分类器路线全面禁用（不只是删文件）

**禁用范围**（用户定性：数据太稀疏，准确率完全不行）：
- ❌ 任何在**本项目数据上训练的字符/字形分类器**
  （35130 类 glyph classifier、7765 类 eval_classifier、latent / pixel classifier）
- ❌ **线性探测（linear probing）**——本质仍是训练分类器，同样被稀疏支配
- ❌ 用 `eval_classifier.pt`（`MultiTaskCalligraphyEvalNet`）算的 `ocr_acc`
  及其 `get_features` 提取的 FID 特征——它是 7765 类分类器，
  fame 约 6.6 张/类，且**远程无 ckpt**

| 资产 | 状态 |
|---|---|
| `glyph_classifier` / `train_glyph_classifier` / `eval_glyph_classifier` / `eval_all_classifiers` | **已删除**（本次） |
| `train_pixel_classifier`(+v2)、`_classifier_pixel*.npz`（373MB）、`glyph_classifier_ckpts/` | **已删除**（本次，同一思路、同样稀疏） |
| `eval_classifier.pt`（7765 类，382MB） | 本地存在、远程无；`ocr_acc` / 其特征 **不采信** |
| `train_eval_classifier.py` / `eval_models.py` | 暂留（`src/eval/eval_metrics.py` import 链依赖），待定清理 |

**关键区分——可用 vs 不可用**：

| ✅ 可用（现成模型，非本项目分类器） | ❌ 不可用（本项目数据上训练的分类器） |
|---|---|
| DINO / dinov2（**自监督预训练**） | glyph / pixel classifier（已删） |
| LPIPS（ImageNet 感知模型） | eval_classifier / MultiTaskCalligraphyEvalNet |
| sd-VAE（生成模型组件，已训练好） | 任何 linear probe / 字符分类头 |

→ **评估体系必须建立在"纯 CV 拓扑统计 + 现成自监督/感知模型"之上，
   不引入任何在本项目数据上训练的分类器。**

### 3.2 base 对后训练的价值维度

| 后训练阶段 | 依赖 base 的什么 | 对应可观测性质 |
|---|---|---|
| **ctrl**（骨架条件细化） | 生成字的**结构正确** | 结构正确性（skel IoU / 骨架检索） |
| **REPA**（表示对齐） | latent 表示的**语义质量** | 表示质量（线性探测/检索，限 eval 子集） |
| 两者共有 | 条件可控、**不 mode collapse** | 条件解耦、多样性 |
| 两者共有 | 输出洁净（噪点少） | tv / saltpepper / ink_purity |

### 3.3 推荐指标体系（全链路零分类器训练）

#### A. 结构 / 拓扑类（纯 CV，零模型）⭐ 核心
| 指标 | 计算 | 反映 | 成本 |
|---|---|---|---|
| **skel IoU** | 生成图细化骨架 vs GT 骨架 IoU | **结构正确性**（ctrl 直接受益） | ✅ 已在算 |
| **骨架检索 top-1** ⭐ | 生成骨架 → GT 骨架库最近邻（IoU / 倒角距离） | **字身份正确性**（分类器的正替代） | 需封装，骨架资产齐全 |
| **骨架拓扑距离** | 端点数 / 分叉点数 / 连通域数 的**分布距离**（Wasserstein） | 笔画结构是否合理 | 需实现 |
| **笔画方向直方图距离** | 骨架像素方向（0–180°）直方图 vs GT | 笔画走向分布 | 需实现 |
| **笔画宽度分布** | distance-transform 半宽直方图 vs GT | 笔墨形态（粗/细分布） | 需实现 |

> 这类指标的关键优势：**完全不需要模型，也不需要"认字"**——只比较结构统计量，
> 天然规避稀疏分类问题，且对书法语义直接可解释（端点数=笔画端点、分叉=笔画交叉）。

#### B. 分布质量类（用现成自监督 / 生成模型，不训练分类器）
| 指标 | 计算 | 说明 |
|---|---|---|
| **DINO-FID** | `dinov2_vits14`（**自监督**）CLS 特征 → Fréchet 距离 | 替代 eval_classifier 特征；dinov2 本地+远程均有（88MB） |
| **latent-FID** | sd-VAE encoder → latent → Fréchet 距离 | 无需额外网络，与训练同一表征空间 |
| **LPIPS 多样性** | 同条件多样本两两 LPIPS 的均值 / 方差 | 防 collapse（LPIPS 为现成感知模型） |

#### C. 区分度 / 解耦类（无训练）
| 指标 | 计算 | 反映 |
|---|---|---|
| **聚类可分层度** | 用 DINO / 骨架特征算同字 vs 异字的 silhouette 或 Fisher 比（**只聚类，不训练分类器**） | 区分度 |
| **条件敏感性** | 固定 noise 改 char → skel IoU / LPIPS 变化量 | 条件是否真在控字 |
| **条件解耦** | 改 callig → skel IoU 高（结构不变）+ 风格变；改 char → 反之 | char / style 各司其职 |

#### D. 洁净度（已有）
`metrics_png.py`：tv / saltpepper / lap_var / hf_energy / ink_purity

### 3.4 判据：什么样的 base 对后训练是好的

同时满足（**全部无需分类器**）：
1. **skel IoU 高** —— 生成的字结构对（ctrl 好教）
2. **骨架检索 top-1 高** —— 生成的字身份对（不是"像字但错了"）
3. **拓扑 / 方向 / 宽度分布距离小** —— 笔画结构统计贴合真实（比"认字"更细粒度）
4. **聚类可分层度高** —— 同字聚在一起、异字分开（区分度，无分类器）
5. **DINO-FID / latent-FID 低** —— 生成分布贴近真实
6. **解耦** —— 改 char 结构变、改 callig 结构不变
7. **多样性合理** —— 同条件多样本不过度相似
8. **ssim / lpips 不倒退** —— 基础保真

> 注意：**SSIM 高 ≠ 结构对**。记住某书家的平均外观可以拿高 SSIM，但骨架是错的——
> 这种 base 对 ctrl 是负资产。**skel IoU + 骨架检索 + 拓扑分布距离**是识破这一点的关键，
> 且三者都不需要任何分类器。

### 3.5 对现有流程的建议

- 现早停指标 `ssim_lpips` 只覆盖保真，**不覆盖结构正确性**
  → 把 **skel IoU** 纳入监控 / 选 ckpt（与 ssim 联合），
    并逐步加入**骨架检索 top-1**（分类器替代，优先实现）
- 选 base ckpt 做后训练时，按 §3.4 八项打分，而不是只看 ssim
- **不要**恢复 / 重训任何字符分类器用于评估（稀疏问题无解）

---

## 4. 待办

- [ ] **补测 hard negative**：同字同书家 vs **同书家异字** 的分离度（决定对比微调是否走风格 shortcut）
- [ ] 实现 InfoNCE 微调（冻结 DINO 主干 + 投影头，可选 GRL 书家对抗）
- [ ] 并行评估**标准字形锚点方案**（§2.3）
- [ ] 产出判别字表并跑 v9 预训练，按 §2.4 **双指标**验收（区分度 + 书家不变）
- [ ] **骨架检索 top-1**（分类器替代，优先实现）
- [ ] **骨架拓扑 / 笔画方向 / 笔画宽度 分布距离**（纯 CV，零模型）
- [ ] **DINO-FID**（dinov2 自监督特征；**不用** eval_classifier 特征）
- [ ] **聚类可分层度**（silhouette / Fisher 比，只聚类不训练）
- [ ] 把 **skel IoU** 纳入 base 选 ckpt 判据
- [ ] 决定 `eval_classifier.pt`(382MB) / `train_eval_classifier.py` / `eval_models.py` 是否清理

## 5. 复现

```bash
# 当前 base 配置
cat src/train/configs/v8a_s30_base.json   # char_embed_dim=384, freeze_char_table=true

# 现有评估（每 ckpt 已跑，含 skel_iou）
#   生成: src/eval/in_process_eval.py (GPU 采样存 PNG)
#   指标: src/eval/eval_metrics_daemon.py (CPU, MSE/SSIM/skel_iou)
#   扩展: python src/eval/metrics_png.py --dir <dir> --tag ctrl --n 100
#         (psnr/tv/lap_var/hf_energy/saltpepper/edge_clean)

# DINO 探测工具（沿用）
#   tools/probe_dino_info_content.py / probe_dino_centered.py / probe_condition_injection.py
```
