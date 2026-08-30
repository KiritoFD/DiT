# 字条件信号决策文档（2026-08-31）

> 状态：**待决策**（用户已选择"先沉淀，不急着 launch"）
> 前置：`23_dino_signal_diagnosis.md`（信号诊断）、`21_design_proposal.md`（梯度实证）
> 本文目的：给下一次 launch 提供清晰的方向选项，避免重复白跑。

---

## 0. 一句话

**DINO CLS 只承载"字身份"（是什么字），不承载"空间结构"（哪一笔在哪）。**
把 CLS 当结构来源是注入错误，不是特征错误。预训练需要的是**结构信号**，
但结构信号的来源必须**训练/推理一致**且**覆盖完整**（不能用标准字形——46.9% 缺口）。

---

## 1. 已确认的事实（不要再重测）

| # | 事实 | 证据 | 来源 |
|---|---|---|---|
| 1 | DINO CLS 身份区分度好 | 不同字余弦 0.0817 | `dino_probe.json` |
| 2 | 字形信息被书体主导淹没 | 同字跨书体最近邻 2.0% | 同上 |
| 3 | **线性变换无法分离字形** | PCA 白化后命中率反降 | `dino_centered.json` |
| 4 | 幅度失衡：书家是字的 1.76 倍 | ‖callig‖=12.9 vs ‖char‖=7.32 | `probe_condition_injection.py` |
| 5 | cfg<1 更优 = 条件有噪声 | flow 下 CFG 在 x0 插值 | 文档 22 |
| 6 | `pred_xstart` flow 下曾恒为 None | 已修复并验证 | 文档 22 |
| 7 | OT-CFM 已实现，仅 4.2ms/step | batch=192 | `_probe_flow_internals.py` |

---

## 2. 三条候选路线（下次 launch 从中选）

### 路线 A：CLS 管身份 + DINO patch tokens 管结构

**做法**：
- CLS（384 维）→ 现有 AdaLN 全局条件（身份）
- patch tokens（16×16×384，**保留空间**）→ 逐层空间注入（结构）

**优点**：
- **100% 覆盖**（无标准字形缺口）
- **自监督**、对未见字泛化好（与预训练同机制）
- 结构信息是空间图，可逐层注入（与 ControlNet 同机制）

**成本**：高
- 需重抽 patch token 特征（每图 256×384，比对 CLS 贵很多）
- 需改模型（加 patch-token 编码 + 逐层注入）
- 存储/加载大（256×384 vs 384）

**风险**：
- patch token 也被书体主导？需先验证（同 CLS 的分离实验）
- 注入方式仍在验证中（单层 vs 逐层）

### 路线 B：只修幅度，验证身份信号是否够（**最便宜，建议先做**）

**做法**：
- 只加 `callig_scale`/`char_scale` 可学习幅度（已实现）
- 从 s21 warm-start，不带标准字形，开 OT
- 看最优 CFG 是否从 <1 回移到接近 1

**优点**：
- 改动最小（+2 参数），验证干净
- 直接回答"幅度失衡是不是主因"

**判读**：
- CFG 回移到 ≈1 → 身份信号够，问题在注入 → 走路线 A
- CFG 仍 <1 → 身份信号也不足 → 更需要路线 A

**成本**：一个实验周期（约 1-2 天）

### 路线 C：标准字形只用于后训练/推理

**做法**：
- 预训练：专注基础表征（CLS + 书家，幅度修正）
- 标准字形 latent → 只用于 ControlNet 之类后训练（推理时从字 ID 查）

**优点**：预训练干净，标准字形覆盖问题被隔离到后训练

**成本**：需两阶段训练

---

## 3. 一个必须回答的问题：为什么 base 需要结构信号？

这不是显而易见的问题。给的理由：

**字形的核心信息是空间结构。** 没有结构信号，模型只能从
"这个字是什么字"（身份）+ 统计先验去**猜测**笔画位置。

- 常见字（训练多）：靠记忆能猜对 → 但这是记忆，不是理解
- **罕见字/未见字**：无记忆可依 → 必须靠结构推理 → **当前做不到**

证据：base 目检 25% 字形错误；未见字能力完全未评估（`eval_unseen` 集存在未用）。

**所以结构信号的价值主要在泛化（罕见字/未见字），而不是提升已见字。**
如果目标只是已见字好看，加结构信号的收益有限。

---

## 4. 建议的决策路径

```
第一步：跑路线 B（便宜，1-2 天）
   目的：确认幅度失衡是不是主因，顺便给 route A 的注入方式提供基线
   |
   第二步：根据 B 结果决定
   ├─ CFG 回移 → 身份信号够 → 路线 A（patch tokens 做结构）
   ├─ CFG 仍 <1 → 身份也不足 → 路线 A 必须做，且要强结构信号
   └─ 同时：建立未见字评测（eval_unseen），否则无法衡量结构信号价值
```

**不建议直接跳路线 A**：patch token 注入成本高，且还没验证它是否也被
书体主导。先用便宜的 B 把信号基线摸清。

---

## 5. 已实现的代码（本次会话产出，未 launch）

| 改动 | 文件 | 状态 |
|---|---|---|
| `callig_scale`/`char_scale` 可学习幅度 | `src/model/dit.py` | ✅ 已实现 |
| glyph_drop_prob 防门控 | `src/model/dit.py` | ✅ 已实现 |
| 逐层 ZeroAdaLNInjection | `src/model/dit.py` | ✅ 已实现（`glyph_inject_layers`） |
| pred_xstart flow 修复 | `src/loss/flow_matching.py` | ✅ 已实现并验证 |
| OT-CFM 支持 | `flow_matching.py`（已有） | 需配置开启 |

这些改动都已验证不破坏 s21 ckpt 加载（`_verify_ckpt_load.py`：missing=2 仅新参数，unexpected=0）。

---

## 6. 工具索引

| 工具 | 用途 |
|---|---|
| `tools/probe_dino_info_content.py` | CLS 信息含量（几何度量） |
| `tools/probe_dino_centered.py` | 去书体主成分后信号可分离性 |
| `tools/probe_condition_injection.py` | 幅度/区分度/边际贡献 |
| `tools/verify_glyph_gradient.py` | 逐层注入梯度通路 |
| `tools/sweep_cfg_sharpness.py` | cfg 扫描 + 清晰度（防变模糊骗分） |
| `tools/sweep_cfg_visual.py` | cfg 网格目检图 |
