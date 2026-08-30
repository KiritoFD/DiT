# 预训练诊断与改进（2026-08-31）

> 触发问题：「推理时 cfg<1 结果显著好，这是否说明我们模型的一些问题？」
> 答案：**是，而且这是一个很强的诊断信号。**
>
> 前置：[19 核心发现](19_core_findings.md)、[21 设计分析与梯度实证](21_design_proposal.md)

---

## 0. 结论速览

| 发现 | 性质 | 状态 |
|---|---|---|
| cfg<1 更优 = 条件是噪声 | **诊断信号** | 已分析，待目检确认 |
| `cond_drop_all_prob=0.05` 偏低 | **可修超参** | 建议 0.10–0.15 |
| `pred_xstart` 在 flow 下恒为 None | **静默失效 bug** | ✅ 已修复并验证 |
| OT-CFM 已实现但未启用 | **未用能力** | 建议启用（4.2ms/step） |
| `shift=1.0` 与注释推理矛盾 | **待 ablation** | 建议试 0.6–0.8 |
| 早停丢掉真正 best | **小 bug** | 建议改 min_delta |

---

## 1. cfg<1 说明了什么

### 1.1 先把 CFG 在 flow 下的语义写清楚

`dit.py:655`：

```python
half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
```

在 **flow matching** 下，模型输出的是**速度** `v = ε − x₀`（直线插值路径）。
代入展开：

```
v_cfg = v_uncond + w·(v_cond − v_uncond)
      = (ε − x0_uncond) + w·[(ε − x0_cond) − (ε − x0_uncond)]
      = ε − [ x0_uncond + w·(x0_cond − x0_uncond) ]
```

**即在 x₀ 空间做插值/外推**：

| w | 在 x₀ 空间的含义 |
|---|---|
| 0 | 纯 uncond（"平均字"） |
| **1** | **纯 cond（标准"无引导"）** |
| \>1 | 沿条件方向外推放大（常规 CFG） |
| **<1** | **从 x0_cond 向 x0_uncond 回拉** |

### 1.2 w<1 更优 = 用偏差换方差

`x0_cond` 与 `x0_uncond` 的统计特性不同：

- `x0_cond`：偏差小（有条件信息），但**方差大**——条件噪声会传导进预测
- `x0_uncond`：偏差大（不知道具体写哪个字，趋向"平均字"），但**方差小**

w<1 是在两者间插值：**牺牲一点偏差，换取方差降低**。
当条件噪声足够大时，这个权衡是划算的 → w<1 更优。

### 1.3 我们有确凿证据表明 char 条件是噪声源

| 证据 | 数值 |
|---|---|
| DINO CLS 有效秩 | 34.1 / 384 |
| 库外检索（同字不同书家）top-1 | **4%**（库内 84%） |
| s22 解冻字 embedding | 0.4664 → **0.4622**（变差） |
| 目检 base 生成 | 约 **25% 明显字形错误** |

**所以 cfg<1 是「条件质量差」的症状，不是独立可调的超参。**
调它只是在给症状止痛，不是治病。

### 1.4 ⚠️ 但必须警惕：cfg<1 可能是"变模糊"骗分

书法场景有个陷阱，SSIM / LPIPS **都无法区分**：

- 清晰但**写错**的字 → SSIM 很低（结构完全不同）
- **模糊**但接近平均轮廓的字 → SSIM 中等（轮廓大致对）

w<1 让输出趋向 uncond 的"平均字"，**很可能是在牺牲字形正确性换取 SSIM**。

**必须用目检验证，不能只看指标。** 已提供工具：

```bash
python tools/sweep_cfg_visual.py --ckpt <best.pt> --config <resolved_config.json> \
    --cfgs 0.0 0.3 0.5 0.7 1.0 --n 8 --out /tmp/cfg_grid.png
```

生成的网格：**行 = cfg（第一行 GT 基准），列 = 同一个字**。

判读：
- 横看某一列：随 cfg 减小，字是否从"清晰但可能错"变成"模糊但轮廓对"？
  - 是 → **cfg<1 是变模糊骗分**，指标虚高
  - 否（字形在各 cfg 下都正确，只是笔画质量变化）→ 是真实质量提升
- 纵看某一行：该 cfg 下 8 个字有几个字形正确？
- 特别关注 **cfg=0**（纯 uncond）：若它也不错，说明条件贡献很小

### 1.5 `cond_drop_all_prob=0.05` 偏低（可修）

`s21_fame_flow_v2.json`：

```json
"cond_drop_all_prob": 0.05,
"cond_drop_one_prob": 0.25,
```

即训练时只有 **5%** 的样本是完全无条件的（null token）。

CFG 的质量**完全依赖 uncond 分支**。标准 CFG 训练通常用 10–20% 的
unconditional dropout，5% 会让 uncond 分支欠训、方向不可靠。

**建议**：`cond_drop_all_prob` 0.05 → 0.10~0.15。

代价：占用训练预算（15% 的样本不学条件）。但由于当前条件本身是噪声，
净收益很可能为正。

### 1.6 一个有用的度量：最优 cfg 值本身就是"条件质量"

| 最优 cfg | 判读 |
|---|---|
| **< 1** | 条件是噪声（当前状态） |
| ≈ 1 | 条件中性 |
| **> 1** | 条件强且可靠 |

**这可以当进度指标用**：glyph cond 若真的提升了条件质量，最优 cfg
应该自然向 1 或 >1 移动。这是一个免费的可观测信号。

---

## 2. OT-CFM（已实现，未启用）

### 2.1 为什么适合我们

标准 CFM 把 noise 与 data **随机配对**。但我们的目标分布是**多模态**的：
给定 (书家, 字)，同一个字有多种合法写法（一字多骨架，实测每字平均
被 ~17 位书家写过，且同一书家也可能有多种写法）。

随机配对 → 概率路径大量交叉 → velocity field 高度弯曲 →
需要更多采样步数、生成质量下降。

**Minibatch OT 重排**让 noise-data 配对更优，使轨迹不再交叉、速度场更平滑
（Tong et al., *Improving and generalizing flow-based generative models
with minibatch optimal transport*, TMLR 2024）。

### 2.2 成本实测（很便宜）

`flow_matching.py:179-186` 已实现，`use_ot=True` 即可启用。
实测（`_probe_flow_internals.py`）：

| batch | 匈牙利耗时 |
|---:|---:|
| 32 | 15.5 ms |
| 96 | 3.1 ms |
| **192** | **4.2 ms** |

batch=192 时 **4.2 ms/step**，而训练 step 本身约 300 ms → **占 1.4%**。

（首次调用含 numba/JIT 预热，故 batch=32 反而更慢；稳态约 4 ms。）

### 2.3 用法

```json
"flow_sampler": "heun",
"use_ot": true
```

建议在 fame 预训练上做 A/B：s21 基线 vs s21+OT。

---

## 3. `pred_xstart` 在 flow 下恒为 None（**已修复**）

### 3.1 问题

`FlowMatching.training_losses` 此前只返回 `{"loss"}`（及可选
`intermediate_feats`），**从不返回 `pred_xstart`**。

而 `train.py` 有一批机制依赖它：

```python
pred_xstart_latent = loss_dict.get("pred_xstart", None)   # flow 下恒为 None
```

受影响的机制**全部静默失效**（不报错、loss 正常下降、但从未生效）：

| 机制 | 作用 |
|---|---|
| **`w_std_mid`** | **把去噪中段预测的 x0 拉向标准字形 latent g——专为「让模型写出正确字形」设计** |
| `w_latent_skel` | latent 骨架损失 |
| `w_latent_canny` | latent canny 损失 |
| `latent_struct_loss_fn` | LatentStructureLoss |

**`w_std_mid` 失效代价最大**：它是一个预训练层面的字形结构锚定机制，
正是我们当前最需要的方向（见 [21](21_design_proposal.md)），
却在 flow 下从未生效过。

> 这是本仓库**第三次**出现「零/None + 宽松检查 = 静默失效」：
> ① `w_glyph_cond` 字典缺失 → 零张量
> ② ControlNet injections 未加载 → zero-init 短路
> ③ `pred_xstart` 缺失 → None
>
> **排查「某个机制看起来没用」时，应优先检查这个模式。**

### 3.2 修复

直线路径下反推是恒等式：

```
x_t = (1−t)·x0 + t·ε
v   = ε − x0
⇒ x0 = x_t − t·v
```

用**预测的** v 代入即得 `x0_pred`（携带梯度）。

- `flow_matching.py`：`training_losses(..., return_pred_xstart=False)`
- `train.py`：仅在 `_need_x0` 时传 `True`（用 `try/except TypeError` 兼容
  `gaussian_diffusion`，后者不支持该参数但本身就会返回）

**默认关闭以保显存**：`pred_xstart` 携带整个 DiT 前向图，
多个 loss 共用会让 autograd 图暴涨（实测 20G→22G）。

### 3.3 验证（`_probe_pred_xstart.py`）

| 检查 | 结果 |
|---|---|
| 默认不返回 | `keys=['loss']`，零额外开销 ✅ |
| 显式请求 | `keys=['loss','pred_xstart']`，shape 正确 ✅ |
| 恒等式 `x_t − t·(ε−x0) == x0` | 误差 2.4e-07 ✅ |
| 常数速度场 0.7 | 与 `x_t − t·0.7` 误差 **0.000e+00** ✅ |
| 模型输出真实速度 → 还原 x0 | 误差 2.4e-07 ✅ |
| 梯度回传到模型参数 | 非零，`requires_grad=True` ✅ |

---

## 4. 其它待验证项

### 4.1 `shift=1.0` 与注释推理矛盾

`flow_matching.py` 的注释自己写道：

> shift < 1 → 步数向低噪声端（t→0）集中，适合**细节/纹理主导**的任务
> 本项目是 32×32 latent（256px 字形），**笔画末端、飞白等细节在 t→0 形成**

推理方向支持 shift<1，但默认值取了 1.0。实测 schedule：

| shift | steps=8 时 t<0.5 的步数占比 |
|---:|---:|
| 0.6 | 55.6% |
| 1.0 | 44.4% |
| 3.0 | 22.2% |

**建议 ablation**：shift ∈ {0.6, 0.8, 1.0}。对细节主导的书法字，
0.6–0.8 很可能优于 1.0。

### 4.2 早停参数丢掉了真正的 best

s21 完整曲线：

| step | SSIM |
|---:|---:|
| 17,500 | 0.4561 |
| 27,500 | 0.4664 ← 记录的 best |
| **30,000** | **0.4670** ← 实际最高 |
| 32,500–40,000 | 0.4591–0.4641（震荡，无进展） |

30k 的 0.4670 > 27.5k 的 0.4664，但提升仅 0.0006 < `min_delta=0.002`，
因此没被记为 best。

**建议**：`early_stop_min_delta` 0.002 → 0.0005。

### 4.3 过拟合

40k 步 × 192 batch ÷ 51,322 样本 ≈ **150 epoch**。
曲线显示 27.5k 步后完全饱和，后 12.5k 步纯属空转。

缓解手段（fame 集内部，不引入外部数据）：
- `tools/aug6.py` 增广（仿射/弹性变换，轻微增广不影响书家风格可辨识性）
- 或降低 `max_steps` 到 30k，把预算省给 ablation

---

## 5. 建议的实验优先级

| # | 实验 | 改动 | 预期 |
|---|---|---|---|
| 1 | **cfg 目检** | 跑 `sweep_cfg_visual.py` | 确认 cfg<1 是真实提升还是变模糊 |
| 2 | **OT-CFM** | `use_ot: true` | 拉直路径，质量提升（1.4% 开销） |
| 3 | **`w_std_mid` 启用** | 修复后配权重试 | 字形结构锚定（此前从未生效） |
| 4 | `cond_drop_all_prob` | 0.05 → 0.12 | uncond 分支更可靠 |
| 5 | `shift` ablation | {0.6, 0.8, 1.0} | 细节主导任务可能受益 |
| 6 | 早停 min_delta | 0.002 → 0.0005 | 不丢真正的 best |

实验 2–6 都是**单配置改动**，可并行排队跑。

---

## 6. 一句话总结

> **cfg<1 是一个诚实的信号：它在告诉我们，我们给模型的条件里
> 噪声多于信息。**
>
> 调小 cfg 是在稀释噪声，不是在学习。真正的解法是把条件本身做好
> （glyph cond），而**最优 cfg 值会随之自然回到 1 以上**——
> 这正好可以当我们进度的度量。
