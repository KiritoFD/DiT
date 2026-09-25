# 65. 实验计划 v2：推理参数 + 模型（容量 / 注入 / 其他）

> 日期：2026-09-16
> 相关：[54](54_results_and_insights_20260913.md)、[59](59_12ch_and_white_zero_retrospective.md)、
> [60](60_what_is_actually_useless.md)、[61](61_ref_moyun_code_audit.md)、[64](64_experiment_backlog.md)
>
> **状态：本文只做规划，未改任何代码、未启停任何训练。**
>
> 结构：第一部分 = 推理参数（**用当前 ckpt，零训练**）；第二部分 = 模型，分
> **容量（宽度/深度）**、**注入方式**、**其他（12ch + drop 比例）** 三方面。

---

## 0. ⚠ P0 前置修复 —— 不修，下面所有实验都白跑

### 0.1 【严重】`factorized_cat` 下条件 dropout 完全失效（本轮新发现）

`src/model/dit.py:1015`：

```python
if (self.condition_fusion in ("factorized_add", "xl_highdim")   # ← 缺 "factorized_cat"
        and self.training
        and (self.cond_drop_all_prob > 0 or self.cond_drop_one_prob > 0)):
    r = torch.rand(...)
    callig_drop = ...
```

v12 的 `condition_fusion = "factorized_cat"`（日志已确认 `fusion=factorized_cat`）
→ **该分支从不进入** → `callig_drop is None` → `y_callig_in = y_callig`（第 1026 行）

**后果**：

| | 期望 | 实际 |
|---|---|---|
| `cond_drop_all_prob=0.1` | 10% 样本走 uncond 分支 | **0%** |
| null embedding | 被训练 | **从未被更新**（初始化 std=0.02 后原封不动） |
| eval `cfg=0.7` | 在 uncond/cond 之间插值 | **向一个未训练的向量插值** |

**影响范围**：v12、v13(XS/2)、v14(S320/2)、v15(XS6/2)、v16(sep) 全部继承（都是 cat）。
**v12b（add）不受影响** → 我上一轮提的 2×2 会被这个 bug 污染，必须先修。

**修复**：把 `"factorized_cat"` 加进第 1015 行的元组。**一行，零风险。**

### 0.2 静默丢弃（第 4、5 次踩同一个坑）

| 键 | 状态 | 后果 |
|---|---|---|
| `repa_layers` | **未注册**（train.py 无 `--repa-layers`） | config 值被丢，永远走硬编码 `or (8,)` |
| `gpu_eval_cfg` | **未注册**（只在 train_controlnet/repa/skel_1cond 里有） | config 的 0.7 被丢；实际生效的是 **`eval_cfg`**（train.py:1939，v12 config 第 64 行也是 0.7，所以巧合一致） |

⚠ **注意**：现在真正控制 in-mem eval 的是 `eval_cfg`（`in_mem_eval.py:361` 取 `args.eval_cfg`），
**不是** `gpu_eval_cfg`。改 cfg 要改 `eval_cfg`。

### 0.3 性能回归

5.28 → **3.92 step/s**（−26%），显存 18.83 → **20.15/20.30 G**。
已排除 compile 失效 / LPIPS 占显存 / CPU 瓶颈 / 功耗墙。
最可能是显存逼近上限导致 allocator 频繁同步。**下次启动加
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`**（零风险，一行环境变量）。

**下面所有"X 小时"的估算按当前 3.92 step/s 给，修好后可 ×0.74。**

---

# 第一部分：推理参数扫描（用当前 ckpt，零训练）

## 1.1 真正生效的旋钮（已逐个核实）

| 参数 | 注册位置 | v12 当前 | 说明 |
|---|---|---|---|
| **`eval_cfg`** | train.py:1939 | **0.7** | ⚠ in-mem eval 读的是这个（`in_mem_eval.py:361`），不是 `gpu_eval_cfg` |
| **`eval_steps`** | train.py:1937 | **50** | ODE 步数；heun = 2 NFE/步 → 实际 100 NFE |
| **`flow_sampler`** | train.py:1783 | **heun** | `euler` / `heun`。help 里写明"等 NFE 下 Heun@25 优于 Euler@50" |
| `--eval-self-cond` | train.py:2041 | **false** | 两遍自条件：pass-1 预测骨架回喂 pass-2 |
| `--eval-blend-alpha` | train.py:2045 | **0.0** | 自条件骨架混合（0=纯预测，0.5=各半） |
| `--in-mem-eval-lpips` | train.py:2036 | **true** | 本轮新接，已出数 |
| `--t-sampler` / `t-mean` / `t-std` | train.py:1773/1777/1779 | logit_normal / 0.0 / 1.0 | **训练侧**，改了要重训；推理不受影响 |
| `--diffusion-type` | train.py:1766 | **flow** | 支持 `ddpm` / `flow`（默认 ddpm） |

## 1.2 ⚠ 一个必须先说的前提

**因为 0.1 的 bug，v12 的 uncond 分支从未被训练 —— 在它的 ckpt 上扫 cfg 是误导性的。**
null embedding 停在 std=0.02 的随机初始化上，`out = uncond + cfg·(cond − uncond)`
实际上是"向一个随机方向外推"，cfg 越大越糟 —— 这是 bug 的产物，不是 cfg 本身的属性。

**所以扫描分两处做：**

| 对象 | 目的 |
|---|---|
| **v11 best ckpt（152.5k，drop 正常）** | 得到**真实的 cfg 响应曲线形状** —— 这条曲线是通用结论 |
| **修复后重训的 v12**（或用 v12b） | 验证形状是否复现 |

如果时间只够一处 → **先做 v11**，那里能拿到干净的因果结论。

## 1.3 扫描设计

### Stage 1：cfg 响应曲线（最关键，约 40 min）

```
cfg ∈ {0.0, 0.3, 0.5, 0.7, 1.0, 1.3, 1.7, 2.2}   # 固定 steps=50, heun
```

**必须包含两个锚点**：

- **cfg = 1.0**：纯条件，无任何引导 → 这是"模型真实能力"的基线
- **cfg = 0.0**：纯 uncond → 直接暴露 uncond 分支训得好不好。
  **如果 v12 上 cfg=0 的输出是噪声/墨团，就再次证实 0.1 的 bug；如果 v11 上 cfg=0 还能出字形，说明 uncond 分支健康。**

判读（三种典型形状）：

| 曲线形状 | 结论 | 下一步 |
|---|---|---|
| 峰值在 cfg > 1 | CFG 有效，当前 0.7 欠配置 | 改 `eval_cfg`，零成本涨分 |
| 峰值在 cfg = 1.0 | **CFG 完全无效** → uncond 分支没训好 | 立刻做 drop 实验（§2.3.2） |
| 单调下降 | CFG 有害 | 同上，且当前 0.7 正在伤害指标 |

### Stage 2：NFE 公平比较（约 30 min）

```
(euler, steps=50) vs (heun, steps=25)    # 都是 50 NFE
(euler, steps=100) vs (heun, steps=50)   # 都是 100 NFE ← 当前档位
(heun, steps=100)                        # 200 NFE，看是否已收敛
```

**判据**：若 heun@50 与 heun@100 差异 < 0.002 → 50 步已足够，不需要加；
若差异明显 → 我们一直在"欠采样"下评估所有历史模型，**所有历史结论都要打折**。

### Stage 3：噪声与样本量（约 20 min）

- 3 个 seed × 当前最优档位 → 得到"eval 本身的复现噪声"σ_seed
- 目的：确定 n=50 的 strict 在多大的差异上才可判显著（§3 判据纪律要用 σ_seed 修正）

### Stage 4：自条件（约 20 min）

`--eval-self-cond 1` × `--eval-blend-alpha ∈ {0.0, 0.3, 0.5}`
纯推理侧改动，doc 60 列为"未 A/B"，成本极低。

### 成本与执行方式

- 单次 eval（seen 10 + strict 50，50 步 heun）≈ 1–2 min
- 全 Stage 1–4 ≈ **2h**（含 LPIPS 的 CPU 开销）
- **执行方式：独立进程加载固定 ckpt 跑，不干扰训练**。
  候选工具 `src/eval/eval_ctrl_ckpt.py` / `gpu_batch_eval_v2.py`；
  大概率需要一个 ~10 行的 sweep 驱动（遍历 cfg×steps，写一张 CSV）。

## 1.4 这一部分的判据

主指标 **LPIPS**（结构敏感，本轮已验证可用：v12 @60k = 0.379）+ strict + seen。
**不看单点，看曲线形状**（峰值位置、平台起点）。

---

# 第二部分：模型

## 2.1 容量（宽度 / 深度）

### 历史证据

| 来源 | 结论 |
|---|---|
| doc 54 §3.2 | 容量决定 **seen** 上限（S/2 0.52 / M/2 0.571 / Sp/2 0.67–0.76），**strict 各容量都在 0.51–0.57** |
| doc 54 §5 | "要视觉质量上 Sp/2，要 strict 容量不是瓶颈" |
| **本轮 v12** | **挑战了这条**：S/2 在 60k 时 strict 0.5366 vs M/2 0.5324（+0.004），seen −0.018（t=−2.43 显著），gap 增速只有 v11 的 **48%** |

⚠ doc 54 的容量表来自 **v10b/v11 旧配方**（adaLN + add + 1px/3px + 12ch）。
我们现在是 cat + glyph_vec + px60，且 v12 的 seen 已经跑到 0.5377@60k ——
**旧表不能迁移**（我上一轮拿它反对缩容量是错的，已作废）。

### 阶梯（配置已生成，与 v12 只差 `model` 一个字段）

| 配置 | depth | h | params | FLOPs vs M/2 | 100k 步 |
|---|---|---|---|---|---|
| v12（基线，在跑） | 12 | 384 | 37.24M | 0.790× | ~7.1h |
| **v13 XS/2** | **8** | 384 | 26.61M | 0.527× | ~4.7h |
| **v14 S320/2** | 12 | **320** | 26.45M | 0.549× | ~4.9h |
| v15 XS6/2（条件性） | 6 | 384 | 21.29M | 0.395× | ~3.5h |

**v13 与 v14 参数量几乎相同（26.61 vs 26.45M），一个缩深度一个缩宽度 —— 天然回答"该缩哪个"。**
理论先验：渲染任务是"浅"的（局部特征 → 少数几轮全局协调 → 局部输出），
且 h=384 对 4 通道 latent 已是 96× 扩张 → **深度比宽度更可能过度配置**（倾向 v13 胜出）。

### 怎么做

1. v13、v14 各跑 **40k** 先做短程筛选（各 ~1.9h）
2. 与 v12 @40k 逐样本配对，看 **gap 斜率** 与 **strict/LPIPS 水平**
3. 只有 Top-1 才继续跑到 100k+；若 v13 与 v14 都显著劣化 → 容量下界在 26–37M 之间，停
4. 若 v13 不劣化 → 再上 v15（XS6/2, 21.3M）

### ⚠ 一个反直觉的风险

缩容量会**收窄 GEMM 形状 → MFU 反而下降**。所以 FLOPs 省 47% 换到的实际提速
会**小于** 47%（v13 现在估 4.7h/100k 是线性外推，乐观）。实测再校正。

---

## 2.2 注入方式

### 现状：g 有三条通路，职能不同（不该重复）

| 通路 | 形式 | 职能 |
|---|---|---|
| ① 输入层 token-add | `x += glyph_scale · g_tok` | 给网络一个"起点"（粗定位） |
| ② 逐层注入 | 4 层 `ZeroAdaLNInjection`（`glyph_inject_layers=4`） | 每层都能看到内容，抗深层稀释 |
| ③ 条件向量 concat | `c = t_emb + Linear(concat[e_callig, e_glyph_vec])` | 让 adaLN 的全局 scale/shift 知道写什么字（**新增**） |

③ 补的是一个真实的洞：原先 `c = t_emb + callig_proj(e_callig)`，
**adaLN 调制分支从来看不到内容**（每个 block 的 scale/shift/gate 都不知道在写哪个字）。

**实测 g 通路是活的**：本轮在 v12 ckpt 上旁路编码器两层 3×3 卷积，
输出变化 **1.619**；完全去掉 g 是 **1.684** → 比值 **0.96**。
且 g 的相对作用 **1.68**，远高于 doc 44 时代的 2.7%（修掉 `glyph_drop_prob` 之后确实活了）。

### 候选矩阵

**A 组：adaLN 层数与位置**

| 变体 | 配置 | 说明 |
|---|---|---|
| A0 | `glyph_inject_layers=4`（现状） | doc 54 判"有用（保留）"，但 **4 vs 12 从未单变量对照** |
| A1 | `layers=12` | 每层可见，+算力 |
| A2 | `layers=4` 但放在**深层**（后 4 层）而非前 4 层 | 位置消融，零额外算力 |

**B 组：xattn 实现**

| 变体 | 说明 | 成本 |
|---|---|---|
| B1 | 现有 `ZeroCrossAttention`（Q=x, K/V=g_tok），12 层 | **+7.2% FLOPs** |
| B2 | xattn 只放**后 6 层** + adaLN 放前 6 层 | 混合，成本居中 |
| B3 | xattn **共享 K/V 投影**（跨层复用） | 省参数，理论损失小 |
| B4 | **token-concat**：`[x; g]` 一起 self-attention，末尾切片 | 不用新模块，但序列长度 ×2 |
| B5 | 现状 adaLN（对照） | — |

⚠ **xattn 的历史证据是自相矛盾的**（doc 60 §4 自己承认打脸）：
doc 54 判"xattn 无用"（依据 80k 持平），但**历史最佳线 0.5680 用的正是 xattn12**。
正确判定是"**不确定**"——xattn 参数更多，80k 持平**恰恰可能是没训够**。

**C 组：条件向量融合**（③ 那条）

| 变体 | 说明 |
|---|---|
| C1 | `factorized_cat` + `glyph_vec`（v12 现状） |
| C2 | `factorized_add` + `glyph_vec`（**v12b，操作数集合完全相同**） |
| C3 | 无 glyph_vec（退化，验证 ③ 是否真有价值） |

### 怎么做（**不要做全因子，单卡跑不起**）

**序贯设计 + 短程筛选**：

```
Round 1（各 40k，~2.8h/个，共 ~14h）
   C2（v12b，add 对照）          ← 必跑，否则 cat 无对照
   A1（adaLN 12 层）
   B1（xattn 12 层）
   A2（adaLN 后 4 层）           ← 零算力成本，性价比高
   
   ↓ 按 gap 斜率 + strict + LPIPS 排序

Round 2（Top-2 跑到 120k，各 ~8.5h）
   + 与 Round-1 胜出的容量档交叉验证
```

**必须遵守的两条**：

1. **xattn 要按 FLOPs 折算，不能按 step 比**。+7.2% FLOPs 意味着同 step 数下它多算了 7.2%
   → 要么多跑 7.2% 的步，要么在报告时把它的 step 轴换算成等效 FLOPs。
2. **至少跑到 100k 才下结论**（80k 下结论已经在 xattn 上出过一次错）。

### 一条设计原则

**注入的三条通路职能不同，不要重复叠加。** doc 60 §6 那条纪律在这里同样适用：
加新注入前先问"这个信号是不是已经从别处给了"。
`callig_spatial` / style token / 12ch 是同一个错误的三种形态 —— 都是往一个
已经提供了该信号的系统里再塞一遍。

---

## 2.3 其他

### 2.3.1 12ch —— 为什么之前不 work

#### ✦ 核心结论：**不是 flow 的锅**

**决定性证据（本轮新查）**：

| 文件 | 事实 |
|---|---|
| `ref/moyi/train_moyun2_RF.sh:11` | `--model test-models-nofeature-12channel` |
| `ref/moyi/moyun/moyun_2.py:737` | 该变体 `depth=24, in_channels=12, hidden_size=1024, num_heads=16`（约 **470M**） |
| `ref/moyi/train_moyun2_RF.py:438` | `--use_12channel` 默认 **1** |
| `ref/moyi/train_moyun2_RF.py:305-320` | `x = cat(image, edge, skeleton)`，三张 RGB 各自 VAE 编码 ×0.18215 |
| 同脚本:16,18 | `--epochs 80000`、`--global-batch-size 256` |

→ **ref 就是"12ch + Rectified Flow + 等权 MSE"，跑 8 万 epoch / batch 256。**
**12ch 与 flow 完全兼容。** 我们之前的失败不能归因于 DDPM→flow。

#### ✦ 那真正的原因是什么（三个前提差异）

| # | 机制 | 证据 |
|---|---|---|
| **1** | **结构信号重复** | ref 所有变体 `use_stroke=False`（**没有结构条件**），结构只能从目标通道进；我们有 g 骨架（char-only 0.3469 → std-skel **0.5680**，+0.22）。→ 12ch 是"同一信号喂两遍"：一遍当条件（无损、每层可见），一遍当目标（要分梯度） |
| **2** | **等权梯度稀释** | img 4ch : aux 8ch = 1:2；aux std 更低更易拟合；实测 aux 吃掉 **~51% final-layer 梯度**（`patch_embed` 上 aux 梯度是 img 的 **2.6×**）。ref 能扛是因为 **470M 参数 / 1.93M 图**；我们 **37M / 28.5k** |
| **3** | **CFG 作用域错误** | `image_channels=None` → 回退 `in_channels=12` → CFG 对**全部 12 通道**引导，aux 分布不同却被同一 scale 放大 → 轨迹跑飞（"墨团"的直接原因）。**现已可显式设 4**（本轮注册了 `--image-channels`） |

**规模差是根因的放大器**：ref 245M–470M × 1.93M 图；我们 37M × 28.5k。
数据差 **68×**，参数差 6–12×。**在这个比例下，"用多任务换表征"的配方不成立。**

#### ✦ flow 相关的次要因素（存在，但不是主因）

- aux 通道（canny/skel 是近二值线稿的 VAE latent）分布与 image 不同：std 更低、更稀疏。
  flow 的线性插值 `x_t = t·x₁ + (1−t)·x₀` 在这个空间里路径更"弯"，
  速度场 `v = x₁ − x₀` 的方差结构与 image 不同。
- 我们没有 per-channel / SNR 加权（DDPM 的 loss 天然带 SNR 加权），
  等权 MSE 下模型优先拟合方差大的方向。
- **但这两点在 ref 的 RF 里同样存在**，所以是**放大器，不是根因**。

#### ✦ 要不要换回 DDPM

**我的判断：不作为主实验。**

- `--diffusion-type` 两者都支持（train.py:1766，默认 ddpm），切换是配置级的
- 但换回去的代价：flow 的直线路径 + 低 NFE 正是单卡 4090 最需要的
- **12ch 的问题是"信号重复 + 梯度稀释 + 规模不匹配"，换扩散类型三条一条都不解决**

只作为 12ch 实验里的**一个归因分支**（见方案 C）。

#### ✦ 如果要让 12ch work：三个方案

**方案 A（推荐）：aux 不作为扩散目标，改为辅助 head**

理由 —— ref 自己就把 aux 扔了：`generate.py:57-58`

```python
samples = samples[:, 0:4, :, :]   # 12 通道只取前 4，8 个 aux 通道直接丢弃
```

**ref 训练了 12 个通道，推理只用前 4 个**。说明 aux 的价值是
"逼主干建立结构感知的表征"，**不是"输出结构图"**。既然如此，
就不该用扩散目标去承载它 —— 那会同时带来梯度稀释和 CFG 作用域两个问题。

设计：
- 主干保持 **4ch** 扩散目标（干净、CFG 作用域天然正确、无梯度稀释）
- 在中间层（建议 layer 8，与 REPA 对齐）挂**轻量 aux decoder**，预测 canny/skel latent
- aux loss 权重 **0.05–0.2**，且可**随训练衰减到 0**（表征 shaping 完成后撤掉）
- 优点：完全避开三个失败机制；**推理零成本**；可随时开关
- 成本：新代码 ~50 行 + 一次 30–50k 验证
- 风险：decoder 容量要小，否则又变成第二个主任务

**方案 B：12ch 修正版（保留 ref 形式，修三处）**

- `image_channels=4`（CFG 只引导图像通道）—— 现已可配
- aux 权重**不等权**：skel 0.05–0.1、canny 0.02–0.05（远低于 img 1.0）
- 只在**前 30% 训练**启用 aux，之后衰减到 0（避免长期抢梯度）
- 风险：修三处仍然解决不了"信号重复"，期望值低

**方案 C：12ch + DDPM 对照（纯归因）**

- 只在**方案 B 的框架下**加一个 `diffusion_type=ddpm` 分支
- 成本：一次 30–50k 短程
- 用途：**只为回答"是不是 flow 的锅"**，不作为生产路线

**建议顺序**：先做 **A**（30–50k 短程）。
若 A 有效 → 12ch 以"辅助 head"的形式复活，且顺带验证了"表征 shaping"这条思路；
若 A 无效 → **基本可以给 12ch 判死刑**，因为 A 是最干净的"把结构信号变成 shaping"的形式，
连它都不行，B/C 只会更差（它们还额外背负梯度稀释和 CFG 问题）。

---

### 2.3.2 drop 比例

#### 现状 vs ref

| | 我们（v12） | ref（`_ful.sh`） |
|---|---|---|
| 总 drop 概率 | **0.10**（`cond_drop_all_prob`） | 0.08+0.08+0.16 = **0.32** |
| 单因子 drop | **0.0**（`cond_drop_one_prob`） | 0.32 |
| 机制 | **互斥 4-way** | **独立 Bernoulli**（`moyun_2.py:235` `torch.rand(...) < dropout_prob`，每个因子各一次） |
| 偏向 | `which_glyph=0.85`（但**用不上**） | callig 是内容的 **2 倍**（0.16 vs 0.08） |

⚠ **ref 内部不一致（读代码要注意）**：`train_moyun2_RF.py:301-303` 硬编码
`"calligrapher_x": 1, "font_x": 1, "charactor_x": 1` → **RF 那份脚本实际上没有 drop**。
"ref 生产 drop 0.32" 来自 DDPM 的 `_ful.sh`。**做 drop 实验时别把两份混为一谈。**

#### 我们当前的实际状态（叠加 0.1 的 bug）

```
cond_drop_one_prob = 0        → 4-way 退化为 2-way（full / uncond）
cond_drop_which_glyph_prob=0.85 → 完全没被用到
+ factorized_cat 不在 guard 元组里 → 连 2-way 都没了，实际 drop = 0
glyph_drop_prob = 0.0         → doc 44 认定 0.25 是 g 通路不激活的根因，已改为 0
```

**→ v12 实际零 dropout，而 eval 用 cfg=0.7。CFG 是在一个未训练的分支上做的。**

#### 建议实验

| # | 改动 | 理由 |
|---|---|---|
| **D0** | **修 0.1 的 guard 元组** | 前置，否则下面全无效 |
| **D1** | `cond_drop_one_prob` 0 → **0.20**；`which_glyph` 0.85 → **0.67** | 让 4-way 真正生效。0.67 对齐 ref 的 callig:content = 0.16:0.24 ≈ 2:1 |
| **D2** | 改为**独立 Bernoulli**（对齐 ref 实现）：callig 0.16 / glyph 0.08 | 我们目前是互斥 4-way，与 ref 语义不同。独立 drop 能产生"多因子同时丢"的样本（虽然 rare） |
| **D3** | `glyph_drop_prob` 0 → 0.05（可选） | doc 44 说 0.25 有害，但那是"g 通路不激活"的语境；现在 g 相对作用已到 **1.68**（当年 2.7%），可以小量重试 |

#### ⚠ 这个实验和第一部分是同一件事的两面

**drop 做对了 → cfg > 1 应该能单调提升（或有明确峰值）；
drop 没做对 → cfg 曲线会是平的或单调下降**（因为 uncond 分支没训好）。

**所以判据是：cfg 响应曲线的形状。** 具体做法：
- 在 D0+D1 修复后重训 40k，重跑 Stage 1 的 cfg 扫描
- **对比修复前后的曲线**：如果修好后峰值从 cfg≤0.7 移到 cfg>1，
  就直接证明了"drop 没做对 → CFG 失效"这条因果链

这是我认为**信息量最大**的一个实验：它同时验证了两件事，
而且如果成立，**当前所有历史 eval 数字（cfg=0.7）都是被低估的**。

---

## 2.4 【补充，本轮最重要】CFG 的"前置/后置"与内容轴缺失

> 这一节是对"cfg 前置还是后置，会不会影响 12ch / xattn"的回答，结论会改变优先级。

### 2.4.1 两个正交的"前置/后置"，答案相反

**(a) 通道维度：必须"后置"（先切 image 通道，再引导）—— 我们已经做对了**

`dit.py:1239`：

```python
eps, rest = model_out[:, :self.image_channels], model_out[:, self.image_channels:]
cond_eps, uncond_eps = torch.split(eps, original_bs, dim=0)
half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
out = torch.cat([half_eps, half_eps], dim=1)     # rest 原样返回
```

"前置"（对全部 `in_channels` 引导）= ref 的 bug（`[:, :3]` 硬编码，24 通道只引导 3 个）
= aux 通道被异分布的同一 scale 放大 = **"墨团"的直接原因**。
**我们比 ref 强，不要回退。12ch 若复活，必须保持后置 + `image_channels=4` 显式设置。**

**(b) 条件维度：问题在这里 —— uncond 分支定义错了**

`dit.py:1231`：

```python
# 标准字形条件 g 始终全给(两半都用真实 g): 字形内容是正条件, CFG 只强化 callig 风格
g2 = torch.cat([g, g], dim=0) if g is not None else None
uncond_callig = torch.full_like(y_callig, self.y_callig_embedder.num_classes)
```

→ **uncond 分支 = (callig=null, g=真实) → 引导方向 = 纯书家风格，内容方向零引导。**

### 2.4.2 ⭐ 推论：所有"只依赖 (x, g)"的通路对 CFG 差分的贡献**精确为 0**

CFG 是 `uncond + w·(cond − uncond)`。同一输入 x、同一 g 走两遍，则下列通路两半输出**完全相同**：

| 通路 | 依赖 | 对 `cond − uncond` 的贡献 |
|---|---|---|
| 输入层 `x += glyph_scale·g_tok` | g | **0** |
| 4 层 `ZeroAdaLNInjection` | g_tok, x | **0** |
| **xattn（Q=x, K/V=g_tok）** | g_tok, x | **0** |
| glyph_vec concat 进 `c` | g | **0** |
| `c` 里的 callig 部分 | callig | **唯一非零** |

**→ CFG 只能通过 adaLN 的 `c` 起作用，且只能放大"书家风格"这一个方向。
我们把字身份的主载体（char-only 0.3469 → std-skel 0.5680，+0.22）整条放在了 CFG 的盲区里。**

### 2.4.3 会不会影响 12ch / xattn

**会影响 xattn，而且是致命的：**

- **xattn 装得越强，CFG 的有效引导越弱。** xattn 输出在两半完全相同，在差分里被精确消掉，
  只能通过后续受 `c` 影响的 adaLN 间接产生差异。
- 这直接解释了 **doc 54 的"xattn strict ≈0"**：不是 xattn 没用，
  **是我们的 CFG 用不上它** —— xattn 是"内容通路"，而 CFG 只引导"风格方向"。
  （这也让 doc 60 §4 的"xattn 不确定"有了机制解释。）
- **也解释了 v12 的 concat**：glyph_vec 进 `c` 本想让内容进入条件，
  但 uncond 半的 g 是真实的 → glyph_vec 也相同 → **同样是 0 贡献**。

**对 12ch：** 通道维度的后置已经正确，不受影响；但若要靠 CFG 拿到 12ch 的收益，
同样受制于"内容轴缺失"。

### 2.4.4 修复：内容轴 CFG（2-axis），代码里已有骨架但用错了因子

`dit.py:1244` 的 `forward_with_2axis_cfg` **已实现** 4 路并行（full / callig / glyph / uncond），
但：

- 它是按**旧的 char 因子**写的：第二轴是 `y_char`，而我们 `use_char_cond=False` → **该轴不存在**
- `g4 = torch.cat([g,g,g,g])` → **g 四路全给**，所以它的"glyph 轴"根本不是 g
- **且从未被 eval 调用过**（`in_mem_eval.py:340` 注释明确走 `forward_with_cfg`）

**要做两件事：**

1. **训练侧**：`glyph_drop_prob` 0 → **0.08–0.10**，让模型真有一个"无内容"分支。
   实现是 `g = g * keep`（`dit.py:1001-1003`），而 g 通路是 `bias=False` 的 Conv
   → `g=0 → g_tok=0` → 干净的"无字形"信号。
   （唯一要核的是 `glyph_vec_proj` 若带 bias 则不严格为零。）
2. **推理侧**：新增 drop-g 分支（改造 2axis 使其以 **g** 而非 `y_char` 为第二轴）：

   | 路 | callig | g |
   |---|---|---|
   | full | 真实 | 真实 |
   | no-callig | null | 真实 |
   | **no-glyph** | 真实 | **0** |
   | uncond | null | **0** |

   `eps = uncond + w_style·(no_glyph − uncond) + w_content·(no_callig − uncond) [+ w_inter·(…)]`

   成本：**推理 NFE ×2**（2N → 4N）。训练侧零额外成本。

**收益**：xattn / adaLN 注入 / glyph_vec / 输入 add **全部第一次进入 CFG 差分**，
CFG 才第一次能放大"写的是这个字"。

**风险与权衡**：doc 44 判 `glyph_drop_prob` 有害（0.25）。
但那是 g 通路不激活的年代（相对作用 2.7%）；现在 g 相对作用 **1.68**（活了），
且我们只要 0.08，远低于 0.25。**这是可检验的权衡，不是猜测。**

**优先级：建议排在所有模型实验之前。** 理由：零训练成本（仅改 drop 概率 + 推理分支），
却可能让 xattn 的价值第一次真正兑现，并解释两件历史怪事。

---

## 2.5 【补充】aux 的定位：之前实现确实不对

**"aux 是辅助"和"我们之前的 12ch 实现"是矛盾的：**

| 维度 | "辅助"应该是什么 | 之前的 12ch 实现 |
|---|---|---|
| 位置 | 主干之外 / 中间层侧挂 | **在扩散目标里**（12 通道一起被 MSE 监督） |
| 权重 | 0.05–0.2 | **等权 1.0**（img:aux 通道 = 4:8 → aux 拿 2/3） |
| 梯度 | 少量 shaping | **吃掉 51% final-layer 梯度**（`patch_embed` 上 aux 梯度是 img 的 2.6×） |
| 推理 | 零成本 | 生成 12 通道再扔掉 8 个 |

**那就是第二个主任务，不是辅助。**

**ref 自己也证明了 aux 是 shaping 不是输出**：`generate.py:57-58`
`samples = samples[:, 0:4, :, :]` —— 训练 12 通道，推理只取前 4，8 个 aux 直接丢弃。

**我们已经有一个"对的辅助"范本：REPA。**
`w_repa=0.03`，作用在 layer 8，对齐 DINO 特征，不在扩散目标里，不抢主梯度，推理零成本，
且历史最优线都带它。→ **aux head 应该照 REPA 的样子做，不是照 12ch 的样子做。**

### ⚠ 但还有一个更深的问题：aux 要预测什么？

我们有 g、ref 没有 → 结构信号**已经从条件侧无损地给过一遍了**。
那么 aux head 还能提供什么 g 没给的？

**答案决定成败**（doc 54 实测）：

| 条件 | strict |
|---|---|
| 标准字形骨架 g（可部署） | 0.5680 |
| **GT 实例骨架（不可部署，归因上限）** | **0.7326**（+0.16） |

→ **我们缺的不是"结构"，是"这个书家写的这个字的具体形态"。**

- aux head 若预测**标准骨架** → 纯重复 g，必败
- aux head 若预测**目标图的实例 canny/skel**（ref 的做法，训练时从 target 派生）→
  提供的是 g 没有的信息，**方向成立**，且与 REPA 同为"表征 shaping"，推理丢弃

**所以方案 A 的规格要写明：aux head 预测实例骨架，不要预测标准骨架。**

### 2.5.1 已实现（本轮）：冻结 probe 版结构 loss

**"之前做过吗？"—— 做过，两版，问题都不在"预测什么"，在"怎么预测"：**

| 版本 | 做法 | 现状 |
|---|---|---|
| (a) **12ch 扩散目标** | GT canny/skel latent 与 image latent 拼 12 通道，**一起当扩散目标** | 在主 train.py 里（等权 → aux 吃 51% 梯度） |
| (b) **frozen probe 结构 loss** | 冻结小网络把 **pred_xstart** 映射成骨架，对 GT 做监督 | 主 train.py **0 次出现 —— 被删了**；模块 `src/train/latent_structure.py` 还在 |

(b) 死在两个地方：
1. 旧代码把 `w_latent_skel/w_latent_canny` 放进 **flow 硬禁用列表**
   （理由"flow 的 velocity target 与 DDPM 辅助项语义不符"——站不住：
   `flow_matching.py:14` 是 `x_t=(1-t)·x0+t·noise`，由 v 反解 x0 完全良定义）
2. `max_timestep` 的 **DDPM 0–1000 口径**混进 flow 的 0–1 阈值
   （`LatentStructureLoss` 的注释已经指出"会被 int() 截成 0 → 门控恒假"）
3. 之后在清理中被整体删除

**本轮实现（新增，非修改现有行为，`w=0` 时完全不生效）：**

| 文件 | 内容 |
|---|---|
| `src/train/latent_structure.py` | 新增 `LatentSkelProbe`（4ch→4ch 骨架 latent）+ `LatentSkelStructureLoss`（冻结 probe + `t<=max_t` 门控 + 通道校验） |
| `src/train/train.py` | 三个新参数 + probe 加载 + 接进总 loss + 日志列 + `pred_xstart` 图门控复用 `w_std_mid` 的既有机制 |
| `tools/train_latent_skel_probe.py` | probe 训练器（从两侧 shards 按 img_id 配对回归，含"常数基线"自检） |

**为什么用冻结 probe 而不是可训练 head**：新增可训练参数 **0**（当前主要矛盾是过拟合，
这是决定性优势）；且可辨识性更好（可训练 head 会去适应 DiT 的糟糕预测而退化成恒等映射）。
梯度仍穿过 pred_xstart 回传主干 —— 这才是"逼主干建立结构感知表征"的本意。

**与 12ch 的对比：**

| | 12ch（旧，已证有害） | 本实现 |
|---|---|---|
| 位置 | **扩散目标里** | pred_xstart 上的侧挂 loss |
| 扩散目标 | 12ch（被污染） | 4ch（干净） |
| CFG 作用域 | 被 12ch 扰乱 | 完全不受影响 |
| 新增可训练参数 | 主干要多学 8ch | **0** |
| 推理成本 | 生成 12ch 再扔 8ch | **0** |
| 梯度占比 | 等权下 aux 吃 51% | 由 `w` 显式控制（建议 0.02–0.1） |

**已验证**：语法 / 三个 config 键确实落位（不是静默丢弃）/ probe 冻结后可训练参数 0 /
`t>max_t` 时 loss 恒 0 / `t<=max_t` 时 loss >0 且**梯度能回传 pred_xstart** /
通道不匹配显式报错（不静默）。

**⚠ 当前还不能跑 —— 缺数据前提：**

v12 的 `skel_latent_shards_dir = shards_std` 且 `skel_as_glyph_cond=True`
→ `batch['skel_latent']` **就是条件 g 本身**。拿它当结构 target 是纯重复，必败且静默
（loss 会正常下降，因为预测 g 太容易）。故**已加硬拦截**：`w_latent_skel>0` 且
`skel_as_glyph_cond=True` 时直接拒绝启动。

**要启用需三步：**
1. 为**实例骨架**单独编码 shards（目标图派生的 canny/skel，不是标准字形）
2. `latent_dataset.py` 增加独立 key（如 `inst_skel`）暴露它，与 g 解耦
3. train.py 结构 loss 改读该 key（当前读 `batch['skel_latent']`）

---

## 2.1.1 【结果】容量阶梯跑完，结论：**S/2 是甜点，两个方向再缩都亏**

> 2026-09-17。`v12_d8`（XS/2）已跑完 100k；`v12_w320` 在 42.9k。

### ⚠ 先说两个会让结论出错的方法论问题

**(a) 三个 run 的 batch 不同 → "同 step" ≠ "同数据量"**

| run | 模型 | params | global_batch | step/s | **samples/s** |
|---|---|---|---|---|---|
| v12 | S/2 d12 h384 | 36.4M | **360** | 5.53 | 1,991 |
| v12_d8 | XS/2 d8 h384 | 25.8M | **576** | 3.72 | 2,143 |
| v12_w320 | S320/2 d12 h320 | 25.8M | **440** | 3.81 | 1,676 |

d8 用 576 的 batch，跑到 100k 步 = **57.6M 样本**，是 S/2 的 **1.6 倍**。
按"同 step"比会让 d8 白占 60% 的数据便宜。**必须按 samples 对齐。**

**(b) gap 小可能是"两边都差"，不是"泛化好"**

d8 的 gap 只有 +0.05（看着比 S/2 的 +0.16 漂亮得多），但它的 **seen 也塌了**（0.58 vs 0.72）。
**gap 必须配 strict 的绝对水平一起看** —— 这是对我之前"主判据是 gap 斜率"那条纪律的重要修正。

### 按同样本量对齐（21.6M samples，配对检验）

| 对比 | strict d | t | seen d | t |
|---|---|---|---|---|
| S/2@60k(21.6M) vs **d8@40k(23.0M)** | **−0.0111** | **−2.44 显著** | −0.0201 | −1.60 |
| S/2@60k(21.6M) vs **d8@35k(20.2M)** | **−0.0117** | **−2.98 显著** | **−0.0336** | **−2.47 显著** |
| S/2@50k(18.0M) vs **w320@40k(17.6M)** | **−0.0142** | **−4.12 显著** | −0.0148 | −1.51 |

**→ 两个方向缩容量都在同样本量下显著更差**（d8 还多看了 6% 数据仍然落后）。

### 完整对照（按 samples 对齐到 36M）

| run | strict | seen | gap | LPIPS |
|---|---|---|---|---|
| **v12 S/2**（37M） | **0.5599** | 0.7214 | +0.1615 | **0.3636** |
| v11 M/2（47M） | 0.5555 | 0.7127 | +0.1572 | — |
| v12_d8 XS/2（27M）@34.6M | 0.5271 | 0.5797 | +0.0526 | 0.3877 |

### 结论

1. **XS/2（d8 h384, 27M）不推荐**：同样本量 strict **−0.011~−0.012（显著）**，
   LPIPS 也更差（0.388 vs 0.364）。**它 lower gap 是"seen 也塌"造成的假象，不是泛化好。**
   它的 strict 平台约在 **0.53**。
2. **S320/2（d12 h320, 27M）不推荐，双输**：同样本量 strict **−0.0142（t=−4.12 显著）**，
   而且**吞吐反而更低**（1,676 < 1,991 samples/s）。
   → **收窄宽度既损质量又损速度**（窄 GEMM → MFU 下降，正是 doc 62 预判的张力）。
3. **S/2（d12 h384, 37M）是甜点**：与 M/2 打平（strict +0.004，噪声内），
   但吞吐 **+33%**（1,991 vs M/2 的 1,494 samples/s）。
4. **容量下界在 S/2 附近** —— 再往下（无论缩深度还是缩宽度）都是亏的。

### ⚠ 自我更正：我之前"S/2 减少过拟合"的结论**被推翻**

当时（60k）我看到 40–60k 窗口 S/2 的 gap 斜率是 M/2 的 **0.48×**，据此说"gap 增速减半"。
把窗口拆开后：

| gap 斜率 | 40–60k | 60–80k | 80–100k | **40–100k 全窗口** |
|---|---|---|---|---|
| v11 M/2 | +0.00171 | +0.00407 | +0.00325 | **+0.00311** |
| v12 S/2 | +0.00083 | **+0.00479** | +0.00297 | **+0.00354** |

**60–80k 时 S/2 反而是 M/2 的 1.18×，全窗口 40–100k 也是 S/2 更快（1.14×）。**
gap@100k：S/2 +0.1615 vs M/2 +0.1572 —— **基本持平**。

→ **那是窗口假象。我犯了"过早下结论"的错** —— 正是我自己写进纪律里的那条（doc 60 §4、64 §3）。

**换 S/2 的正确理由是：同质量下快 33%。不是"减少过拟合"。**

---

# 3. 判据纪律（沿用 64，加两条）

1. **主判据是 gap 斜率，不是单点 strict。**
   n=50 的配对 SE ≈ 0.005 → MDE ≈ 0.014，而我们关心的效应只有 0.004
   → **单点 strict 比较在这个样本量下永远检不出**。
   gap 斜率跨 checkpoint 聚合，n=9 就有 t=+11。
2. **看四元组 `strict / seen / gap / lpips`**，缺一不可（ssim 单独会撒谎，doc 59）。
3. **每个实验同预算（步数）、同数据、跑到平台**（≥100k）。80k 下结论已在 xattn 上出过错。
4. **加组件前问"这个信号是不是已经从别处给了"**；
   **减组件前问"它有没有被单独消融过"**（`glyph_embedder_depth=2` 就是空白，49 个 config 全 2）。
5. **新增：改任何条件相关代码后，先确认 CFG 仍能工作**
   （本轮 0.1 的 bug 就是加 `factorized_cat` 时漏改 guard 元组造成的，
   而且它**不报错、不崩溃，只是静默让 dropout 归零**）。
   检查方法：训练日志里确认 `callig_drop` 非空的样本比例 ≈ `cond_drop_all_prob`。
6. **新增：比较实验必须按"已见样本量"对齐，不是按 step。**
   `v12_d8` 用 576 的 batch 而 `v12` 用 360 —— 跑到 100k 步，d8 多看了 **60%** 的数据。
   按 step 比会让大 batch 的实验白占便宜。**先查 global_batch_size 再比。**
7. **新增：gap 必须配 strict 的绝对水平一起看。**
   gap 小可能是"两边都差"而非"泛化好" —— d8 的 gap +0.05 看着漂亮，
   实际是 seen 塌到 0.58（S/2 是 0.72）。**单看 gap 会把"更弱"误读成"更不过拟合"。**
8. **新增：评估任何"内容侧"组件前，先问它对 CFG 差分有没有贡献。**
   在 `forward_with_cfg` 把 g 全给两半的前提下（§2.4.2），
   一切只依赖 (x, g) 的通路（输入 add / adaLN 注入 / xattn / glyph_vec）
   对 `cond − uncond` 的贡献**精确为 0**，在 cfg 引导的评测口径下**天然被低估**。
   要公平评价它们，必须先上内容轴 CFG（§2.4.4），否则测的是"CFG 盲区里的性能"。

---

# 4. 总路线图与预算

```
【P0 修复】(30 min)  0.1 cat drop guard  ★必做★
                     0.2 注册 repa_layers / gpu_eval_cfg
                     0.3 expandable_segments
        │
        ▼
【第一部分 · 推理】(~2h，零训练)
        Stage1 cfg 曲线（先在 v11 上做，v12 的 uncond 未训练不可信）
        Stage2 NFE 公平比较   Stage3 seed/样本量   Stage4 自条件
        │  判读：cfg 峰值在哪 → 决定 drop 实验的紧迫性
        ▼
【第二部分 Round 1】各 40k 短程筛选
        2.1 容量：v13 XS/2 ∥ v14 S320/2                    (~4h)
        2.2 注入：C2(v12b) / A1(adaLN12) / B1(xattn12) / A2(后4层)  (~11h)
        2.3 其他：D1(drop 0.2) / 方案 A(aux head)           (~6h)
        │  按 gap 斜率 + strict + LPIPS 排序
        ▼
【第二部分 Round 2】Top-2 × 120k                          (~17h)
        + 容量胜出档 × 注入胜出档 交叉验证
        ▼
【条件性】v15 XS6/2（仅当 v13/v14 不劣化）
        方案 B/C（仅当方案 A 有效）
        数据：增强 83,909 / 书家 36→90（需你决定）
```

| 阶段 | 预算 |
|---|---|
| P0 修复 | 0.5h |
| 第一部分 | **2h（零训练）** |
| 第二部分 Round 1 | ~21h |
| 第二部分 Round 2 | ~17h |
| **合计（不含条件性项）** | **≈ 40 GPU 小时**（当前 3.92 step/s；修好性能回归后 ≈ 30h） |

**推荐先做且只做**：P0 修复 → 第一部分 Stage 1（cfg 曲线，40 min）。
这两步零训练成本，但可能直接推翻"当前 eval 数字"的解释
（如果证实 cfg=0.7 一直在稀释条件，那么 strict 0.5366 是被低估的，
而所有"模型不行"的判断都要重估）。
