# 两阶段方案归档与结论 (2026-10-03)

## 一句话结论

**两阶段（先用 SkelNet 预测骨架 + 再渲染）判定失败，全部实验已归档到
`_archive/20261003_twostage/`（18 个目录，`exp/` 已清空）。** 失败原因不是
"没训够/参数不够/注入不强"，而是**监督信号与判据两条链都有问题**，详见下面
"必须记住的 5 条"。

---

## 归档内容

`_archive/20261003_twostage/`：

| 阶段 | 目录 |
|---|---|
| 早期 backbone / 拼接矩阵 | `v26_gtskel`, `v30_stitch_matrix`, `v31_stage1_skel`, `v32_stage2_img` |
| 加固/混合课程 | `v34_stage2_mix25`, `v39_render_mix50`, `v41_render_mix90`, `v42_render_curriculum`, `v43_render_20to90_curriculum` |
| 联合（union） | `v35_union`, `v36_union_w7`, `v37_union_20g`, `v40_joint_render_frozen` |
| SkelNet（骨架生成器） | `v36_stage1_skel_w7`, `v37_skelnet_sp`, `v38_skelnet_s2_pure`, `v44_skelnet_concat`, `v45_skelnet_struct` |

`assets/results/` 里是**两阶段之前**的老实验（v11~v17 时代），未归档。

---

## 必须记住的 5 条（都踩过、都有实测证据）

### 1. `use_checkpoint=True` 是 `DiT_2Cond` 的**默认值**
`src/model/dit.py:889`。不显式关掉就会被静默开启 —— 之前 "batch 1440 / 20G" 的
吞吐全是靠它换来的。关掉后本卡（24G）上限约 **batch 128**、~600 samples/s
（`sps` 是 steps/s，曾被误读成 s/step）。

### 2. 条件必须传**真实 id**：`y_char=zeros` 是错的
stage1/v33 是**带字条件**的模型。联合训练器里写死 `y_char=zeros_like(y)`，
等于整轮都在"字 id = 0"下采样 → 条件与目标不符 → 生成器漂向无信息解。
(`src/model/joint_skel2img.py` 已修，`gen_sample(..., y_char=y_char)`。)

### 3. 评测口径里藏着**泄露**：`eval_blend_alpha`
评测 ckpt 的 args 里存着 `eval_blend_alpha = 0.5`，harness 的 "predskel" 一路
实际喂的是 `0.5×生成骨架 + 0.5×标准骨架` —— 而标准骨架源自 GT 字形。
所以 "predskel 拿到 0.6534" **是泄露出来的数**，不是生成器的能力。
`tools/run_skel_calibration.py` 已强制 pred 口径 `eval_blend_alpha=1.0`。

### 4. **SSIM 会骗人，必须用 IoU（+LPIPS）并同时打基线**
实测（v45，同一 ckpt）：

| 指标 | 生成 | "什么都不做"基线 | 判读 |
|---|---|---|---|
| 骨架 SSIM | 0.6484 | 0.6410 | ✓ |
| 骨架 IoU | **0.0870** | **0.1044** | **✗ 不如不做** |

SSIM 背景主导（整幅大部分是白），对"糊但位置偏"很宽容。**任何 eval 都必须
同时输出 IoU / LPIPS 与"直接用标准字"的基线**，否则会得出反向结论。

### 5. latent L2 与骨架结构**不同源**
- `L_flow` 平台在 0.24~0.27，同时骨架 IoU/SSIM 卡在门槛下；
- 原因：**VAE 解码器是收缩的** —— latent 层面的误差在解码后表现为
  "粗布局还在、细笔画没了"，L2 对这种误差不敏感；
- 结构监督（解码后空间）才有效：`Dice（主）+ 投影剖面（辅）`，latent MSE 降权
  保留（0.3）只负责把 latent 留在 VAE 流形上。

---

## CPU 探针结论（换监督的依据）

对真迹墨迹施加**已知严重度**的破坏（平移 1/2/4/8px、加粗/变细、挖块、换成标准字、
换成别的字）后，各损失与 (1−IoU) 的 Spearman 相关：

| 损失 | ρ vs (1−IoU) | 1px 平移增量 | 缺点 |
|---|---|---|---|
| Dice | **+1.000** | +0.151 | 坏端饱和（0.80/0.84 挤在一起 → 梯度小）|
| **投影剖面 proj** | **+0.959** | +0.091 | 只量粗布局，分辨力偏粗 |
| 距离场 dt | +0.924 | +0.002 | **只罚多出的墨，不罚缺失的笔画**（变细/挖洞 = 0）|
| SWD | **+0.732** | +0.101（基数 2.0，仅 5%）| **小误差非单调**（变细反而变小）；对墨量不敏感 |

结论：**SWD 不推荐**；用 `proj（主/辅视权重）+ Dice（主）+ dt（辅）`。
参考阈值：`proj(std, gt) = 0.5467`、`IoU(std, gt) = 0.1044`（严格像素口径）。

另：条件不可约方差探针（CPU）—— 同一 (字, 书家) 组内目标散度占总方差
**11.36%**，31% 的组有多条样本 → **多值性存在但不是主因**（89% 的信息是可解释的）；
且"投影剖面/白化"都不能降低它。

---

## skelnet 为什么"看着收敛却永远不过线"

| 配置 | 步数 | 最好 骨架 SSIM | 备注 |
|---|---|---|---|
| v38（add 注入, batch1440+ckpt）| 2000 | 0.6101 | 之后退化 |
| v44（**concat** 注入, batch192, 无 ckpt）| 4000 | 0.6282 | 同步数比 v38 好，样本效率 7.5× |
| v44 续训 | 10000 | 0.585~0.628 震荡 | **真收敛，无上升趋势** |
| v45（Dice 结构监督）| 1000 | 0.6484 | 过 SSIM 基线，但 **IoU 0.0870 < 0.1044** |

→ 参数（34M）、训练量（150+ epochs）、注入方式（add vs concat 同步数逐位相同）
**都不是瓶颈**；瓶颈是"用单条真迹骨架 latent 做 L2 目标"这件事本身。

---

## 下一步（已定）

1. **单阶段渲染**：条件只用两种骨架 —— `std`（标准字，推理时的真实条件）与
   `gt`（真迹骨架，起点多数=给模型看答案）。**排除 SkelNet**。
2. **课程**：20% std / 80% gt → 40 → 60 → 80 → 90 → **100% std**
   （`tools/train_render_20to100_curriculum_20g.py`，已在 v43 基础上加 Stage 6）。
3. **判据**：每次 eval 输出 `SSIM / IoU / LPIPS` 三口径 + **"直接用标准字"基线**
   + ✓/✗；ckpt 选取改用 **IoU**（原来存 `best_ssim.pt` 是在错的指标上选点）。
4. **容量问题**：先量再说 —— 对比训练 loss 与 200-eval 的 gap。gap 大（过拟合）
   → 减小模型/加强 `deform-prob`/`deform-scale`；gap 小且训练 loss 仍高 → 才考虑加大。
5. **重评历史最好单阶段**：老的 `assets/results/v11~v17` 里，
   `v11_pretrain_M432_adaln4_fame_kxl_tj_px60@115000 ssim_mean = 0.7416`、
   `@105000 = 0.7247`、`v17_s2_s2z_baseline@50000 = 0.5992`、`v13_styletok@165000 = 0.5279`
   —— 但这些都在 **`seen` 小集（n=10/20）** 上，与两阶段用的 **200 样本 strict**
   口径不可比。**要用同一口径重评后才能下"哪个最好、两阶段是否是倒退"的结论。**

---

## 本轮踩坑清单（工程向）

- `pkill -f <名字>` 会匹配到**自己这条命令**（命令里含同名文件路径）→ 自杀；
  `tmux kill-server` 会**杀掉所有会话**（曾误杀训练）。
- 评测脚本的 poster 会把 256px 骨架缩到 64px 网格并 LANCZOS → 细线被平均成白图，
  造成"输出全是白的"错觉（已加 `POSTER_CELL/POSTER_NMAX` 覆盖）。
- `_lpips_per_sample` 要 **NHWC、值域 [0,1]**（内部自己 ×2−1），传 CHW + 已归一化
  会得到 nan。
- 手工 `load_state_dict` 拿不到 `y_callig_embedder.null_embed`（`freeze_callig_table`
  后处理才会产生）→ 一律走 `src/eval/model_io.load_model_from_ckpt`。
- 训练占卡时**不要并发跑评测**（实测把训练挤 OOM）。
- 铁律：**不允许 `expandable_segments`**；训练/评测都不开 gradient checkpointing。
