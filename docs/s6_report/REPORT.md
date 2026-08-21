# S6 Top6 实验总结报告:canny+skel 结构损失是副作用吗?

**日期**: 2026-08-21 | **模型**: DiT-2Cond-S/2 | **数据**: MCCD top6 子集 (`5script/train_top6.csv`)
**评测**: eval100_top6.csv 自由采样 DDIM (cfg=4.0, steps=50) → MSE / SSIM

---

## 1. 结论(TL;DR)

1. **同数据、同模型、同评测集的公平对比下,全程等权重的 pixel 结构损失(canny 0.5 + skel 1.0)显著拖累生成质量**:
   同为 120000 步,diff-only **MSE 0.475 / SSIM 0.700**,struct **MSE 0.788 / SSIM 0.403**。
2. **diffonly(纯扩散损失)是当前最佳模型**:195000 步收敛到 **MSE 0.432 / SSIM 0.732**,样本目检为干净锐利的书法字。
3. struct 运行伴随**彩色噪声伪影**与 **x0 隐空间漂移**(X0Lat raw ≈ 36-39,diffonly 仅 1~2.5),提示 pixel 空间结构损失把 x0 预测推离了 VAE 流形。
4. 已启动 **S7 爬升实验**验证补救方案:从 diffonly@195000 出发,struct 权重 20000 步线性爬升,batch8 全体 vs batch32 subset8 两实验串行。

---

## 2. 实验设置

| 项 | diffonly | struct_fp32 (首跑) | struct_fp32 (续训) |
|---|---|---|---|
| results_dir | `s6_top6_diffonly` | `s6_top6_struct_fp32/20260820-002342` | `s6_top6_struct_fp32/20260821-073004` |
| 结构损失 | 无 (use_canny/skel=false) | canny×0.5 + skel×1.0 | 同左 |
| struct_subset | — | 8 | 8 (batch48 时为子集) |
| 初始化 | 从零 | 从零 | 续训自首跑 0090000.pt |
| 训练区间 | 20000 → 195000 | 0 → 90000 | 90000 → ~129000(被叫停) |
| 速度 | ~4.0 step/s | — | 1.60 step/s (fp32 VAE decode 开销) |
| 早停 | ssim patience 6 | — | — |

两个实验**都从零开始**、都用 `train_top6.csv` 训练、都用 `eval100_top6.csv` 评测——对比公平,无预训练混淆。
(用户确认:diffonly 之前的 steps 也是纯 diff-only 训练。)

---

## 3. 定量曲线

### 3.1 diffonly(完整 35 个 eval 点,单调改善)

| step | MSE | SSIM | | step | MSE | SSIM |
|---|---|---|---|---|---|---|
| 25000 | 0.803 | 0.525 | | 115000 | 0.480 | 0.695 |
| 40000 | 0.688 | 0.571 | | 130000 | 0.460 | 0.709 |
| 55000 | 0.604 | 0.614 | | 145000 | 0.449 | 0.719 |
| 70000 | 0.555 | 0.645 | | 160000 | 0.437 | 0.726 |
| 85000 | 0.526 | 0.665 | | 175000 | 0.434 | 0.729 |
| 100000 | 0.510 | 0.678 | | 190000 | 0.434 | 0.731 |
| 105000 | 0.496 | 0.686 | | **195000** | **0.432** | **0.732** |

- 25000→105000:SSIM 快速爬升 (+0.16)
- 130000→195000:进入平台期,末段每 5000 步 SSIM 增益 <0.002 → **已收敛**

### 3.2 struct_fp32 续训(被叫停前 7 个 eval 点,仍在爬升)

| step | MSE | SSIM |
|---|---|---|
| 95000 | 0.893 | 0.340 |
| 100000 | 0.873 | 0.348 |
| 105000 | 0.849 | 0.361 |
| 110000 | 0.824 | 0.379 |
| 115000 | 0.804 | 0.391 |
| 120000 | 0.788 | 0.403 |
| 125000 | 0.783 | 0.404 |

### 3.3 正面对比(同 step)

| step | diffonly MSE/SSIM | struct MSE/SSIM | 差距 |
|---|---|---|---|
| 100000 | 0.510 / 0.678 | 0.873 / 0.348 | SSIM 差 2 倍 |
| 120000 | 0.475 / 0.700 | 0.788 / 0.403 | SSIM 差 1.7 倍 |
| 125000 | 0.466 / 0.705 | 0.783 / 0.404 | SSIM 差 1.7 倍 |

struct 曲线健康上升(每 5000 步 SSIM +0.01~0.02),但**按此速度追平 diffonly 需要额外 10 万步以上**,且不排除在更低水平进入平台期。

---

## 4. 视觉评价

### 4.1 全程海报(每行 = 一个 ckpt,样本按 img|canny|skel 三联格展示)

- **diffonly 35 行海报**: `poster_diffonly.png`
  ![diffonly poster](poster_diffonly.png)
  - 25000-50000:字形成形但笔画细弱
  - 60000-120000:逐步干净,风格(楷/隶)稳定
  - 125000-195000:锐利干净的书法字,与 GT 几乎无异

- **struct_fp32 续训 7 行海报**: `poster_struct_resume.png`
  ![struct poster](poster_struct_resume.png)
  - show5 的"倪"相对干净,但"陳"发糊带重影
  - **seen5 的"之/性"出现严重彩色噪声**(绿/黄/红斑块),结构仅勉强可辨

### 4.2 同字符直接对比("性", seen 样本)

| diffonly @195000 | struct @120000 | GT |
|---|---|---|
| ![diffonly](imgs/comparison/diffonly_s195000_seen2.png) | ![struct](imgs/comparison/struct_s120000_seen2.png) | ![gt](imgs/comparison/gt_xing.png) |

diffonly 笔画锐利、结构完整;struct 被彩色噪声淹没,几乎不可读。

### 4.3 关键视觉洞察

1. **伪影集中在条件强的样本**(seen 字符):模型对训练对越"自信",x0 预测漂移越远,VAE 解码越失控。
2. **canny/skel 列对比**:struct 生成的边缘图碎片化、有大量孤立噪点;diffonly 的边缘图与 GT 笔画一一对应。
3. **X0Lat 佐证**:struct 日志 X0Lat raw ≈ 31-39,diffonly ≈ 1-2.5 —— 相差 30 倍,x0 隐变量明显偏离 VAE 编码分布(彩色噪声正是 VAE 解码离流形输入的典型表现)。

---

## 5. 为什么结构损失是副作用?(机理分析)

1. **梯度竞争**:canny/skel 损失直接作用于 VAE 解码后的像素,其梯度经 decoder 回传到 x0 预测;结构监督是低频信号,与扩散损失的高频保真目标在共享主干上冲突。
2. **流形约束缺失**:扩散损失在 latent 空间约束预测贴近训练分布;pixel 结构损失只约束边缘/骨架,对颜色/纹理无约束 → 模型找到"边缘对了但颜色烂"的捷径。
3. **有效监督稀释**:struct_subset=8 时每步只有 8 个样本贡献结构梯度,而 batch48 的 diff 梯度被 48 个样本平均——但 struct 实验的退化在 batch48/subset8 下依旧发生,说明不是简单的信噪比问题。
4. **速度代价**:fp32 VAE decode 使训练慢 2.5 倍(1.6 vs 4.0 step/s),同等墙钟时间内 struct 获得的学习量进一步落后。

**但这不否定结构监督本身**:s5 阶段 B 模型实验(canny0.5+skel1.0,从 115000 续训)曾把 SSIM 从 0.391 提升到 0.433——说明**在强基线上温和引入**结构信号可能有益,问题出在"从零+全程等权重"的训练方式。

---

## 6. 下一步:S7 爬升实验(已在跑)

| 项 | 实验A: b8all | 实验B: b32sub8 |
|---|---|---|
| 初始化 | diffonly @195000 | 同左 |
| struct 权重 | 0 → canny0.5/skel1.0,20000 步线性爬升 | 同左 |
| batch / struct覆盖 | 8 / 全体(8) | 32 / 子集(8) |
| LR | fresh-scheduler,余弦 80000 步 | 同左 |
| 收敛判据 | early-stop: ssim patience 6 (每 5000 步 eval) | 同左 |
| 串行 | A 完成后自动启动 B(tmux `s7` 会话,脱离 ssh) | |

**判读标准**:若 A/B 在爬升结束后 SSIM 超过 0.732(diffonly 基线)且无彩色噪声 → 结构监督有增量价值;若回落或伪影复现 → 确认该结构损失在此管线中无正向作用,转向 latent 空间结构损失或条件注入方案。

---

## 8. 大测试集额外评测(本地 GPU)

**方法**:本地 RTX 4070 Laptop,两 ckpt 在完全相同的条件上自由采样(DDIM 50 步,cfg=4.0,seed=0)。构建了两个互补集合:

- **eval500(拟合集参考)**:从 top6 训练池分层抽样 493 张——经核查 **100% 是训练见过的(书家,字)对乃至原图**,只衡量拟合/记忆能力;
- **eval_unseen255(真 held-out)**:从 `5script/test.csv` 筛 top6 书家、**(书家,字)对完全未在训练出现**的样本,楷 161 + 隶 94 共 **255 张**,GT 用与训练一致的远程 `final_images/` 权威版本。

### 8.1 三层指标总表

| 评测集 | 性质 | diffonly @195000 | struct @125000 | SSIM 差距 |
|---|---|---|---|---|
| eval100 | 72% 见过对 | MSE 0.432 / SSIM 0.732 | MSE 0.783 / SSIM 0.404 | 1.8× |
| eval500 (493) | **100% 见过(拟合)** | MSE 0.593 / SSIM 0.654 | MSE 0.820 / SSIM 0.394 | 1.66× |
| eval_unseen (255) | **100% 未见过(held-out)** | **MSE 0.856 / SSIM 0.508** | **MSE 0.944 / SSIM 0.364** | 1.4× |

**关键读数**:
1. **结论在真 held-out 上成立**:struct 的 SSIM 只有 diffonly 的 ~72%,与拟合集一致——结构损失副作用不是"记忆差异"造成的假象;
2. **diffonly 存在明显但可控的泛化鸿沟**:SSIM 0.654(拟合)→ 0.508(unseen);struct 几乎无差距(0.394→0.364),因为它两头都差;
3. unseen 集部分 GT 本身质量较差(勵/剮/暇等带框、噪底),两边指标都被压低,但不影响相对结论。

### 8.2 同条件三方对比(同 seed 同字符)

**held-out unseen(255 样本中的前 16 个)**:
![compare_unseen](large_eval/compare_unseen.png)

**拟合集 eval500(前 16 个)**:
![compare](large_eval/compare_diffonly_vs_struct.png)

**目检结论**:
1. diffonly 在**从未见过的字**上风格迁移良好(寢/儀/鑫/單/夢等结构正确、笔意到位),复杂生僻字(勵/逋/聪)较弱;
2. struct 在 seen 和 unseen 上**16/16 全部**彩噪退化——系统性 x0 流形漂移,与条件是否见过无关;
3. 结构上 struct 字形骨架大致正确(结构损失确实"起效"),坏的是像素保真。

---

## 9. 产物清单

| 文件 | 说明 |
|---|---|
| `poster_diffonly.png` | diffonly 全程 35 ckpt 海报 |
| `poster_struct_resume.png` | struct 续训 7 ckpt 海报 |
| `imgs/diffonly_{eval,seen}/step*/` | 原始样本图(35 步 × show/seen) |
| `imgs/struct_resume_{eval,seen}/step*/` | struct 样本图(7 步) |
| `imgs/comparison/` | 同字符直接对比图 |
| `large_eval/compare_diffonly_vs_struct.png` | 拟合集(eval500)三方对比图 |
| `large_eval/compare_unseen.png` | held-out(unseen255)三方对比图 |
| `large_eval/<tag>/grid.png`, `latest/`, `metrics.json` | 本地 GPU 评测产物(4 组:2 ckpt × seen/unseen) |
| `jsons/diffonly_tmp/`, `jsons/struct_resume_tmp/` | eval_auto_*.json 原始指标 |
| `csv/`, `configs/` | 评测集与 resolved_config |

**代码提交记录**(本轮):
- `de4da75` feat(src): train.py 对齐远程运行版(pixel 结构损失 + 早停)
- `dffac3a` feat(s6): top6 实验配置与串行启动脚本
- `ecf3d5e` chore(ops): 运维/环境脚本
- `fea2d96` feat(exp): 历史实验配置
- `8f0d425` fix(tools): pull_log/make_eval_poster 改进
- `73aa874` feat(src): struct 权重爬升 + fresh-scheduler
- `b87d66a` feat(s7): 爬升双实验配置 + 串行启动
- `7cd67f8` fix(s7): 启动脚本用 /opt/conda/bin/python
- (待提交) make_eval_poster 自动检测每侧样本数(n=2 兼容)
