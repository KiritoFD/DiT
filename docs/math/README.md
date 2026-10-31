# 书法条件生成的数学框架：内容、Writer、随机写法与 VAE


## 数学专题导航（证据修订版）

建议先读 [理论修订账本](00_THEORY_REVISION_LEDGER.md)，再根据需要阅读推导。每篇都区分 **数学恒等式 / 模型机制假说 / 已有测量 / 可以推翻的实验**。

| 文档 | 关键推导或待验证命题 |
|---|---|
| [00 · 证据反例与理论修正](00_THEORY_REVISION_LEDGER.md) | 纠正“0.65上界”“Writer仅7维”“t=1学完整分布”“S5唯一最优”等过强判断 |
| [01 · 条件Flow Matching与时间](01_CONDITIONAL_FLOW_AND_TIME.md) | Bayes风险差恒等式；`t=0`与`t=1`极限；中t的score/Writer差；正确SDE |
| [02 · Writer表示与可辨识性](02_WRITER_INFORMATION_AND_IDENTIFIABILITY.md) | DINO能量分解、跨字Writer探针、来源混杂、Writer-swap方向/幅度/内容 |
| [03 · VAE、Decoder与辅助目标](03_LATENT_DECODER_AND_OBJECTIVES.md) | Decoder Jacobian；12ch梯度竞争；SkelNet级联误差与拓扑非连续 |
| [04 · 迁移曝光与评价统计](04_TRANSFER_EXPOSURE_AND_STATISTICS.md) | P75有效Writer曝光19.125k；等目标曝光vs等总算力；字符簇bootstrap |
| [05 · 数学预测与可证伪实验](05_PREDICTIONS_AND_FALSIFICATION.md) | 12项明确的假设→测量→失败即修正；4090已有任务与追加试验的先后 |

**本阶段最重要的严格新推导**：在本项目独立高斯线性桥上，`v_w^*(z,1)=z-\mathbb E[Z_0\mid G,S,W=w]`，因此**单一纯噪声端的Writer Pair只监督条件均值**；中间时间 `\Delta v_{a,b}=-\tfrac{t}{1-t}\Delta s_{a,b}`，可能承载Writer分布形状变化。真实数据是否属于“协方差主导”，并**未被DINO QDA实验支持**；目前只是数学上的允许机制，必须用同位置t轴风险和同噪声Writer-swap验证。

> **下面保留原数学总览稿作为历史上下文**。其中任何与细化章节或 [00修订账本](00_THEORY_REVISION_LEDGER.md) 冲突的普适推论都应按新推导及原始数据降级，而不是把总结稿当定理。

> **研究状态**：2026-10-31。本文分清 **数学恒等式**、**对模型的机制假说**、**本项目的数值证据**。与 [阶段实验结论](../1031/README.md) 配套；实现依据为 [FlowMatching](../../src/loss/flow_matching.py)、[Writer 诊断](../../ANALYSIS/style_structure_diagnosis_20261025.md)、[S/J 配对报告](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)、[E-G 检验](../1030/eval_style_v2/FINAL_REPORT.md)。不把任何解释当作已经证明的结构定理。

## 0. 总论：我们要学的不是“唯一正确答案”，而是一个受多重条件约束的书写分布

令：

- `C`：目标 Unicode 字符身份；`G`：由字体/字形渲染器得到的标准字形空间条件（如 std-soft 64×64）；`S`：楷/行/隶等书体。
- `W`：书家 ID；`A`：同一书家对某字选择的字形变体、草化/省笔和结字方案（latent allograph）；`R`：一次具体运笔的随机实现。
- `D`：作品来源、扫描条件、拓片材质及采集过程，这些变量可能同时与 `W` 相关，是潜在混杂项。
- `X`：真实单字 RGB 图；`Z_0`：编码器得到的图像 latent；`\epsilon`：用于流匹配的独立高斯噪声。

真实生成过程可概念化为：

\[
P_{\mathrm{real}}(x\mid c,g,s,w,d)
=\int P(x\mid g,s,w,a,r,d)\,P(a,r\mid c,g,s,w,d)\,\mathrm da\,\mathrm dr.
\]

`G` 由 `C` 与渲染规范产生，但并不自动包含书家习惯、异体字、草化选择或作品来源。**同一个字、同一个书家也可能有多个合法样式**；单 GT 图像只是其分布中的一个观测。

这是一个**部分可辨识的问题**：若每位书家只来自某一种纸张/碑帖或扫描器，单靠 `(X,W)` 不能分离“真实书家运笔”与“采集来源”的因果贡献。可在相同书体、不同字符、多个独立作品来源下验证 Writer，但不能凭一个网络的 AUC 宣告去除了所有混杂。

### 任务约束并非“内容、风格完全正交”

希望实现：

1. **内容忠实**：生成 `X_\theta` 能被认作目标 `C`，允许书法中的合法结构变化。
2. **Writer 忠实**：固定 `C,G,S` 时改变 `W`，生成差异应与同书家真实作品之间的条件差异一致。
3. **随机多样性**：固定 `C,G,S,W` 时不同噪声可以生成多个合法样本，而不是一张均值墨团。
4. **跨字泛化**：即使 `C` 未在该 Writer 的训练样本中出现，仍能沿着 `G` 转移共同的构字能力。
5. **视觉细节**：Decoder 应能稳定表达飞白、笔锋、墨量和细线连通。

**不要求** `I(X;C\mid G)=0` 或 `I(X;W\mid G)=0`：`G` 未必决定 Unicode 的全部异体选择，Writer 的合法形变也可能影响字符形态。真正需要的是**干预 Writer 时 OCR 身份保持、Writer 特征按照真迹方向变化**，而不是从像素中硬剥离 Style 的全部内容信息。

## 1. FM 主目标：从噪声到真实书法的条件速度场

按照现行 [`src/loss/flow_matching.py`](../../src/loss/flow_matching.py) 的实际时间方向：

\[
Z_t=(1-t)Z_0+t\epsilon,\qquad
U=\frac{\mathrm d Z_t}{\mathrm dt}=\epsilon-Z_0,\quad t\in[0,1],
\quad\epsilon\sim\mathcal N(0,I).
\]

`t=0` 是数据端，`t=1` 是噪声端；采样沿 **1→0** 积分。训练网络的时间输入为 `1000t`。`t` 可按指定 `\pi(t)`（如 logit-normal）采样，改变的是训练风险加权，不改变速度目标定义。

记条件 `H=(G,S,W)`，则：

\[
\boxed{
\mathcal L_{\mathrm{FM}}(\theta)=
\mathbb E_{Z_0,\epsilon,t,H}
\left[\big\|v_\theta(Z_t,1000t;H)-(\epsilon-Z_0)\big\|_2^2\right].
}
\]

这里的 latent 约定（VAE posterior sample/mode、scale/shift、Decoder EMA）属于数学问题的一部分：若训练/采样中 `Z_0` 的分布不一致，就已经改变了目标分布。旧 Calli-VAE 的缩放、白底归零与 posterior mode 事故属于此类。详见 [latent contract](../vae_latent_contract.md)。

### 1.1 平方损失的 Bayes 最优解（恒等式）

令 `Q=(Z_t,t,G,S,W)`。在足够表达力、无限样本、正确模型装载的理想条件下：

\[
\boxed{v^*(Q)=\mathbb E[U\mid Q].}
\]

由条件期望的正交投影性质，

\[
\mathbb E\|U-f(Q)\|^2
=\underbrace{\mathbb E\|U-v^*(Q)\|^2}_{\text{条件不可约随机性}}
+\underbrace{\mathbb E\|v^*(Q)-f(Q)\|^2}_{\text{模型近似与优化误差}}.
\]

**结论**：真实书法的“同条件多种写法”使得单条训练配对的预测不可能都精确；训练 Loss 平台不等于网络不会生成合法多样性。模型学的是在给定带噪 latent 与条件时的**平均速度场**，通过反向概率流产生分布，而不是对每张 GT 的 deterministic 回归。

### 1.2 Writer 条件有用的严格数学含义（恒等式）

比较两套理想条件：
`Q_0=(Z_t,t,G,S)` 与 `Q_1=(Q_0,W)`；在**同一真实联合分布和同一损失**上，

\[
v_0^*=\mathbb E[U\mid Q_0],\qquad
v_1^*=\mathbb E[U\mid Q_1],\qquad
\Delta v_W=v_1^*-v_0^*.
\]

令最优风险分别为 `\mathcal R_0,\mathcal R_1`。由嵌套条件期望投影：

\[
\boxed{
\mathcal R_0-\mathcal R_1
=\mathbb E\|\Delta v_W\|_2^2\ge 0.
}
\]

这是**Writer 提供速度预测信息**的精确定义：若正确 Writer 能严格降低条件 FM 风险，就说明 Writer 在该统计模型下提供了额外信息。

但它**不是实际网络一定受益的保证**：有限样本、Writer ID 表没训练好、条件旁路、来源混杂、编码器容量与优化不稳定，都可能使加条件后的真实泛化更差。若 `W` 主要代理扫描来源，风险下降仍可能是虚假的“书家”信号。

**对应可做的实验**：对同一批 `(Z_t,U,G,S)` 固定 `t`、噪声、model weights，分别输入正确 Writer、错误 Writer、null Writer，测

\[
\Delta L_t^{\mathrm{wrong}}=
\mathbb E\big[\|v_\theta(Z_t,t;G,S,W')-U\|^2
-\|v_\theta(Z_t,t;G,S,W)-U\|^2\big].
\]

当 `\Delta L_t>0`，条件在这个面板上有预测价值；仍需独立 E-G、来源留出与 Writer-swap 才能确认是实际笔法而非捷径。

## 2. 一个关键推导：仅在 t=1 的 Writer Pair 监督只学习条件均值

这个命题对早先的 `S9 Pair`、Writer-only 梯度和大 K 原型实验至关重要。

### 2.1 在两个端点，最优速度的解析形式

利用 `Z_t=(1-t)Z_0+t\epsilon` 与 `U=\epsilon-Z_0`，且噪声与数据在给定条件下独立：

在 **数据端 t=0**，`Z_t=Z_0`，所以

\[
\boxed{v^*(z,0;G,S,W)=-z.}
\]

**在同一个数据 latent z 上，理想端点速度不依赖 Writer**。这不意味着最终生成不依赖 Writer，而是说明 Writer 的条件价值不能仅由 `t≈0` 速度差判断。

在 **纯噪声端 t=1**，`Z_t=\epsilon`，而 `\epsilon\perp Z_0\mid G,S,W`，所以

\[
\boxed{
v^*(\epsilon,1;G,S,W)
=\epsilon-\mu_W(G,S),\quad
\mu_W(G,S):=\mathbb E[Z_0\mid G,S,W].
}
\]

对同 `\epsilon,G,S` 的两个 Writer `a,b`：

\[
\boxed{v_a^*(\epsilon,1)-v_b^*(\epsilon,1)
=\mu_b(G,S)-\mu_a(G,S).}
\]

**因此，t=1 的条件速度差只包含 Writer 的条件平均 latent 的差异，而不包含其全部协方差、多峰性、逐字草化的条件分布形状。** 这是一个严格结论，不依赖生成器采用 U-Net 或 DiT。

### 2.2 为什么 S9 的同噪声监督“方向正确但不充分”

在同一个纯噪声 `\epsilon` 下取两个真迹 latent `z_a,z_b`，各自训练速度目标 `u_a=\epsilon-z_a` 和 `u_b=\epsilon-z_b`。则

\[
\boxed{u_a-u_b=z_b-z_a.}
\]

**注意顺序**：这里是 `v_a-v_b` 的真值 `z_b-z_a`，不是 `v_b-v_a`。与 [S9 实验记录](../1028/generation2.md) 的原有 Pair 方向一致，实际代码必须逐个核对取差顺序。

然而，对条件抽样的随机真迹，训练 Pair 的理想极限是条件均值差。单张 GT `z_a,z_b` 的差还包含随机写法/扫描噪声。把 Pair 只放在 `t=1`，最多帮助对齐 Writer 的**第一矩**；它不会强制学到多种合法结字之间的概率比例。

**设计启示**：要回答“书家分布形状是否学对”，必须检查**中间 t** 的条件速度差、同字同噪声 Writer-swap、输出图的 Writer 关系几何，不能仅用端点 Pair loss 或 Writer table prototype 自我验收。

### 2.3 一个高斯反例：均值完全相同，Writer 仍可重要

假设 `Z_0\mid W=w\sim\mathcal N(\mu_w,\Sigma_w)`，`\epsilon\sim\mathcal N(0,I)` 独立，则

\[
Z_t\mid w\sim\mathcal N((1-t)\mu_w,\,(1-t)^2\Sigma_w+t^2I).
\]

设 `K_w(t)=(1-t)^2\Sigma_w+t^2I`，可用高斯条件期望直接计算：

\[
\mathbb E[Z_0\mid Z_t=z,w]
=\mu_w+(1-t)\Sigma_wK_w(t)^{-1}(z-(1-t)\mu_w)
\]

以及

\[
v_w^*(z,t)=\frac{z-\mathbb E[Z_0\mid Z_t=z,w]}{t},\qquad 0<t\le1.
\]

即使两个 Writer **均值相等** `\mu_a=\mu_b`，只要 `\Sigma_a\ne\Sigma_b`，它们在 `0<t<1` 的速度仍不同；**在 t=1 差异才消失**。

这给出一个可证伪预言：如果现实书家信息更多体现在合法写法的分布形状，而不是一个固定均值偏移，那么**只训练 t=1 Pair 会表现不足，而中间时间的 Writer 反事实监督更敏感**。现有 DINO 原型、QDA/LDA 实验只提供间接动机，尚不能宣布该反例就是本项目的真实生成机制。

## 3. 怎样理解“字符主效应93.9%，Writer主效应0.5%”

[2026-10-25 DINO诊断](../../ANALYSIS/style_structure_diagnosis_20261025.md) 在 DINOv2-B CLS 特征上构造平衡面板，采用类 ANOVA 的几何分解：

\[
h_{w,c}=\mu+a_c+b_w+d_{w,c}+\eta_{w,c},
\]

其中 `a_c` 是字符内容主效应，`b_w` 是书家共享主效应，`d_{w,c}` 是字和书家的交互/剩余结构，`\eta` 是作品噪声（实际分析可因有限样本分组约定而归类不同）。

报告在其**指定表征/平衡面板**测得：

| DINO CLS 量 | 经验占比 / 检验 | 数学上能说什么 |
|---|---|---|
| 字符主效应 | **93.9%** | 这套特征主要区分内容字形 |
| Writer 主效应 | **0.5%** | Writer 在**全局平均向量**的能量相对小 |
| Writer×字残差 | **5.6%** | 逐字差异存在，但不等于可从 `G` 的形状类别泛化预测 |
| Writer 线性探针 | 10 类留字 **43.6%**（随机10%） | Writer 信号跨字可读，低能量≠无信息 |
| 加局部 patch | 留字 **45.5%** | 局部特征能增益，幅度有限 |
| 原型 K-shot | K1 **12.6%**→K32 **24.0%**，未饱和 | 小 K 均值原型不足以稳定表示真实 Writer |
| Writer×STD 双线性交互 | `\Delta R^2_\mathrm{test}\approx0` | 在该 DINO 特征/低秩模型/留字协议下额外结构信息无稳定贡献 |

另有判别有效谱 `d_{\mathrm{eff},95}\approx7`，是当前**10 类线性判别子空间中信息如何分布**的描述，不是“所有书法家本质只有 7 个风格维度”、也不是“7 维 embedding 保证生成无损”。

与之不冲突的 E-G 实验：E-G 是专门监督训练的 Writer 特征，在 real 同书体异字 pair 上 AUC **95.98%**；DINO 是主要承载视觉语义的通用表征，两者**编码器不同、Top-1与AUC任务也不同**，不能比较 43.6% 和95.98% 的大小来宣告 Writer 可分性发生矛盾。

### 为什么 Writer-ID 比“先提2张参考图再平均”更合理，但不是定理

对固定的 10 个**训练时已知 Writer**，可训练的 Writer-ID 表为每位书家累积利用大量样本；K-shot 均值原型只利用少量参考图。统计上，若每张图像的 Writer 信号被更强的 `a_c` 和逐字 `d_{w,c}` 淹没，小 K 均值具有较高估计方差。

但这只能支持**当前数据/成本下优先 Writer-ID**，不能推导“参考图总是失败”或“Writer×STD 交互必然零”：更大的 K、更好的特征、未见 Writer 的迁移任务、可学习聚合器都可能改变结论。这也是为什么不应把书家条件强行缩成单笔粗细、局部曲率或人工码本。

## 4. VAE 与 Decoder：把“条件速度正确”和“像素画出来”分开

记 VAE 编码和解码为 `E_\eta,D_\psi`，latent contract 为：

\[
z_0=\alpha E_\eta(x)\quad(\text{按模式选择 posterior sample 或 mode}),
\qquad \widehat x=D_\psi(z_0/\alpha).
\]

对于标准 SD-VAE，常见 `\alpha=0.18215`，但旧 Calli-VAE、其他模型可能有额外 shift；**不能把此常数无条件套到每个 checkpoint**。必须以其实际训练与采样代码为准。

生成器学习 `p_\theta(z_0\mid g,s,w)`；Decoder 映射回 RGB。对固定 latent 的小扰动有局部线性化：

\[
D_\psi(z+\delta z)-D_\psi(z)\approx J_D(z)\,\delta z.
\]

对模型参数改变有：

\[
D_{\psi+\delta\psi}(z)-D_\psi(z)
\approx J_\psi(z)\,\delta\psi.
\]

**含义**：Decoder 微调即使不改变 `z` 也能改变边缘、笔宽、飞白、背景甚至局部连通性。由于 `J_D` 非均匀且非线性，不能从 LPIPS 改善直接证明 Writer 速度场更准确；**也不能武断说 Decoder 只改变高频、永远不改变字符内容**。

### J0/J1/J2/J3 的数学解释

设普通 FM 为 `L_\mathrm{FM}`，图像域辅助项例如

\[
L_{\mathrm{img}}=
\|\widehat x-x_{\mathrm{GT}}\|_1+
\lambda_p\,\mathrm{LPIPS}(\widehat x,x_{\mathrm{GT}})
+\lambda_e\,\|\nabla\widehat x-\nabla x_{\mathrm{GT}}\|_1,
\]

还可有来自**真实 latent**的 Decoder 重建约束 `L_\mathrm{rec}`。这只是 J 类损失的概念形式，系数/梯度比例须按本轮配置与模型重新校准。

J1 冻结 Decoder 时 `\nabla_\psi L_{\mathrm{img}}=0`；J2/J3 可更新 `\psi`，使网络多了一个直接改善 RGB 误差的自由度。结果 J0→J2 的 SSIM **+0.0049**、LPIPS 约 **−0.007**，而 Writer Top-1 未同步上升，符合“图像域目标直接优化局部成像”的解释。

**识别因果的方法**：固定相同 `z`，分别解码 `D_{\psi_0}(z)` / `D_{\psi_1}(z)`；再交换同一 Encoder/生成器与不同 Decoder 的配对。这能区分 Decoder 直接改变画质与联合微调让 U-Net 改变 latent 的贡献，仍需检查细笔与字符错误是否变化。


## 5. 为什么“更多监督”反而可能损害图像目标：12ch 的梯度竞争

Moyi/早期 DiT 的 12ch 将 4 个图像 latent 通道与 8 个 Canny/骨架辅助 latent 通道一起预测。令

\[
L_{\mathrm{multi}}=L_{\mathrm{image}}+\lambda L_{\mathrm{aux}},
\quad g_i=\nabla_\theta L_{\mathrm{image}},
\quad g_a=\nabla_\theta L_{\mathrm{aux}}.
\]

沿一次小学习率梯度下降 `\theta'=\theta-\eta(g_i+\lambda g_a)`，对主要图像损失作一阶展开：

\[
L_{\mathrm{image}}(\theta')-L_{\mathrm{image}}(\theta)
\approx-\eta\left(\|g_i\|_2^2+\lambda\langle g_i,g_a\rangle\right).
\]

因此若 `\langle g_i,g_a\rangle<0` 且 `\lambda|\langle g_i,g_a\rangle|>\|g_i\|^2`，**总 loss 在下降时图像 loss 仍可能上升**。即使内积为正，额外分支也会消耗参数、显存、NFE 和样本预算。

项目中的物理记录：12ch 等权辅助任务曾占最终层约 **51%** 的梯度量，patch_embed 处辅助梯度约为图像的 **2.6×**；所用 4ch+g 配方比 12ch+wz 更早生成清晰字符。但历史 12ch 同时存在 **CFG image-channel 作用域**、**白底归零恢复**等 bug，因此不能把所有性能损失都因果归于梯度冲突。[12ch/白底归零复盘](../archive/system_notes/59_12ch_and_white_zero_retrospective.md)。

### 为什么骨架条件与骨架目标不等价

`G` 是**预测时已知条件**；令辅助目标 `A=\mathrm{Canny}(X)` 或 `\mathrm{Skeleton}(X)`，它仍是关于真实 `X` 的标签。训练 `(X,A)` 的联合分布可能产生有效正则化，**并非信息论上“完全冗余=必然有害”**；只是当前骨架信息已由 `G` 注入，额外预测 `A` 的边际监督价值要与其训练干扰一起测量。

正确消融要冻结数据、参数/NFE、VAE、条件、LR、随机性；同时记录 per-channel loss、梯度范数及 cosine：

\[
\rho_{\mathrm{grad}}=
\frac{\langle g_i,g_a\rangle}{\|g_i\|\,\|g_a\|}.
\]

只有这些证据足够，才能决定权重、分支解耦或直接删除辅助目标。

### 字形/白底尺度同属潜空间合同

旧白底归零令 `z'=z-z_{\rm white}`，它改变了 latent 的均值、稀疏度与尾部；解码时必须 `z=z'+z_{\rm white}`。否则 `D_\psi(0)` 并非白纸，图像呈现错误颜色。这是一个**可逆变量替换的工程合同被破坏**，不是“网络不擅长黑白图”的理论结论。具体的 gradient/CFG/scale-shift 问题皆要求强断言，而非依赖散落于脚本中的默认值。

## 6. 两种评价风险：字符识别风险和 Writer 验证风险

### 6.1 OCR 应被视为含测量误差的内容判据

设冻结 OCR 为 `q_\varphi(c\mid x)`，生成图 `\widehat X_\theta`，可定义：

\[
A_{\rm OCR}(\theta)=
P[\arg\max_c q_\varphi(c\mid \widehat X_\theta)=C].
\]

此量是模型可识别率，不是字符结构绝对正确率。由于读光当前有未覆盖/域外字、异体、书法简写，必须同时记录：

\[
A_{\rm GT}=P[\arg\max q_\varphi(c\mid X_{\rm GT})=C],
\]

以及 **仅在 OCR 能识别对应 GT 的固定索引子集上**的条件准确率：

\[
A_{\rm conditional}=
P[\widehat C(\widehat X_\theta)=C
\mid\widehat C(X_{\rm GT})=C].
\]

要同时报告全体 `A_{\rm OCR}` 与 `A_{\rm conditional}`；后一项不能偷偷替代全体。识别对错还要检查 vocab 覆盖率、置信 margin、混淆类及 **Confident-Wrong**。局部断笔和错部件需要人工错误注入的 Corruption-Test；高 Top-1 不等于形态全对。

**特别的统计条件**：不能用 `sorted(glob('g*.png'))` 的字典序配对字符。每张图需由 `(panel_id,sample_idx,Unicode,writer,script,GT_sha)` 明确 join，对文件名索引做整数解析，并有 GT sanity；单纯 GT OCR 很高也无法排除少量繁简/异体错配。旧 0%/7.8% 正是配对与输入宽高事故，不是模型结论。[本日修复](../../1031.md)。

### 6.2 E-G 是关系度量，不是可直接求和的“风格质量”

E-G 在同书体、不同字符的真实作品上进行 Writer 核验。`AUC=95.98%` 对应的是**正负对的得分排序能力**，不是“书家分类准确率95.98%”。E-G 的训练、参考库、测试作品必须识别来源重叠；低阶墨量/宽度/布局线性探针 AUC49.35% 与残差后 E-G93.11%，仅排除这一**类简单捷径**，不排除扫描/作品来源复杂偏差。

对于不同协议的 AUC，可定义诊断性的相对饱和度：

\[
S_{\rm EG}=\frac{\mathrm{AUC}_{\rm gen}-0.5}
{\mathrm{AUC}_{\rm GT}-0.5},
\]

前提是二元 Writer 核验、正负采样和参考池设计一致，且分母大于0。`\mathrm{AUC}_{\rm GT}` 是该面板**经验真迹参照**，不是真正的数学上界。例：旧 S5@50k AUC0.8798、GT0.9238，`S\approx0.896`；Deep85 在其187协议 AUC0.8557、GT0.8976，`S\approx0.895`。因此原始 AUC 的跨协议大小不能直接宣告 S5 胜出。[E-G旧协议](../1030/eval_style_v2/EG_RANKING.md)。

**AUC、EER、mAP、Style-FID/KID 均用到同一 E-G 特征，不是四个独立裁判。** 另一个 HWD 网络在真实中文作品上的跨字 Writer AUC 只有约51–54%，无法作可靠“独立复核”；CHAED+Ridge的字符留出审美回归也未过可靠校准。应优先获得独立数据/来源和真正能判别 Writer 的第二网络，再声称多裁判一致。[HWD](../1029/hwd/REPORT.md) / [CHAED](../1029/chaed/REPORT.md)。

## 7. Writer-swap 的几何理论：方向、幅度与内容稳定性

给定同 `(G,S,\epsilon)`，只改变 Writer：

\[
\widehat X^a_\theta=D_\psi(F_\theta(\epsilon,G,S,a)),
\qquad
\widehat X^b_\theta=D_\psi(F_\theta(\epsilon,G,S,b)).
\]

`F_\theta` 表示完整 FM 反向采样，并非单步网络前向。设冻结的跨字 Writer 特征为 `\phi(X)`，则生成的两书家差向量

\[
\Delta_{\rm gen}^{a,b}
=\phi(\widehat X^a_\theta)-\phi(\widehat X^b_\theta).
\]

真实参考差向量需在尽量匹配书体、字符、作品来源的协议下估计，例如

\[
\Delta_{\rm real}^{a,b}(c,s)=
\mathbb E[\phi(X)\mid a,c,s]
-\mathbb E[\phi(X)\mid b,c,s],
\]

现实共享字可能太少，因此要对真实中心估计的不确定性做 bootstrap，并在全局/字符层次分开展示。

至少测量三轴：

\[
\operatorname{Direction}=
\frac{\langle\Delta_{\rm gen},\Delta_{\rm real}\rangle}
{\|\Delta_{\rm gen}\|\,\|\Delta_{\rm real}\|},
\qquad
\operatorname{Amplitude}=
\frac{\|\Delta_{\rm gen}\|}{\|\Delta_{\rm real}\|},
\]

\[
\operatorname{ContentStability}
=P[\operatorname{OCR}(\widehat X^a)=
\operatorname{OCR}(\widehat X^b)=C].
\]

**解释**：

- `\operatorname{Direction}` 高但 `\operatorname{Amplitude}\ll1`：书家差异方向正确，却被平均化/压缩；
- 幅度大而方向错：有风格变化，**不是目标 Writer 风格**；
- 风格提高同时内容稳定性下降：模型可能用错误部件/过度草化换取判别得分；
- 幅度接近1不是充分条件：真实特征可能含扫描来源混杂，且不同字的合法差异强度不同。

已有 J 系关系几何中方向余弦约 **0.85**、幅度比 **0.66–0.75**；这是**该组特征与现有面板**的观察，尚缺完整同字同噪声 swap 的独立重建，不可把 0.72 当所有生成模型的普适风格压缩率。[E-G关系分析](../1030/eval_style_v2/FINAL_REPORT.md)。

## 8. 为什么不能只用一个加权总分宣布胜利：Pareto 与分布评估

设模型性能向量为：

\[
\mathcal M(\theta)=
\big(A_{\rm OCR},\,A_{\rm EG},\,-\mathrm{LPIPS},\,
\mathrm{InkIoU},\,A_{\rm unseen},\,\operatorname{ContentStability}\big).
\]

同一数据/同一评估器/同一推理算力/同一采样协议下，如果模型 A 在每个预设主指标不弱于 B 且至少一个指标显著更优，则可称 **Pareto 改善**；否则是一个**权衡点**，不是“全面优于”。

不要任意将多个数值标准化后相加成单标量，因为每个指标的度量不确定度、单位、训练教师独立性不同。例如：

- SSIM/LPIPS 衡量与**单个配对 GT** 的相似度，合法第二写法也会受到惩罚；
- OCR Top-1 衡量识别器是否猜对 Unicode，忽略同字多写法的细腻程度；
- E-G AUC 衡量跨字符身份验证，不能代替内容或墨色；
- Ink-IoU 高可能奖励粗笔、连笔、牺牲真实书家飞白；
- FID/KID 的特征分布距离无法判断目标 Writer 是否对号入座。

### 真迹—真迹相似度为什么不是 SSIM“上界”

以**平方欧氏距离**为例（不是 SSIM），若 `Y,Y'` 是给定条件 `H=(G,S,W)` 下独立同分布的两张作品，则

\[
\mathbb E[\|Y-Y'\|^2\mid H]
=2\,\mathrm{tr}\operatorname{Cov}(Y\mid H).
\]

该恒等式解释“同条件真迹之间仍会有显著差异”。如果对单个 GT 的像素 MSE 最优预测为 `\mathbb E[Y\mid H]`，它可能表现为平均字形甚至模糊笔画；但 **SSIM 不是平方距离、存在非线性局部对比度/结构项**，不能简单把平均真实真迹 SSIM 作为理论最高。仓库里实测的“条件平均0.6503、均匀抽对0.6235”属于**协议依赖的参考统计量**。[107 评测](../107/02_model_benchmarks.md)。

### 不确定性处理

- 一组汉字每个有3个噪声 seed，不等于 `3N` 个独立汉字。**先逐字聚合 seed，再按字符聚类 bootstrap**；涉及 Writer 级结论还要按 Writer 分层或层级重采样。
- 对同样本两模型采用配对差、预先指定优先指标，而不是单独看两个均值差。
- 超参数、CFG/SDE、模型与样本如果都在同一 test200 上选择，再报其 test 置信区间，会产生选择偏差；验证集选择后要在冻结测试集确认。
- 样本来源、异体Unicode与同源作品必须标注；“全187字”不等于“187个全阶段严格未见字”。

## 9. SDE/ODE 推理的数学边界：旧 free-lunch 为什么不能直接搬

仅在 `\epsilon\perp Z_0\mid H`、噪声真正是标准高斯、流匹配采用前述线性桥且时间 `t>0` 时，有精确 score 关系：

\[
s^*(z,t,H)=\nabla_z\log p_t(z\mid H)
=-\frac{\mathbb E[\epsilon\mid Z_t=z,H]}{t}
=-\frac{z+(1-t)v^*(z,t,H)}{t}.
\]

令生成时间 `u=1-t`。无噪声的概率流 ODE 漂移是 `-v^*`。对任意仅依赖时间的非负扩散系数 `D(t)`，构造

\[
\boxed{
\mathrm dZ_u=
\left[-v^*(Z_u,t,H)+D(t)s^*(Z_u,t,H)\right]\mathrm du
+\sqrt{2D(t)}\,\mathrm dB_u.
}
\]

Fokker–Planck 方程里新增的 `-D\nabla\cdot(p_ts^*)` 与 `+D\Delta p_t` 相互抵消（`p_t s^*=\nabla p_t`），因此**在精确速度/score、合适边界条件下**，它与原条件 ODE 共享**每个时间的边缘分布**；不保证每条固定噪声路径逐像素相同。

当 `D(t)=\gamma^2t(1-t)`，用代码的下降 `t` 记号：

\[
b_{\rm code}=v^*-D(t)s^*
=v^*+\gamma^2(1-t)[z+(1-t)v^*].
\]

旧式直接放大 `v` 而不加 `z` 状态项并不是这个方程；C2OT 通过依赖 `Z_0` 的噪声重排破坏了推导所需独立性，不能对旧模型直接假定保持同分布。`CFG\ne1` 或有限模型误差又引入额外偏差。

本项目证据：v68 的旧采样器大增益确实曾被观察到，但在 v80 更正独立配对链上，三 seed 的 SSIM **+0.000576**，95%CI **[−0.000315,+0.001512]** 跨零；Frag **−0.071973**，探索性95%CI **[−0.147870,−0.002186]**。因此只支持“某些断笔统计可能轻微改善”，**不支持普适 SDE 免费画质提升**。详见 [独立 SDE 数学审计](../experiments/2026-10-12-v68-v70-sde-role-and-free-lunch.md)。

## 10. 通用预训练、目标适配与混合阶段：如何识别真实迁移增益

设通用分布 `P_0(Z_0,G,S,W)` 和目标 Top10 分布 `P_T` 不同。预训练的价值可以体现为：
- 更快学习字符/STD→墨迹的共同映射；
- 目标域在相同**有条件**样本曝光量下更低的风险；
- 对 `\mathrm{Unseen\ in\ All\ Stages}` 字符的更好泛化；
- 这些收益在引入 Writer 后仍不造成明显域偏置。

但观察到 `P0@50k` 已能生成书法字或 P1 后 Writer AUC 上升，**不是净正迁移证据**；需要一个同架构、同总算力、同目标样本曝光的无预训练对照。

### 10.1 两种必须区分的曝光量（不是 optimizer 更新数）

令阶段 `k` 训练了 `T_k` 次更新，Top10 图像概率为 `p_k`，Writer 非空概率为 `1-d_k`，则按相同 batch 大小归一后的 Batch 等效量是

\[
E_{\rm Top10,image}=\sum_k T_kp_k,
\qquad
E_{\rm Top10,conditional}=\sum_kT_kp_k(1-d_k).
\]

P75 的已有配方（P0不含目标Top10；P1 5k×30%目标×75%非空，P2 15k×90%，P3 5k×90%）给出：

\[
\begin{aligned}
E_{\rm image}&\approx 1.5k+15k+5k=\boxed{21.5k},\\
E_{\rm conditional}
&\approx1.125k+13.5k+4.5k=\boxed{19.125k}.
\end{aligned}
\]

这不是说“模型只更新了19.125k步”，而是对目标域条件曝光的**等价样本量估计**。需要按实际日志和所有 dropout/配对采样确认。若 A 再新增 25k Top10、Writer dropout10%，则预计 `E_{\rm image}≈46.5k`，`E_{\rm conditional}≈41.625k`；先前把这两个口径都称“46.5k条件曝光”是混淆。

### 10.2 当前 A/B/C 说明什么，不能说明什么

- **A**：从原 P3A 继续 Top10，已完成 +5k 首评：SSIM `0.5857→0.5914`。同时修正了 Style LR、optimizer 组及 restart，因此只是**更充分的新后训练方案**有效，不能隔离纯步数收益。
- **B**：从同一 P0@50k 直接 Top10 25k，跳过 P1；相比原 P75 混合轨迹同时改变 Top10曝光、LR、数据顺序与 optimizer，因此即使 B 胜也不等于“混合阶段有害”。
- **C**：P0@50k Writer-first 5k + 联合20k；如果 B/C 起点、总数据、噪声、训练步数、测试协议完全一致，**C vs B 更接近测试 Writer-first 学习率/冻结配方**的因果实验。

**真正验证预训练**建议双口径：① 相同目标曝光比较 `P0+FT` vs 只在 Top10 训练的 scratch；② 相同**总算力**比较 `P0+FT 75/100k` vs 同架构 scratch75/100k，并在严格 Unseen-in-Top10 / Unseen-in-All-Stages 上报告内容与风格。总算力对照与曝光对照回答不同问题，缺一不能宣布正迁移。

## 11. 一个“可以推翻”的下一代研究框架

| 数学或机制命题 | 当前证据 | **能够支持或推翻它的实验** |
|---|---|---|
| Writer 有独立条件信息（`\mathcal R_0-\mathcal R_1>0`） | E-G 能区分 Writer，S5 身份领先 | 同`Z_t,U,G,S`固定输入，比较 correct/wrong/null Writer FM risk；按 t/字/书家分组 |
| Writer 效应在中间时间比 `t=1` 包含更多分布信息 | 第2节高斯反例是**严格理论可能性**，真实数据未证 | 跨 t 的 Writer 条件响应、同字同噪声 Writer-swap 与 GT 方差/分布匹配 |
| Writer ID 比小 K 均值原型更适合当前 Top10 | K32 Top-1仍24%、全量线性42.6% | 同数据/同预算：ID embedding vs 大 K、端到端参考聚合，留字符和来源 |
| 复杂空间注入能提高真实 Writer 差异但不必提高内容 | S4/S5 E-G AUC 优，S0 SSIM 优 | 85k/同面板/三seed，独立 E-G2/人工 2AFC、Writer-swap 内容稳定 |
| Decoder 主要改变局部成像 | J2/J3 SSIM/LPIPS 增益，Writer 未同步增 | 固定 latent Decoder swap + OCR/多尺度结构和真迹笔锋统计；若字符改变即否定“只改纹理” |
| 额外辅助目标可能竞争主生成目标 | 12ch 梯度/4ch 结果 | 同NFE/参数/数据的4ch vs12ch和梯度 cosine/图像主损失差 |
| P75 是后训练不足而非预训练无用 | 当前条件曝光约19.1k，A首点改善 | A/B/C完结后等样本、等总算力的 scratch/transfer 严格 unseen 对照 |
| OCR+E-G 能共同衡量任务 | 真实OCR80%（200张）、E-G真迹AUC95.98 | 完整 Real-Test、Corruption-Test、来源留出、writer/char 反事实，加人工 blind review |

### 建议保留的最小新训练接口

1. **内容**：STD-soft `G` 作为空间条件；字符 ID 若用于对照，必须作为另一个独立模式，不能与STD混合后宣称纯开集。
2. **风格**：训练时已知 Writer 的 Embedding，可经共享条件网络进入生成器；是否需要局部表征/额外分支，先看真实 Writer 条件风险与swap收益。
3. **主训练**：普通、严格定义时间与 latent contract 的 FM；辅助 Loss 必须有独立收益和梯度门控。
4. **成像**：Decoder 作为独立可评审的画质模块，可后训练，但必须同时监测OCR与Writer。
5. **评价**：不对不同协议求总分，严格声明每个样本的 char/Writer/GT/seed/VAE/EMA/metric 版本。

**数学总结**：在理想平方损失条件 FM 下，Writer 的价值由 `\mathbb E\|\Delta v_W\|^2` 精确定义；但在纯噪声端它只体现平均 latent，真正的多种写法与笔法分布必须在中间时间和完整生成样本上检验。这解释了为什么“更强 Writer loss”“多一个空间注入器”“更好的 Decoder 指标”都不能单独宣告书风生成问题解决。

## 12. 证据与实现导航

- [全部实验阶段总结](../1031/README.md) · [All](../../All.md) · [Milestones](../../Milestones.md) · [Current](../../Current.md)。
- [FM 源码](../../src/loss/flow_matching.py) · [VAE合同](../vae_latent_contract.md) · [SDE推导/复核](../experiments/2026-10-12-v68-v70-sde-role-and-free-lunch.md)。
- [DINO风格结构](../../ANALYSIS/style_structure_diagnosis_20261025.md) · [S9 Pair 方向](../1028/generation2.md) · [S/J实验](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)。
- [E-G检测及混杂边界](../1030/eval_style_v2/FINAL_REPORT.md) · [E-G协议归一](../1030/eval_style_v2/EG_RANKING.md) · [HWD](../1029/hwd/REPORT.md)。
- [旧108评测账本](../108/03_metrics_ledger.md) · [107勘误](../107/03_evidence_grade_and_errata.md) · [P75 LR审计](../../1030p125.md) · [OCR修复及后训练状态](../../1031.md)。
