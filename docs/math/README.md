# 书法条件生成的数学框架：内容、Writer、随机写法与 VAE

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
