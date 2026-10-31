# 条件 Flow Matching 的数学基础与时间轴：可证定理和经验限制

> 目标：给 Writer 条件训练提供**可计算而不是口号式**的机制解释。本页用仓库 [`src/loss/flow_matching.py`](../../src/loss/flow_matching.py) 的真实时间定义；推导属于数学恒等式，任何“哪个时间段值得加Writer监督”的结论仍需[4090反事实评测](../1031/05_NEXT_EXPERIMENTS.md)验证。

## 1. 模型的生成分布与插值定义

- `Z_0∼p_{\mathrm{data}}(z\mid G,S,W)`：VAE训练数据latent；`G`是STD图，`S`书体，`W`书家。
- `ε∼N(0,I)`，并假设 **`ε\perp Z_0\mid(G,S,W)`**。这是标准独立配对 FM，**旧 C2OT 噪声重排未必满足**。
- 训练时间 `t∈[0,1]`，物理网络时间输入为 `1000t`；`t=0`数据端、`t=1`噪声端：

\[
Z_t=(1-t)Z_0+t\epsilon,\qquad
U_t=\frac{\mathrm dZ_t}{\mathrm dt}=\epsilon-Z_0.
\]

采样从 `t=1→0` 数值积分。训练目标：

\[
L_{\rm FM}(\theta)=
\mathbb E_{Z_0,\epsilon,t,G,S,W}
\big[\|v_\theta(Z_t,1000t;G,S,W)-U_t\|^2\big].
\]

时间采样分布 `\pi(t)`（如logit-normal）会改变目标风险中**哪些时间点得到更多训练梯度**，但不改变上式中单点的最优条件速度。

## 2. 第一条严格定理：最优FM速度是条件期望

对固定输入 `Q=(Z_t,t,G,S,W)` 和平方损失，Bayes最优解是

\[
v^*(Q)=\mathbb E[U_t\mid Q].
\]

用条件期望的正交投影恒等式，对任意可测函数 `f`：

\[
\mathbb E\|U_t-f(Q)\|^2
=\mathbb E\|U_t-v^*(Q)\|^2+
 \mathbb E\|v^*(Q)-f(Q)\|^2.
\]

第一项是同条件下写法随机性/噪声产生的不可约项，第二项是模型的近似+优化差距。因此**训练FM loss继续降，不一定表示字符泛化或Writer忠实度继续升**；还要知道它在哪个时间、字符与Writer上下降。

### 嵌套条件的Writer价值恒等式

设 `Q_0=(Z_t,t,G,S)`，`Q_1=(Q_0,W)`，两种最优速度 `v_0^*=\mathbb E[U_t\mid Q_0]`、`v_1^*=\mathbb E[U_t\mid Q_1]`。在**同一联合数据分布、相同损失度量**下：

\[
\boxed{R^*_0-R^*_1=\mathbb E\|v^*_1-v^*_0\|^2\ge0.}
\]

这说明Writer条件在理想模型里**不会增加Bayes最小FM风险**；若存在差异则信息量严格为正。它**不保证有限网络、错误Writer映射、少样本泛化、不同优化器条件下实际风险也下降**。也不等于Writer信息一定来自笔法；Writer与扫描作品来源的共线同样可以降低该风险。

**对应测量**：对固定 `Z_t,U_t,G,S`，输入正确/错误/null Writer，比较 `ΔL_t` 与 `‖Δv_t‖`。这只能说明模型在当前样本/训练分布上用了Writer，若想说“用的是书家风格”还需来源分层、OCR和Writer-swap验证。

## 3. 第二条严格结论：两端Writer速度不同的本质

由 `U_t=(Z_t-Z_0)/t`（`t>0`）：

\[
v_w^*(z,t)=\frac{z-\mathbb E[Z_0\mid Z_t=z,G,S,W=w]}{t}.
\]

### t=0 数据端

在 `Z_t=Z_0=z` 时且噪声独立、均值为0：

\[
\boxed{v_w^*(z,0)=-z.}
\]

**对固定相同数据latent z，理想端点速度不依赖Writer。** 这不意味着Writer只影响生成早期，更不能把 `t≈0` 的小差异解释为“条件网络坏掉”。有限神经网络会存在非零端点偏差。

### t=1 纯噪声端

`Z_t=ε=z`，独立于数据 `Z_0`：

\[
\boxed{v_w^*(z,1)=z-\mu_w,\quad\mu_w=\mathbb E[Z_0\mid G,S,W=w].}
\]

同 `z,G,S` 两个Writer `a,b` ：

\[
\boxed{v_a^*(z,1)-v_b^*(z,1)=\mu_b-\mu_a.}
\]

**精确结论**：纯噪声端Writer速度差只能直接反映其**条件均值**差，不能在这一点约束协方差、多峰写法的概率比例。不是说 `t=1` 的训练没有价值，而是说它本身不包含完整的条件分布信息。

### S9 Pair真值的符号与位置

同一个 `ε`，两个真迹latent `z_a,z_b`：

\[
u_a=\epsilon-z_a,\qquad u_b=\epsilon-z_b,\qquad
\boxed{u_a-u_b=z_b-z_a.}
\]

这是 `v_a-v_b` 的方向；若实现比较的是 `v_b-v_a`，真值就要取反。[S9实现设计](../1028/generation2.md)。

**更重要的区别**：
- 当 `t=1`，两个配对的输入坐标都是 `Z_1^a=Z_1^b=ε`，属于**同位置条件差**。
- 当 `0<t<1`，即使共用噪声，`Z_t^a=(1-t)z_a+tε` 与 `Z_t^b=(1-t)z_b+tε` **不是同一个位置**。两边速度差混合了Writer条件变化与输入位置变化，不能当成纯粹的 `Δv_W(z,t)`。

所以想测**真正的Writer干预**，必须把两种Writer喂给**同一个** `z_t`，其余条件完全固定。

## 4. 第三条严格关系：Writer速度差和条件score差是同一信息的不同坐标

独立高斯线性桥的条件边缘密度记为 `p_t(z\mid G,S,W)`，score为 `s_w(z,t)=\nabla_z\log p_t(z\mid G,S,W)`。高斯积分恒等式：

\[
s_w(z,t)=-\frac{\mathbb E[\epsilon\mid Z_t=z,G,S,W=w]}t
=-\frac{z+(1-t)v_w^*(z,t)}t.
\]

对 `0<t<1` 解得：

\[
\boxed{v_w^*(z,t)=-\frac{z+t\,s_w(z,t)}{1-t}.}
\]

在相同 `z,t,G,S`、不同Writer：

\[
\boxed{\Delta v^*_{a,b}(z,t)
=-\frac{t}{1-t}\Delta s_{a,b}(z,t).}
\]

这是**精确关系**，解释为什么中间时间的Writer响应包含完整条件密度的score差，而不是单一全局Writer均值。但它不说明差异最大一定出现在 `t=.5`；那取决于真实分布、训练时间密度和信号幅度。

在 `t→0`，若score差有界，`Δv`按 `O(t)` 缩小；在 `t→1`，公式中的系数虽发散，但`Δs`同时趋零，极限为前面的均值差。**绝对Writer差随t的曲线不应在没有标准化的情况下硬比较不同噪声段**，必要时与目标速度的均方范数和预测误差同时报告。

## 5. 高斯反例：同均值Writer可能仍有不同的写法分布

假设 `Z_0\mid W=w,G,S∼N(\mu_w,\Sigma_w)`，协方差为正定：

\[
Z_t\mid w\sim
N\big((1-t)\mu_w,\ K_w(t)\big),\quad
K_w(t)=(1-t)^2\Sigma_w+t^2I.
\]

高斯条件期望：

\[
\mathbb E[Z_0\mid Z_t=z,w]
=\mu_w+(1-t)\Sigma_w K_w(t)^{-1}
[z-(1-t)\mu_w].
\]

若两个Writer `\mu_a=\mu_b` 但 `\Sigma_a\ne\Sigma_b`，在 `t=1` 它们 `Δv=0`，在某些 `0<t<1` 则 `Δv\ne0`。这是**数学上严格存在的情形**：Writer既可改变平均字形，也可改变合法写法的方差/形态分布。

**研究边界**：[DINO诊断](../../ANALYSIS/style_structure_diagnosis_20261025.md) 的QDA即使有分类准确率变化，留字 logloss 比 LDA差 **1.34**，并未证实本项目Writer的稳定“协方差几何”就是生成主要瓶颈。这只是说明“仅对齐均值”理论上不足，而不是给Writer协方差分支立项的证据。

## 6. SDE/ODE：完整公式而非“噪声能自动修笔画”

对 `t=1→0` 的生成时间 `u=1-t`，若速度和score精确，且 `D(t)\ge0` 仅依赖时间，则：

\[
\mathrm dZ_u=
[-v^*(Z_u,t)+D(t)s^*(Z_u,t)]\,\mathrm du
+\sqrt{2D(t)}\,\mathrm dB_u.
\]

其 Fokker–Planck 中 `D(t)\nabla p` 的score漂移与扩散项抵消，因此共享精确边缘分布 `p_t`。回到代码下降时间 `dt<0` 的漂移：

\[
b_{\rm code}=v^*-D\,s^*
=v^*+\frac{D(t)}t[z+(1-t)v^*].
\]

例如 `D(t)=\gamma^2t(1-t)`，得到：

\[
b_{\rm code}=v^*+\gamma^2(1-t)[z+(1-t)v^*].
\]

**不能只加随机项却用不匹配的 `v` 漂移**；旧式乘性速度修正不具备这项精确边缘保持证明。C2OT曾令训练噪声与数据相关，独立高斯前提不再自动成立。CFG≠1、网络预测误差与有限NFE会改变实际最优γ。详情参见 [SDE仓库审计](../experiments/2026-10-12-v68-v70-sde-role-and-free-lunch.md)。

同一v80正确链的三seed：SSIM变化 **+.000576**，CI **[−.000315,+.001512]**；Frag变化 **−.071973**，CI **[−.147870,−.002186]**。所以目前只支持**局部断笔统计探索性改善**，不支持普适“免费增强”结论。末段 ODE时间窗是合理候选，但尚未在当前目标域系统验证。

## 7. 从定理到下一次实验的最短距离

在 **同一真实GT latent、固定ε、同一z_t** 上计算 `correct/wrong/null` Writer三个场的：
- 分t预测误差与正确/错Writer gap；
- `\|Δv_W\|`、条件score差的代理（如果要算，使用上述同一桥的关系）；
- 按未见字、Writer、来源分组的变化；
- 生成同噪声 Writer-swap的方向/幅度/内容稳定性。

**预测若被推翻应怎样改理论**：如果中间t的Writer风险差不显著而生成Writer-swap表现很好，可能模型对分布的控制并不表现为单样本FM风险差；需要检查对照样本是否在支持分布内、尺度/归一化、评价器和有限采样误差。**不能把 Bayes 理想恒等式错用为有限网络单次样本的强制不等式。**

配套：[Writer可辨识性](02_WRITER_INFORMATION_AND_IDENTIFIABILITY.md)、[下一轮4090审计](../1031/05_NEXT_EXPERIMENTS.md)。
