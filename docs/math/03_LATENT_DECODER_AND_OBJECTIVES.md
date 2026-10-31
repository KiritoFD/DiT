# Latent、VAE Decoder、骨架与辅助监督：为什么“更容易优化”不一定“生成更好”

> 核心问题：像素/骨架的训练信号经过连续 VAE 与 U-Net 后是否保真？为什么4ch有时优于12ch、两阶段骨架难以晋级、解冻Decoder能改善LPIPS却不改善Writer？证据：[MoYi审计](../107/03_evidence_grade_and_errata.md)、[两阶段归档](../experiments/2026-10-03-twostage-archive.md)、[VAE报告](../experiments/2026-10-13-vae-and-related-changes-report.md)、[S/J实验](../1028/CONCLUSION_AND_NEXT_PHASE_20261028.md)。

## 1. 先分清三个对象

真实图像 `X`，VAE latent `Z_0`，给定STD/Writer生成器的输出为 `\hat Z_0`：

\[
Z_0=E_\eta(X),\qquad
\hat Z_0\sim p_\theta(\,\cdot\mid G,S,W),\qquad
\hat X=D_\psi(\hat Z_0).
\]

实际 latent 有缩放/平移/后验采样等合同；对 SD-VAE 常见 `scale=0.18215`，但不适用于所有历史 Calli-VAE。严格比较时一定要让 `E_\eta,D_\psi`、scale/shift、posterior sample或mode的选择与checkpoint对应；否则连目标 `p_\theta(Z_0\mid cond)` 都变了。

### VAE重建误差与生成误差不是同一个量

设 VAE重建 `X_\mathrm{rec}=D_\psi(E_\eta(X))`、真正生成 `\hat X=D_\psi(\hat Z_0)`。若Decoder在局部是 `L_D`-Lipschitz，

\[
\|\hat X-X\|\le
L_D\|\hat Z_0-E_\eta(X)\|+
\|D_\psi(E_\eta(X))-X\|.
\]

这是三角不等式下的一种**上界**，不是SSIM、LPIPS、InkIoU的可加分解：
- 第二项：VAE自身重建瓶颈；
- 第一项：生成latent是否真的接近有效真迹latent，并受Decoder局部放大增益影响。

VAE重建更好只减少第二项的某类上界；它可能把数据latent变成**更难学习的分布**，不能推出下游网络必然同幅度提高。项目 v79物理 VAE 自重建 PSNR **30.56→34.04dB**、高频保持率报告 **.901**；但不同下游训练的SSIM并没有固定等量增加，证实**上游重建收益不自动转移为生成收益**。GAN独立贡献未做无GAN严格消融。

## 2. Decoder微调为何能改善画质，却未必改善Writer条件

对固定latent `z`，若仅改变Decoder参数 `\psi→\psi+\delta\psi`，

\[
D_{\psi+\delta\psi}(z)-D_\psi(z)
\approx J_\psi(z)\,\delta\psi.
\]

对latent本身变化则：

\[
D_\psi(z+\delta z)-D_\psi(z)\approx J_z(z)\,\delta z.
\]

这说明**修改Decoder权重**与**修改生成器给出的latent**在理论上是两个独立控制来源。局部线性公式不保证哪一方只管高频；Decoder完全可能改变墨块、笔宽、连通性甚至OCR类别。因此“Decoder只处理纹理”最多是当前实验的**经验假说**。

J0–J3从相同 Deep-Sym4@85k EMA起点额外训练5k，数据：

| arm | Decoder/图像监督 | SSIM↑ | LPIPS↓ | IoU↑ | Writer Micro Top1↑ |
|---|---|---:|---:|---:|---:|
| J0 | FM / 冻结Decoder | .6149 | .3220 | .3437 | 64.17% |
| J1 | 图像Loss / 冻结Decoder | .6149 | .3222 | .3428 | 60.96% |
| **J2** | 图像Loss / 解冻Decoder / GT重建 | **.6198** | .3150 | **.3447** | 63.64% |
| J3 | 图像Loss / 解冻Decoder / 无GT重建 | .6193 | **.3147** | .3433 | 62.03% |

J2−J0 SSIM **+.0049**，配对区间**[+.0039,+.0060]**；独立Writer未同步提升。**因果限制**：J2可能同时更新U-Net和Decoder，仅看两个端点不能把全部收益算给Decoder。需在**同一个** `z` 上交换 `D_{\psi_0},D_{\psi_1}` 直接比较，再做完整U-Net/Decoder四种交叉组合；不能把近饱和的 J5–J7 OCR96.15%不变当作“Decoder绝不伤字符”的定理。

### 图像Loss冻结Decoder的梯度去哪了

若 `L_\mathrm{img}(D_\psi(\hat z_\theta),x)`，则：

\[
\nabla_\theta L_\mathrm{img}
=J_\theta(\hat z)^\top J_z(D_\psi)^\top
\nabla_{\hat x}L_\mathrm{img}.
\]

即使Decoder不更新，图像损失也会经其Jacobian回传给 U-Net；但信号在当前latent、尺度、梯度裁剪下可能不利或很弱。J1≈J0并不能证明图像Loss“一般无效”，只能说明**这条被冻结Decoder的实现和5k预算没产生可验证的生成提升**。

## 3. 12ch辅助目标为什么会和4ch主目标竞争

记 `L_\mathrm{img}` 为生成图像latent的训练损失，`L_\mathrm{aux}` 为Canny/骨架额外输出通道的训练损失。等权12ch模型等价于某个 `\lambda` 取值下：

\[
L_\mathrm{total}=L_\mathrm{img}+\lambda L_\mathrm{aux}.
\]

对共享参数 `\theta`，令 `g_i=\nabla_\theta L_\mathrm{img}`、`g_a=\nabla_\theta L_\mathrm{aux}`。做小步梯度下降，图像目标一阶变化：

\[
\Delta L_\mathrm{img}\approx
-\eta\left(\|g_i\|^2+\lambda\langle g_i,g_a\rangle\right).
\]

当内积为负且幅度足够大时，联合总目标可以下降而**图像目标短期上升**。即使内积为正，12ch也会消耗网络容量和前向计算，并改变输出头与梯度尺度。

项目测得等权12ch在某层最终梯度中辅助部分占比约 **51%**、早期 patch embed辅助梯度约为图像的 **2.6倍**。同50k历史MoYi Strict：12ch SSIM **.5935**、LPIPS **.3724**，4ch **.5949/.3678**。**严格结论是“当前等预算12ch无胜出，存在梯度竞争风险”**，并非“辅助预测结构绝对没有价值”。旧12ch还发生过CFG只引导图像通道/白底归零解码合同错误，需要先区分工程问题与数学原因。

**可证伪测试**：在同一batch/初始化上测 `\rho=\langle g_i,g_a\rangle/(\|g_i\|\|g_a\|)`、比例/每通道梯度、主图像Loss与真实内容错误；控制模型参数、训练/推理NFE、VAE、噪声、相同训练总曝光后再比较有无辅助目标。没有这些数据就不要由最终指标反推梯度一定为负。

### 结构条件与结构标签不是同义词

STD图 `G` 是预测时已知输入；GT骨架 `K=\mathrm{skeleton}(X)` 是训练标签。即使 `G` 已包含字形粗拓扑，也**不意味着从GT提取的K在统计意义上完全冗余**——后者可能包含Writer化结字与真实运笔信息。当前问题在于：将K作为额外模型输出，需要解决标签稀疏、VAE解码和两任务竞争，而且训练期提供GT K却推理时无法得到会产生分布偏差。

不能简单说“骨架天然无用”；应说**现有辅助目标和两阶段接口未交付可用的净收益**。

## 4. 为什么单独学SkelNet会遭遇误差级联

两阶段路线可以抽象为：

\[
\hat K=f_\alpha(G,W,S,\epsilon_1),\qquad
\hat X=f_\beta(\hat K,G,W,S,\epsilon_2),
\]

而训练第二阶段若使用真实 `K`，训练看到 `K\sim p(K\mid G,W,S)`，部署却只收到 `\hat K\sim p_\alpha(K\mid G,W,S)`。即便第二阶段在 GT 骨架条件下做得很好，也不保证 `\hat K` 的错误分布下表现相同。

若 `f_\beta` 对K是 `L_\beta`-Lipschitz，可得简单界：

\[
\|f_\beta(\hat K)-f_\beta(K)\|\le L_\beta\|\hat K-K\|.
\]

但是细骨架的**拓扑连通性不随像素范数连续变化**：1像素接触会改变连通块，丢一条细线可能使大面积SSIM仍好看。因此 latent MSE 或骨架SSIM下降不足以证明第二阶段可以利用。

项目 v45 骨架SSIM **.6484** 高于无模型基线 **.6410**，但骨架IoU **.0870** 低于 **.1044**；这是**中间结构指标误导**的直接反例。另有1px线经连续VAE丢失比例高，3px几何在VAE往返更稳。v46尝试直接纯STD，约35k得到SSIM **.5516**、LPIPS **.3825**，优于同期“50%GT骨架混合”旧面板，但训练步数/配置仍有差异，不能做超出数据的强归因。

**架构含义**：如果坚持两阶段，必须给第二阶段输入**预测骨架的真实误差分布**，并验收拓扑可识别性；本研究预算下更简单可靠的是 STD条件直接进入生成器，减少显式误差接口，而不是堆更多SkelNet层。

## 5. 放大图片相似度为什么会“奖励坏书法”

单GT `x_{\rm GT}` 的 SSIM/LPIPS会惩罚与此GT不同、但依然合法的Writer变体；像素目标也可能更偏好白背景占比高、细节低方差的平均输出。

例如对平方误差，若 `Y` 是给定Writer/字形的合法随机作品，则 MSE最优确定性预测是 `\mathbb E[Y\mid G,S,W]`，而不是某个锐利样本；它可能在笔画边界产生模糊。**SSIM不是平方误差，其最优解不能直接由这个定理得出。**

S1的 Frag较低却伴随笔宽过粗（1.234倍GT）；在某些早期干预中替换错误char table竟短时提高SSIM却欠墨，进一步说明“分数高”需要 OCR/墨量/Writer 同步审查。

## 6. 后续工程与研究结论

1. **保留纯图像latent主输出和标准字空间条件为简单基线**，不要为了提供“更多监督”先扩大12ch或再堆显式SkelNet。
2. **优先用Decoder后训练追细笔/边缘质量**，但必须固定latent做交叉解码，不能由LPIPS增益推出书家身份正确。
3. **VAE重建提升不代表生成器容易学**；下游应同时测 real GT latent覆盖、生成latent分布、decoder Jacobian与内容结构。
4. 只有出现可复现的主任务梯度/结构净收益时，才再考虑额外骨架通道或条件Decoder；否则新算力投向 Top10有效曝光和严格OOV。

配套：[1031实验数据表](../1031/01_EVIDENCE_MATRIX.md)、[Decoder训练方案](../1031/03_TRAINING_TRANSFER_AND_DECODER.md)、[下一步](../1031/05_NEXT_EXPERIMENTS.md)。
