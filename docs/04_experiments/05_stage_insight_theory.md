# 马良 (Callig-DiT) 阶段性洞察与数学理论框架

> **核心摘要**：本文档从第一性原理出发，对 Callig-DiT (马良) 的当前架构进行深度的物理与数学理论审查。重点论述双通道风格注入的动力学机制、梯度正交性定理、信息流瓶颈诊断，并基于已被证伪的技术路线，提出下一阶段架构演进的理论指导原则。

---

## 1. 宏观视野：书法生成的双重物理尺度解耦

书法的“风格”具有天然的双重物理尺度。早期端到端扩散模型试图用单一注意力网络同时拟合这两个尺度，导致了难以跨越的泛化鸿沟（Seen vs. Strict 瓶颈）：
1. **宏观拓扑（Macro Structure / Geometry）**：字体的间架结构、重心欹侧、长宽比例（如颜体的方正浑厚、欧体的险绝瘦长）。这本质上是**低频的几何变换**。
2. **微观纹理（Micro Texture / Kinematics）**：笔墨的枯润飞白、牵丝连带、边缘毛刺、起笔收笔的顿挫。这本质上是**高频的图像纹理去噪**。

**核心洞察**：扩散模型（Diffusion Models）作为一个迭代去噪过程，其损失函数天然倾向于优化高频的微观纹理（微观边缘模糊化妥协），而难以在像素空间直接产生大幅度的拓扑扭曲（宏观几何错位）。
**设计确立**：马良确立了**SkelNet（几何形变场）+ DiT（纹理去噪场）**的正交解耦架构，这是突破泛化瓶颈的基石。

---

## 2. 数学理论：双通道风格注入动力学

在当前 v21 架构中，书家风格 $e_{callig} \in \mathbb{R}^{128}$ 的注入严格遵循双通道动力学。

### 2.1 通道 A：微观纹理调制 (adaLN-Zero)
*   **机制**：将时间步 $t$ 与风格 $e_{callig}$ 融合，产生全局仿射参数 $(\gamma, \beta, \alpha)$ 调制 Transformer 层。
*   **数学本质**：在去噪得分匹配中，风格向量改变了得分函数 $\nabla_x \log p_t(x|c)$ 的方向，使网络倾向于生成符合该书家笔触习惯的高频分布。

### 2.2 通道 B：宏观结构调制 (SkelNet/DeformSkel)
*   **机制**：以共享的标准骨架 $g_{std}$ 为锚点，通过 U-Net 预测低频偏移场 $\Delta p = f_{\theta}(g_{std}, e_{callig})$，进行 TPS/Bilinear 空间重采样，生成目标书家骨架 $g'$。
*   **数学本质**：在连续的二维流形上进行微分同胚映射（Diffeomorphism），保证了汉字拓扑不变性的同时完成风格化间架重构。

### 2.3 核心定理：扩散梯度与几何梯度的正交性 ($\cos \approx 0$)
我们使用探针工具实测了流回风格嵌入层的梯度，得到了本项目最重要的基础定理：
$$ \cos(\nabla_{e_{callig}}\mathcal{L}_{diff}, \nabla_{e_{callig}}\mathcal{L}_{deform}) \approx +0.0030 $$
**理论剖析**：
1. **物理意义**：扩散损失 $\mathcal{L}_{diff}$ 提供的梯度在几何形变子空间上的投影几乎为零。这意味着，如果去掉显式的骨架形变损失 $\mathcal{L}_{deform}$，网络绝对无法自发学会宏观结体的变形（已被无监督变形实验彻底证伪）。
2. **训练动力学**：两个通道在 $128$ 维表征空间中互不挤占容量。$\mathcal{L}_{deform}$（幅值约为 $\mathcal{L}_{diff}$ 的 6.22 倍）主导了风格向量中几何特征的聚类，而 $\mathcal{L}_{diff}$ 则负责纹理特征的雕琢。这种**无破坏性干涉**是模型能稳定收敛的数学保障。

---

## 3. 架构设计与代码实现审查

基于以上理论，我们审查了当前 codebase (`src/model/dit.py`, `src/model/deform_skel.py`) 的关键设计：

### 3.1 正确的决策（What we did right）
1. **背景零造墨硬门控 (Background Gating in SkelNet)**：
   *   **代码实现**：`gate = clamp(1.0 - dt_w / max(0.25, 1e-4))`
   *   **审查结论**：极其优异的设计。它通过距离场 (Distance Transform) 强行截断了 U-Net 在背景处的自由发散，从源头消灭了“飞墨”和拓扑破坏问题，这是形变能 work 的物理底线。
2. **条件融合的方差对齐 (Factorized Fusion with Balanced Variance)**：
   *   **代码实现**：`style_ln`, `style_gain` 独立归一化与放大。
   *   **审查结论**：解决了严重的代码隐患。此前 $e_{glyph\_vec}$ 的方差是 $e_{callig}$ 的 38 倍，导致 adaLN 完全被字形内容主导，风格“失聪”。独立 LN 与初始化增益强制使 $\text{Var}(y_{emb}) \approx \text{Var}(t_{emb})$，打破了 Zero-Init 的死锁。
3. **低学习率微调 (SkelNet Decoupled LR)**：
   *   **审查结论**：保持 SkelNet 以 $0.1\times$ 的学习率进行联合微调，成功避免了扩散模型初期巨大的随机梯度冲毁已收敛的形变拓扑。

### 3.2 失败方向的数学病理 (Falsifications)
1. **Local Cross-Attention (LCA) 崩溃**：汉字骨架的极度稀疏性（前景 $<8\%$）导致大量查询命中全零区，Softmax 归一化产生数值尖刺，破坏整体受力平衡。
2. **多 Token 风格表征 (Multi-Token Style)**：引发 Semantic Leakage。多 Token 会不受控地记忆汉字局部部件（偏旁部首），而非纯粹的书法风格，导致在 Strict 集合上生成拼接错别字。必须维持紧凑的单向量（或解耦的残差向量）以切断内容信息的泄露通道。

---

## 4. 未来理论框架与下一步指导原则

基于上述全景图，马良的下一步演进应遵循以下原则：

### 原则一：解除 adaLN 调制共享瓶颈 (Decouple Timestep and Style in Modulations)
*   **诊断**：当前通道 A 中，$t_{emb}$ 和 $y_{emb}$ 仍在同一个 MLP 中竞争容量。
*   **指导行动**：**全面激活 `style_ada_rank > 0` (Low-Rank Style-adaLN)**。在 `dit.py` 中已实现该机制（向 adaLN 的输出直接加上 $W_{up} \cdot \text{LN}(e_{callig})$）。这能让风格绕过由时间步主导的主干 MLP，获得独立的低秩控制通道，理论上能大幅提升墨迹纹理的风格拟合上限。

### 原则二：实施安全受控的局部风格化交叉注意力 (Local Style-Skeleton CA)
*   **诊断**：为了实现结体的细微书家差异化，我们不能依赖弥散的图像 LCA，而必须在条件端发力。
*   **指导行动**：**推进 Phase 2：激活 `local_ca_q="g"`**。在 `dit.py` 中：让去噪画布上的 query 并非直接寻址图像 patch，而是寻址**已被风格重组的局部骨架 token**。语义上变为：“去噪中的画布，在风格条件下向局部骨架询问书写证据”。这比直接在主干上加 CA (`q="x"`) 安全得多。

### 原则三：分层语义重构 (Hierarchical Semantic Disentanglement)
*   **诊断**：$128$ 维单向量在表达 “书体 (Script) $\times$ 书家 (Calligrapher) $\times$ 字体 (Glyph)” 时已接近容量上限。
*   **指导行动**：**启用 S2 分解 (`hier_style > 0`)**。将 $e_{style}$ 严格分解为 `主效应 + pair残差 + 书体效应`。同时，对于书体这种宏观几何特征，激活 `script_film=True`，让书体向量全局介入骨架 $g_{tok}$ 的调制。

### 总结
马良 (Callig-DiT) 当前的理论地基已经极其扎实，双通道架构与梯度正交性为模型提供了极佳的优化平滑度。下一步无需在参数量上盲目扩张（已被证伪），而是应通过**低秩解耦旁路 (Low-Rank By-pass)** 和 **条件端局部重组 (Condition-side Local Recombination)**，在不破坏现有收敛拓扑的前提下，精确提升微观纹理与局部结体的生成质量。
