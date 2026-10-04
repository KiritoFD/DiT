# 待决待验变量库：从混杂归因中解耦现代组件与单变量消融设计

> **文档代号**：`DOC-105-02`  
> **制定日期**：2026年10月  
> **状态**：规范与实验指引  
> **核心宗旨**：正本清源，纠偏过度归因，设计纯粹干净的单变量消融对照组。

---

## 1. 历史误区反思：混杂归因（Confounding Attribution）的教训

在历史阶段（v17 ~ v32 及早期分析）中，出现过一种流传较广的假说：
> *“现代组件（RMSNorm、SwiGLU、2D-RoPE、QK-Norm）是导致书法二值生成停滞的元凶。RMSNorm 不减均值保留了白底直流偏置，QK-Norm 抹杀了笔锋高频注意力，RoPE 破坏了绝对九宫格，SwiGLU 引起震荡。”*

**然而，严谨的科研审计表明，这一推论在逻辑上存在严重的混杂变量（Confounders）干扰：**

1. **确凿的实证反例：`v53 (100k)` 的突破**：
   在远端 4090 机器上刚刚跑完 10 万步满跑收官的 `v53-triple-tables-noskel`，创下了 4090 机器的历史最高纪录：
   - **Strict SSIM** = **`0.5875`**；
   - **Strict LPIPS** = **`0.3428`**；
   - 查阅其完全真实的配置文件 `v53_triple_tables_noskel.json`，其配置项赫然为：
     ```json
     "norm_type": "rms",
     "mlp_type": "swiglu",
     "qk_norm": 1,
     "rope": 1
     ```
   **`v53` 在 59M 的较小参数量下，全套开启了 RMSNorm、SwiGLU、QK-Norm 和 RoPE，却打破了此前所有老版基线的纪录！**
2. **混杂变量的真相**：
   历史模型（如 v19 ~ v31）表现不佳时，现代组件往往与**“12 层交叉注意力插桩（xattn）”**、**“高权重自然图像 DINOv2 REPA”**、**“单像素 VAE 骨架断裂”** 以及 **“参数量过小（S/2 36M）”** 等真正致命的缺陷（见 `DOC-105-01`）绑定在一起。将系统性失败全盘推给现代组件，属于未受控制的过度推论。

因此，**RMSNorm、SwiGLU、2D RoPE、QK-Norm 绝非死刑犯，而是处于“待决状态（Pending Status）”的高价值候选变量**。必须在严格控制其他所有变量的前提下，开展科学的单一变量消融（Ablation Study）。

---

## 2. 四大待决现代变量的理论假说与机制辨析

### 2.1 变量一：归一化算子 —— RMSNorm vs Classic LayerNorm

- **机制差异**：
  - **LayerNorm (LN)**：$y = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$（中心化 + 缩放）
  - **RMSNorm**：$y = \frac{x}{\sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}} \odot \gamma$（仅根据均方根做二范数缩放，不减去局部均值 $\mu$）
- **书法任务中的冲突假说**：
  - **假说 A（偏置积累论）**：书法图像背景为大面积纯白（像素值接近 $+1.0$）。在残差流中，如果不减去均值 $\mu$，白底信号的直流分量（DC Component）会逐层正向累积，导致深层特征漂移；
  - **假说 B（尺度不变性优势论）**：RMSNorm 省去了减均值计算，在现代大模型（LLaMA、Flux）中被证明能显著提升数值稳定性并降低显存/时间开销（Step 耗时降低约 3%~5%）。如果去噪网络的残差流本身是零均值速度场（Velocity Target），白底偏置并不会通过 RMSNorm 爆炸。
- **验证目的**：在统一 DiT-B 1024 架构下，单独测试更换为 RMSNorm 是否会引起背景噪点增多或 SSIM 变异。

---

### 2.2 变量二：注意力稳定化 —— QK-Norm (ON) vs Standard Attention (OFF)

- **机制差异**：
  - **Standard Attention**：$\text{Attn}(Q, K, V) = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$
  - **QK-Norm**：在计算点积之前，先对 $Q$ 和 $K$ 分别做 LayerNorm / RMSNorm，即 $Q' = \text{Norm}(Q), K' = \text{Norm}(K)$，使得点积的模长受到严格约束。
- **书法任务中的冲突假说**：
  - **假说 A（笔锋钝化论）**：书法的锋芒、折笔、顿挫依赖局部注意力权重的“极度尖锐化”（High Attention Entropy Contrast）。QK-Norm 约束了向量模长，强制压平了注意力分布的峰度（Kurtosis），使得笔锋边缘失去雕刻感；
  - **假说 B（训练防爆与精度保护论）**：在混合精度（FP16 / BF16）或长序列注意力中，未经约束的 $Q K^T$ 极易在个别 Token 处产生极大点积值（$> 80$），导致 Softmax 输出下溢或梯度消失。QK-Norm 彻底杜绝了注意力爆炸，有助于模型在 10 万步以上的极长周期训练中维持微观梯度的持续更新。
- **验证目的**：检验 QK-Norm 对严格测试集文字骨架边缘锐利度与收敛稳定性的实际净效应。

---

### 2.3 变量三：位置编码 —— 2D Axial RoPE vs MAE 2D Sin-Cos

- **机制差异**：
  - **2D Sin-Cos (加性)**：在输入 Patch 嵌入上直接相加绝对坐标的正余弦编码：$x \leftarrow x + P_{\text{abs}}$；
  - **2D Axial RoPE (乘性)**：在注意力计算前，分别在 $Q$ 和 $K$ 的垂直与水平通道上施加旋转矩阵变换：$q_m = R_{\Theta, m}^d q_m$，相对距离以角度差 $\Delta m = m - n$ 的形式体现在点积中。
- **书法任务中的冲突假说**：
  - **假说 A（九宫格锚定破损论）**：汉字书法不是普通图像，其字形结构受到极其严格的九宫格、米字格“绝对天地留白”约束。绝对位置编码直接告诉网络当前 Patch 是处于“画布的正中心中宫”还是“左上角天头”；而相对位置编码只能表达“我和邻近 Patch 的相对距离”，容易导致汉字整体间架发生漂移或尺度自发缩放；
  - **假说 B（相对平移不变性优势论）**：RoPE 具有天然的相对距离平移不变性，对于不同书法家在字形内部的偏旁穿插（如左窄右宽、上紧下松）具有更平滑的拓扑流形泛化能力。
- **验证目的**：对比在无外部骨架输入的纯真迹去噪任务中，绝对位置编码与 2D RoPE 的结构一致性（skel_iou）差异。

---

### 2.4 变量四：前馈网络 —— SwiGLU vs Classic GELU MLP

- **机制差异**：
  - **GELU MLP**：$\text{FFN}(x) = \text{Linear}_2\Big(\text{GELU}(\text{Linear}_1(x))\Big)$，参数量为 $8 D^2$；
  - **SwiGLU**：$\text{SwiGLU}(x) = \text{Linear}_3\Big(\text{Swish}(\text{Linear}_1(x)) \odot \text{Linear}_2(x)\Big)$；在保持参数量严格相等（$8 D^2$）的前提下，中间隐层维度缩减为 $h = \frac{8}{3} D$。
- **书法任务中的冲突假说**：
  - **假说 A（门控震荡论）**：双线性元素级乘法（$\odot$）引入了二次非线性，在早期扩散去噪的大梯度扰动下可能引起局部特征方差不稳定；
  - **假说 B（高维流形容量提升论）**：GLU（Gated Linear Unit）结构在所有主流语言与视觉大模型中均表现出对稀疏特征更强的选择性门控能力，能让网络在等参数前提下更高效地分离“笔画墨色”与“背景白纸”。
- **验证目的**：确认在同参数量（$8 D^2$）下，SwiGLU 是否带来感知质量（LPIPS）与结构 SSIM 的实质性净增益。

---

## 3. 标准单变量隔离消融实验设计协议 (Ablation Protocol)

为彻底阻断多变量交织污染，消融实验必须严格遵守以下控制协议：

### 3.1 固定的“基准环境锚点”（Common Baseline Anchor）
所有消融实验均统一锁定在目前已被证实最强大、无过拟合的主线架构上：
- **模型主干**：`DiT-2Cond-B1024/2`（Hidden 1024, Depth 12, Heads 8，Patch 2，~233M 参数量）；
- **扩散目标**：纯 4-Channel 流匹配（Logit-Normal, Heun ODE, 50 步）；
- **先验系统**：挂载并完全冻结 A 级正交三表（`assets/triple_tables_best_minimal/`）；
- **数据与批大小**：`train.csv`（26,002 样本），Batch Size = 128；
- **步数与学习率**：统一评测节点为 **`Step 40,000`** 与 **`Step 80,000`**，学习率 1e-4 Cosine。

### 3.2 待执行的 4 组干净单变量对比矩阵

| 实验代号 | 单一变更变量 (Only Variable) | 对照组设计 (Control) | 实验组设计 (Treatment) | 待观察核心指标 |
|---|---|---|---|---|
| **`ABL-NORM`** | 归一化算子 | `norm_type = layer` (当前 v61) | `norm_type = rms` | 背景纯净度、Strict SSIM、训练速度 |
| **`ABL-QKNORM`** | 注意力 QK-Norm | `qk_norm = 0` (当前 v61) | `qk_norm = 1` | 笔画边缘锐利度 (skel_iou)、注意力熵 |
| **`ABL-ROPE`** | 位置编码类型 | `rope = 0` (2D Sin-Cos, v61) | `rope = 1` (2D Axial RoPE) | 九宫格中宫偏移量、Strict SSIM |
| **`ABL-FFN`** | MLP 门控机制 | `mlp_type = gelu` (当前 v61) | `mlp_type = swiglu` | 相同参数量下的 LPIPS 质感与收敛斜率 |

---

## 4. 判定规则与准入标准

1. **正向准入标准（Adopt）**：
   - 在 Step 40,000 与 80,000 两个节点上，Strict SSIM 提升 $\ge +0.0030$ 且 LPIPS 改善 $\ge -0.0050$；
   - 训练步速不发生显著下降（$> 2.5 \text{ steps/s}$）；
2. **中性保留标准（Neutral）**：
   - 指标波动在 $\pm 0.0015$ 以内，若显存节省 $> 10\%$ 或速度提升 $> 5\%$，则作为工程加速构件准入；
3. **负面剔除标准（Reject）**：
   - Strict SSIM 发生确定性下跌 $\ge -0.0030$，或视觉上产生可辨认的笔画虚化、背景散斑。届时方可正式将其列入 `DOC-105-01` 架构死刑清单。
