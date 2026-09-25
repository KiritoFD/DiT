# DiT-2Cond 主干模型与现代网络算子

## 1. 模型规约与超参数 (DiT-2Cond-S/2)

Callig-DiT 采用现代轻量高效的 Transformer 架构，专为 $256 \times 256$ 图像分辨率的潜空间生成设计：

| 参数名称 | 设定值 | 架构设计考量 |
| :--- | :--- | :--- |
| **Model Type** | `DiT-2Cond-S/2` | Patch 大小 $2 \times 2$，在 $32 \times 32$ 潜空间展开为 $16 \times 16 = 256$ 个空间 Token |
| **Hidden Dim ($d$)** | $384$ | 平衡表征容量与训练吞吐，稳态显存控制在 18.6GB |
| **Depth ($L$)** | $12$ | 12 层标准 Transformer 块，第 8 层外接 REPA 自监督表征损失 |
| **Attention Heads ($H$)** | $6$ | 单头维度 $d_k = 64$ |
| **Positional Embedding** | **2D RoPE** ($\theta=100.0$) | 完全废弃绝对正弦位置编码，保证汉字局部空间几何平移与缩放不变性 |
| **Normalization** | **RMSNorm** | 替代 LayerNorm，消除均值计算开销，提升 FP16/BF16 混合精度数值稳定性 |
| **Feed-Forward Network** | **SwiGLU** | 隐藏层扩展倍率经过精心调整，提供远优于传统 GELU MLP 的非线性拟合能力 |
| **Attention Kernel** | `sdpa` (FlashAttention-2) | 结合 PyTorch 2.0+ 内核加速，无须显存物化注意力矩阵 |

---

## 2. 2D 旋转位置编码 (RoPE for 2D Spatial Grids)

传统的 1D 或 2D 绝对位置编码在汉字结构的细微笔画平移下无法保持内积的相对不变性。马良架构引入二维轴对齐旋转位置编码：

对于处于二维潜空间坐标 $(i, j)$ 的 Token，其查询向量 $q \in \mathbb{R}^{d_k}$ 分解为两组独立的子空间：
$$
q_{(i, j)} = \begin{bmatrix} R_{\Theta, i} q^{(1)} \\ R_{\Theta, j} q^{(2)} \end{bmatrix}
$$
其中 $R_{\Theta, i}$ 为标准的 2D Givens 旋转矩阵，基频参数设为 $\theta = 100.0$。在注意力内积计算中，两点之间的相对注意权重仅取决于空间相对位移 $(\Delta i, \Delta j)$：
$$
\langle q_{(i_1, j_1)}, k_{(i_2, j_2)} \rangle = f(x_{i_1, j_1}, x_{i_2, j_2}, i_1 - i_2, j_1 - j_2)
$$
实验证明，RoPE 彻底消除了笔画边缘的网格周期伪影（Grid Artifacts）。

---

## 3. 条件调制机制：adaLN-Zero 与 Cross-Condition Fusion

在 DiT 架构中，条件信息的注入决定了去噪方向的收敛效率。

### 3.1 调制参数生成
时间步 $t \in [0, 1]$ 经由频率嵌入和 2 层 SiLU MLP 映射为时间向量 $e_t \in \mathbb{R}^{d}$；风格 ID 经由查找表与线性投影映射为风格向量 $e_{style} \in \mathbb{R}^{d_{callig}}$（$d_{callig}=128$）。

通过条件融合层（`factorized_cat`），将两组向量拼接后经由线性层输出每个 Block 的 6 组仿射参数：
$$
[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = \text{Linear}(\text{SiLU}(\text{Linear}([e_t, e_{style}])))
$$

### 3.2 adaLN-Zero 稳态更新
在模型初始化阶段，残差连接缩放参数 $\alpha_1, \alpha_2$ 的投影权重全部初始化为零（Zero Initialization）：
$$
x_{l}' = x_{l} + \alpha_{1, l} \odot \text{SelfAttn}\left((1 + \gamma_{1, l}) \odot \text{RMSNorm}(x_{l}) + \beta_{1, l}\right)
$$
$$
x_{l+1} = x_{l}' + \alpha_{2, l} \odot \text{SwiGLU}\left((1 + \gamma_{2, l}) \odot \text{RMSNorm}(x_{l}') + \beta_{2, l}\right)
$$
在训练初始阶段，各 Transformer 块退化为恒等映射，梯度无阻碍流向底层的骨架与输入层，彻底规避了深度扩散模型的训练初期发散。

---

## 4. 骨架条件注入：Glyph Embedder

形变骨架 $S_{def}$ 并非直接与加噪潜变量 $x_t$ 在通道维度拼接（通道拼接被实验证明会导致模型强行忽略风格信号），而是通过专用的 **Glyph Embedder**：
1. 骨架经过独立的 Patchify 投影为骨架 Token 序列 $T_{skel}$；
2. 经过 2 层专用的 Transformer 块进行拓扑自注意力提炼；
3. 在 DiT 主干的前 4 层（`glyph_inject_layers=4`），通过自适应层归一化（adaLN）逐层注入结构先验；
4. 初始化注入强度设定为 $\alpha=0.6$。后 8 层释放注意力容量，专注于运笔纹理、枯润飞白的书法风格拟合。
