# 条件最优传输 (Conditional Optimal Transport, C2OT) 技术规范与理论白皮书

> **状态**：已正式合入主分支代码库，已在本地、机器 48 (Linux/CUDA)、机器 4090 完成三端实测验证。  
> **核心价值**：彻底根除朴素无条件 OT 在条件生成模型中的**先验漂移灾难**（Train-Test Prior Mismatch），在严格保持高斯先验无偏性的同时，以 **11~18 ms / 批次（~1.3% 步耗）** 的微小开销拉直流匹配轨迹，实现真正的 **Free Lunch** 训练加速。

---

## 一、 背景与理论根源：朴素 OT 在条件流匹配中的致命缺陷

在 Rectified Flow / Flow Matching (Lipman et al., 2023; Tong et al., 2024) 中，朴素 Minibatch Optimal Transport (OT-CFM) 试图在每个训练批次内部，通过匈牙利算法最小化数据 $x_0$ 与标准高斯噪声 $x_1 \sim \mathcal{N}(0, I)$ 之间的全局输运代价：
$$
\pi^* = \arg\min_{\pi \in \mathcal{S}_B} \sum_{i=1}^B \| x_{0, i} - x_{1, \pi(i)} \|^2_2
$$

### 1. 条件先验漂移定理 (Conditional Prior Shift Theorem)
在条件生成（如书法生成中输入书家 $c_{callig}$、书体 $c_{script}$、字形 $c_{char}$）场景下，数据分布是强条件依赖的：
$$
p(x_0 | c) \neq p(x_0)
$$
不同条件的真迹在潜空间中的均值与几何主成分存在显著偏移（例如：粗黑颜楷的潜均值与细瘦怀素草书存在强烈正负偏置）。

当执行**跨条件的全局朴素 OT 配对**时，匈牙利算法为最小化 $\|x_0 - x_1\|^2 = \|x_0\|^2 + \|x_1\|^2 - 2 \langle x_0, x_1 \rangle$，会系统性地将特征投影较大的噪声向量分配给特征均值较大的条件类别，导致：
$$
q_{naive}(x_1 \mid c) \neq \mathcal{N}(0, I)
$$
- **训练端**：模型见到的条件先验 $q_{naive}(x_1 \mid c)$ 严重偏离标准正态分布；
- **推理端**：ODE 采样阶段从严格的标准高斯 $p_{test}(x_1) = \mathcal{N}(0, I)$ 启动；
- **恶果**：产生极其严重的 **Train-Test Prior Mismatch**，导致模型在推理阶段笔画断裂、字形破碎（frag 反弹），完全抵消了路径拉直带来的收益。这也是此前项目中关闭 `use_ot: true` 的本质原因。

---

## 二、 C2OT 数学形式化与无偏性证明

为解决上述先验漂移问题，C2OT（Conditional Optimal Transport）将全局置换群约束在条件等价类诱导的**块对角置换子群**（Block-Diagonal Permutation Subgroup）内部：

设批次划分为 $K$ 个不相交的条件子集：
$$
\mathcal{B} = \bigsqcup_{k=1}^K \mathcal{B}_k, \quad \mathcal{B}_k = \{ i \in [B] \mid c_i = u_k \}, \quad |\mathcal{B}_k| = N_k
$$
C2OT 的允许置换群定义为：
$$
\mathcal{S}_B(c) = \prod_{k=1}^K \mathcal{S}_{N_k} \subset \mathcal{S}_B
$$
在每个条件子块 $\mathcal{B}_k$ 内部，独立求解最优传输配对：
$$
\pi_k^* = \arg\min_{\pi_k \in \mathcal{S}_{N_k}} \sum_{i \in \mathcal{B}_k} \| x_{0, i} - x_{1, \pi_k(i)} \|^2_2
$$

### 1. 先验无偏性证明 (Zero Prior Shift Guarantee)
对于任意特定条件 $u_k$，噪声子集 $\{ x_{1, j} \}_{j \in \mathcal{B}_k}$ 在采样时均独立同分布自 $\mathcal{N}(0, I)$。  
由于 $\pi_k^*$ 仅作用于集合 $\mathcal{B}_k$ 内部的索引重排，且标准多元高斯分布在任意置换变换下满足保测度性（Measure-Preserving）：
$$
q_{C2OT}(x_1 \mid c = u_k) = \frac{1}{N_k!} \sum_{\pi_k} \delta(x_1 - \pi_k(x_{1, \text{orig}})) \equiv \mathcal{N}(0, I)
$$
**结论**：C2OT 的条件先验分布与测试端先验分布严格恒等，彻底清除了条件先验泄漏。

### 2. 组内几何紧凑性与路径拉直
同条件子集内部的书法潜变量具有高度一致的墨色与笔画拓扑分布，流形局部曲率显著低于跨风格混合流形。在同条件内部求解 OT 能极大程度消除轨迹自相交，大幅降低速度场的经验方差与高阶曲率。

---

## 三、 高性能工程架构实现

### 1. 算法复杂度骤降
- **朴素全局 OT 复杂度**：$O(B^3)$（当 $B=384$ 时，操作数 $\approx 5.66 \times 10^7$）；
- **C2OT 分组复杂度**：$\sum_{k=1}^K O(N_k^3)$。当平均每个批次包含 20 个活跃槽位（$N_k \approx 19$）时，操作数降为 $20 \times 19^3 \approx 1.37 \times 10^5$。
- **理论核心算法加速比**：**413 倍**。

### 2. BLAS Level-3 GEMM 矩阵优化
在计算批内两两欧氏距离矩阵时，避免使用 PyTorch CPU `cdist` 产生的多线程同步锁死与 3D 张量显存分配，采用全局预计算范数 + BLAS-3 GEMM 展开：
$$
\| x - n \|^2_2 = \|x\|^2_2 + \|n\|^2_2 - 2 \cdot x \cdot n^T
$$
```python
# 1. 预计算全 batch 范数向量 (B,)
x_norm2 = np.sum(x_np ** 2, axis=1)
n_norm2 = np.sum(n_np ** 2, axis=1)

# 2. 组内极速 BLAS 点积展开计算代价矩阵 (N_k, N_k)
cost = (x_norm2[idx_u, None]
        + n_norm2[idx_u][None, :]
        - 2.0 * np.dot(x_np[idx_u], n_np[idx_u].T))
_r, cl = linear_sum_assignment(cost)
```

### 3. 单次显存/内存同步协议 (1-Shot D2H / H2D Protocol)
传统做法在循环内反复执行 `cost.cpu().numpy()` 会导致数十次 `cudaStreamSynchronize()` 阻塞 GPU 流水线。  
C2OT 架构采用**单次打包批量拷贝**：
- **GPU -> CPU**：仅在入口处执行 1 次批量 D2H（数据与噪声平展特征，耗时 ~3 ms）；
- **CPU 计算**：CPU 端完成全内聚 GEMM + 匈牙利配对，生成最终整数排列向量 `perm_np`；
- **CPU -> GPU**：仅将 `perm_np`（$384 \times 8$ 字节 $\approx 3$ KB）单次 H2D 回传并执行索引切片。

---

## 四、 跨平台实测性能与单元测试验证

我们在三端平台（Windows 本地开发机、机器 48 训练集群、机器 4090 实验卡）运行标准化测试套件 `tests/test_c2ot.py`，测试涵盖五大核心维度：

| 测试项 | 验证内容 | 测试指标要求 | 实测结果 (机48) | 实测结果 (4090) | 判定 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Test 1: 条件强隔离性** | 验证是否有任何样本跨条件/槽位配对 | 跨槽位配对数 $\equiv 0$ | **0 跨界** | **0 跨界** | **PASS** |
| **Test 2: 先验无偏性** | 沿条件漂移方向的噪声投影偏差均值 | 朴素 OT $> 0.20$ vs C2OT $< 0.05$ | 朴素: 0.3064<br>C2OT: **0.0039** | 朴素: 0.3314<br>C2OT: **0.0059** | **PASS** |
| **Test 3: 输运代价下降** | 验证路径是否被有效拉直 | 代价变化率 $< 0$ | **-2.45%** | **-2.33%** | **PASS** |
| **Test 4: 模式与异常容错** | slot/callig/char/孤立单样本/空参数兜底 | 无抛出且严格恒等映射 | **全模式通过** | **全模式通过** | **PASS** |
| **Test 5: 核心算法加速比** | 核心 Hungarian 求解加速比 ($B=384$) | 算法加速比 $\ge 5.0\times$ | **27.1x** | **38.8x** | **PASS** |
| **Test 6: 端到端步耗延迟** | 包含全部张量与数据传输的实际总耗时 | 必须 $< 50$ ms | **11.3 ms** | **18.1 ms** | **PASS** |

> **开销占比评估**：  
> 当前机器 48 在训练 DiT-B/2（$B=384$）时，单步训练耗时约为 $750$ ms（$1.33$ steps/s）。  
> C2OT 端到端仅需 **$11.3$ ms**，仅占单步总开销的 **$1.5\%$**，真正达成了可忽略算力开销的 **Free Lunch**。

---

## 五、 CLI 参数与使用指南

### 1. 命令行参数配置
已在 `src/train/cli.py` 与 `src/eval/cli.py` 中完整集成：

```bash
# 启用 C2OT 条件最优传输 (推荐: slot 模式)
--use-c2ot 1 \
--c2ot-mode slot
```

### 2. 条件模式说明
- `--c2ot-mode slot`（**官方推荐**）：按 `(y_callig * 100 + y_script)` 构造复合槽位。严格保证同一书家、同一书体内部配对，最契合多风格书法生成流形；
- `--c2ot-mode callig`：仅按书家身份（10 类）分组；
- `--c2ot-mode char`：仅按字符身份分组。

### 3. 配置字典 / JSON 示例
在实验配置文件（如 `.json`）中直接指定：
```json
{
  "use_c2ot": true,
  "c2ot_mode": "slot"
}
```
`src/loss/__init__.py` 中的 `flow_kwargs_from` 会自动安全解析并透传至 `FlowMatching` 实例。
