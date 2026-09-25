# 最优传输流匹配理论与常微分方程数值求解器

## 1. 为什么选择流匹配 (Flow Matching vs DDPM)

在传统基于离散时间步高斯扩散模型（DDPM / DDIM）中，去噪路径是一条高曲率的随机布朗运动曲线，需要 250 到 1000 步离散反向扩散才能得到清晰样本。而在汉字生成中，高曲率路径极易在早期步中偏离结构吸引子，造成不可挽回的笔画错乱。

马良采用**最优传输条件流匹配（Optimal Transport Conditional Flow Matching, OT-CFM）**：
- **定义**：将标准正态先验分布 $p_0 = \mathcal{N}(0, \mathbf{I})$ 映射到真实数据分布 $p_1$；
- **直线插值轨迹**：在潜空间中构造确定性的直线插值流：
  $$
  x_t = (1 - t)x_0 + t x_1, \quad t \in [0, 1]
  $$
- **目标速度场（Ground-Truth Velocity Field）**：
  $$
  u_t(x_t | x_0, x_1) = \frac{d x_t}{d t} = x_1 - x_0
  $$
  速度场在全时间区间恒定为常向量，轨迹曲率恒等于零。这使得数值常微分方程（ODE）求解器仅需 20 - 50 步即可极速、无漂移地积分出高保真图像。

---

## 2. 时间采样分布：Logit-Normal 密度聚焦

标准的均匀时间分布 $t \sim \mathcal{U}[0, 1]$ 赋予所有时间步相同的损失权重。但在实际生成中：
- 当 $t \to 0$ 时，样本充满高斯噪声，网络只能猜想模糊的全局色块；
- 当 $t \to 1$ 时，信号占主导，网络仅调整微小高频细节；
- **中段区间 $t \in [0.3, 0.7]$ 是字形拓扑骨架决断的最关键阶段（Decisive Phase）**。

马良采用 **Logit-Normal 时间分布采样器**：
$$
z \sim \mathcal{N}(\mu, \sigma^2), \quad t = \sigma(z) = \frac{1}{1 + e^{-z}}
$$
配置参数设为 $\mu = 0.0, \sigma = 1.0$。该概率密度函数在 $t=0.5$ 处呈现光滑单峰，将 68% 以上的梯度回传算力精准聚焦在中段拓扑决断区间，大幅提升了汉字笔画的闭合率与交界清晰度。

---

## 3. 数值求解器与推理采样

在推理与内存内评估（`in_mem_eval`）中，模型支持两种经典的高阶数值求解器：

### 3.1 一阶欧拉积分器 (Euler Solver, 20 Steps)
$$
x_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)
$$
适用于快速预览和在线指标吞吐监控，单字生成耗时仅需 $\approx 15\text{ ms}$。

### 3.2 二阶休恩预估-校正求解器 (Heun Predictor-Corrector, 50 Steps)
$$
\tilde{x}_{t + \Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t, c)
$$
$$
x_{t + \Delta t} = x_t + \frac{\Delta t}{2} \left[ v_\theta(x_t, t, c) + v_\theta(\tilde{x}_{t + \Delta t}, t + \Delta t, c) \right]
$$
Heun 求解器有效抵消了一阶截断误差，在严格零样本测试集（strict set）上使 SSIM 稳定提升 $+0.008$ 到 $+0.012$，飞白与连笔过渡极为流畅。

### 3.3 无分类器引导 (Classifier-Free Guidance, CFG)
在去噪速度场上施加风格外推引导：
$$
\hat{v} = v_\theta(x_t, t, \emptyset) + s \cdot \left( v_\theta(x_t, t, c) - v_\theta(x_t, t, \emptyset) \right)
$$
在配置中推荐设置 $s = 0.7$（`eval_cfg: 0.7`）。适度的 CFG 增益显著加强了书法家的个人辨识度，同时规避了高 CFG 导致的笔画粗暴膨胀。
