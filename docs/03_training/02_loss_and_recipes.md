# 损失函数多任务协同与训练超参配方

## 1. 多任务联合目标函数 (Joint Multi-Task Objective)

马良的总体优化目标定义为一个精巧加权的三目标函数：
$$
\mathcal{L}_{total} = \mathcal{L}_{diff} + w_{repa} \cdot \mathcal{L}_{repa} + w_{deform} \cdot \mathcal{L}_{deform}
$$

```
                         [总损失 L_total]
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
 [L_diff (Flow Matching)] [L_repa (DINO Alignment)] [L_deform (SkelNet GT)]
  权重: 1.0                权重: 0.03              权重: 1.0
  目标: 速度场无偏回归     目标: 语义空间正交对齐  目标: 几何拓扑刚性锚定
```

### 1.1 流匹配回归损失 ($\mathcal{L}_{diff}$)
速度场均方误差损失：
$$
\mathcal{L}_{diff} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t, c, S_{def}) - (x_1 - x_0) \|_2^2 \right]
$$
在训练稳态下，Diff 损失值稳定在 $\approx 0.3060 - 0.3120$ 之间。

### 1.2 表征自监督对齐损失 ($\mathcal{L}_{repa}$)
将 DiT-2Cond 第 8 层的隐藏特征向量 $h_8$ 通过轻量线性投影头 $g_\phi$ 映射至 768 维，与离线缓存的 DINOv2 表征 $f_{dino}$ 计算负余弦相似度：
$$
\mathcal{L}_{repa} = 1 - \frac{\langle g_\phi(h_8), f_{dino} \rangle}{\|g_\phi(h_8)\|_2 \cdot \|f_{dino}\|_2}
$$
- **超参权重**：`w_repa = 0.03`
- **经验教训**：在过往的消融实验（如 `w_repa = 0.5` 或 `0.2`）中，强迫生成模型对齐高维语义表征会导致文字失去精细的笔画边缘锐度，产生油画般的模糊晕染。低权重 $0.03$ 既提供了充分的流形正则化，又彻底保护了生成锐度。

### 1.3 骨架几何中间监督损失 ($\mathcal{L}_{deform}$)
$$
\mathcal{L}_{deform} = \| S_{def} - S_{GT} \|_1
$$
- **超参权重**：`w_deform_skel = 1.0`
- 确保 SkelNet 预测出的形变骨架严格锚定真实书法字中心线，平均位移误差限制在 $0.43\text{ px}$ 以内，防止对抗性发散。

---

## 2. 训练超参数与学习率调度配方 (v21 黄金基准)

旗舰配置 [`src/train/configs/v21_skelnet_200k.json`](file:///g:/GitHub/DiT/src/train/configs/v21_skelnet_200k.json) 的完整训练配方如下：

| 参数 | 设定值 | 机制与功能 |
| :--- | :--- | :--- |
| **Max Steps** | `200,000` | 20 万步充分拟合（约 200 个 Epoch，耗时约 13.6 小时） |
| **Global Batch Size** | `320` | 单卡极限大批次，提升梯度估计信噪比 |
| **Base LR** | `5e-4` ($0.0005$) | 主干 DiT-2Cond 学习率 |
| **Warmup Steps** | `3,000` | 线性热身，使优化器动量矩估计平稳建立 |
| **LR Schedule** | `Cosine Annealing` | 3,000 步后余弦衰减至 `5e-5` (`min_lr_ratio: 0.1`) |
| **SkelNet LR Scale** | `0.1` | SkelNet 有效学习率 $5 \times 10^{-5}$，解耦更新步长 |
| **Weight Decay** | `0.02` | AdamW 权重衰减，控制非线性层范数 |
| **Optimizer** | `AdamW` | $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$ |
| **EMA Decay** | `0.9999` | 影子权重指数移动平均，更新间隔 `ema_interval = 4` |
| **Mixed Precision** | `BF16` / `FP16` | PyTorch AMP 自动混合精度，杜绝溢出 |
| **Condition Dropout** | `0.1` | 全局无条件丢弃率 10%，用于支持无分类器引导（CFG） |

---

## 3. 梯度裁剪与稳态诊断

- **梯度裁剪范数**：`grad_clip = 1.0`，拦截一切局部拓扑剧烈震荡带来的梯度爆炸；
- **全通道梯度模长监控**：训练日志每 5,000 步输出各子模块梯度分布：
  ```
  [diag] grad-norm: blocks=0.037 glyph_embedder=0.014 inj_out_proj=0.022 other=0.544
  ```
  保证所有网络主干模块的梯度处于同一数量级，未发生任何梯度弥散（Vanishing）或死区。
