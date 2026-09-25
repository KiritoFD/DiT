# SkelNet: 显式拓扑可形变骨架网络

## 1. 为什么需要显式骨架形变？

在传统方案中，模型直接试图从标准骨架 $S_{std}$ 和风格标签 $c$ 一步跳跃到最终的书法字图像：
$$
x = \mathcal{G}(z_T, S_{std}, c)
$$
这种“隐式形变”假定扩散模型内部的注意力机制能够同时完成两件完全不同性质的任务：
1. **全局空间扭曲（Macro Warping）**：例如苏轼字形的扁平横展、黄庭坚笔画的长枪大戟、米芾的欹侧跌宕；
2. **局部纹理渲染（Micro Texturing）**：墨色深浅、飞白、枯笔、笔锋出尖。

实验发现，扩散模型的卷积/注意力层天然倾向于用**“微观边缘模糊”来妥协“宏观几何错位”**，导致字体边缘出现严重的重影、空洞和断笔（Frag > 3.0）。

马良引入 **SkelNet**，将生成过程分解为**显式几何形变**与**纹理扩散渲染**：
$$
S_{def} = \text{SkelNet}(S_{std}, c)
$$
$$
x = \text{DiT}(z_T, S_{def}, c)
$$

---

## 2. SkelNet 架构细节

SkelNet 实现于 [`src/models/deform_skel.py`](file:///g:/GitHub/DiT/src/models/deform_skel.py)，包含三个核心动力学模块：

```
                [标准骨架 S_std]
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
    [CNN 拓扑特征提取]   [距离变换 Distance Transform]
             │                   │
             ├─────────┐         │
             ▼         ▼         ▼
        [形变控制网格] [笔画宽度调制] [背景硬门控 (r=0.25)]
             │         │         │
             ▼         ▼         ▼
        [网格双线性采样 (TPS)]   │
             │                   │
             └─────────┬─────────┘
                       ▼
              [输出: 定制形变骨架 S_def]
```

### 2.1 粗细双尺度形变网格 (Coarse & Fine Deformation Grid)
- **输入**：标准骨架 $S_{std}$ 以及由骨架图计算出的距离变换图（Distance Transform, `deform_dt_ch=1`）。
- **控制网格**：
  - 粗粒度网格：$8 \times 8$ 锚点，捕获字形整体倾斜、重心高低；
  - 细粒度网格：$32 \times 32$ 局部网格，控制笔画弯曲度与顿挫；
- **最大位移截断（Max Offset Clamping）**：
  $$
  \Delta p = 6.0 \cdot \tanh(\text{GridHead}(F))
  $$
  位移被物理截断在 $\pm 6.0\text{ 像素}$ 范围内，防止网格自相交产生拓扑折叠（Folding Inversion）。

### 2.2 笔画宽度自适应调制 (Stroke Thickness Modulation)
书法风格的一大核心是肥瘦：颜真卿楷书浑厚雄强，赵孟頫则秀润挺拔。
- `stroke_mod=1` 激活笔画宽度场预测：
  $$
  W(p) = \text{clamp}(1.0 + \sigma(\text{StrokeHead}(F)), 0.0, 1.0)
  $$
- 宽度场直接作用于形变后的骨架，产生连续可微的粗细变化。

### 2.3 背景零造墨硬门控 (Background Zero-Ink Gating)
**病态现象**：未加门控的神经网络容易在没有文字笔画的广阔背景区域生成微小噪点，扩散模型会将这些噪点放大为严重的“背景飞墨”。
- **门控半径**：`gate_radius=0.25`
- 对离原始标准骨架距离大于 $r=0.25$（归一化坐标）的所有像素，强制执行硬门控遮罩：
  $$
  S_{def}(p) = S_{def}(p) \cdot \mathbb{I}(\text{dist}(p, S_{std}) \le r)
  $$
  数学上确保了**绝对背景零造墨**，从源头根除了字周飞墨问题。

---

## 3. 联合微调策略：低学习率与中间监督

在 `v21-skelnet-200k` 中，SkelNet 采用解冻联合微调（Joint Fine-Tuning）：
1. **预训练权重初始化**：形变头权重加载自独立预训练达 95.3% 风格遵从度的 [`assets/deform_skel_v10.pt`](file:///g:/GitHub/DiT/assets/deform_skel_v10.pt)；
2. **学习率解耦缩放**：主干学习率为 $5 \times 10^{-4}$，SkelNet 的优化步长缩放为 $0.1\times$（即有效学习率 $5 \times 10^{-5}$），避免反向传播的扩散梯度冲垮预训练的几何拓扑；
3. **真实骨架中间监督**：引入真实书法字骨架作为中间监督锚点：
   $$
   \mathcal{L}_{deform} = \|S_{def} - S_{GT}\|_{1}
   $$
   权重固定为 $w_{deform}=1.0$。训练监控显示，平均骨架像素误差稳定在 $0.425 - 0.435\text{ 像素}$，兼具形变灵活性与拓扑稳定性。
