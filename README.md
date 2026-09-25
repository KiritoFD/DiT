# Callig-DiT（马良）

> **Multi-Condition Diffusion Transformer for Stylized Chinese Calligraphy Generation**  
> 基于多条件扩散 Transformer 的高可控风格化中国书法字形生成模型。

---

## 1. 架构总纲 (Architecture)

### 1.1 任务形式与建模边界

书法字形生成的核心矛盾在于**字形结构的开集泛化**与**书家风格的微观辨识**。Callig-DiT 抛弃离散字符 ID，采用两路物理输入联合驱动去噪生成：

1. **开集标准骨架潜变量 $g_{std} \in \mathbb{R}^{4 \times 32 \times 32}$**：由规范字体跨书体渲染生成。开集字符无需加入闭集词表，只要字体库能渲染该字，模型即可泛化书写；
2. **闭集历史书家类别 $y_{callig}$**：查表获得 128 维密集风格嵌入向量 $e_{callig} \in \mathbb{R}^{128}$；
3. **输出**：书法真迹潜变量 $x_0 \in \mathbb{R}^{4 \times 32 \times 32}$，通过预训练 VAE 解码为 $256 \times 256$ 图像。

```
                       [ 标准骨架 g_std ] (4, 32, 32)
                                     │
                                     ▼
[ 书家标签 y_callig ] ──> e_callig ──> [ SkelNet (DeformSkel) ] ──> [ 形变骨架 g' ] ──┬──> L_deform 监督
  (闭集类别查表 128d)          │                 │                                   ├──> ① 输入残差加和 (0.6x)
                               │                 ├─ FiLM 特征调制                     ├──> ② 4层 ZeroAdaLN 跨注意力
                               │                 ├─ 全局仿射 (错切/缩放/旋转)          └─> ③ 全局均值池化
                               │                 ├─ 全局偏移场 (dx, dy)                         │
                               │                 └─ 门控笔画调制 (Gated Stroke Mod)             ▼
                               │                                                         e_glyph_vec (128d)
                               ▼                                                                │
                   [ cond_fusion 融合层: LayerNorm + Linear ] <─────────────────────────────────┘
                                               │
                                               ▼
                                      y_emb (384d 条件向量)
                                               │
    [ 时间步 t ] ──> t_emb (384d) ────────────(+)──> c = t_emb + y_emb (全局调制向量)
                                                            │
                                                            ▼
    [ 加噪隐变量 x_t ] ─────────────────────────> [ 12 层 DiTBlock ] (DiT-S/2, 36.5M)
                                                      ├─ RoPE 旋转位置编码
                                                      ├─ RMSNorm + QK-Norm + SwiGLU
                                                      ├─ adaLN: 6 路调节 (scale, shift, gate)
                                                      └─ REPA DINOv2 表征对齐 (Layer 8)
                                                            │
                                                            ▼
                                              [ FinalLayer & Unpatchify ] ──> 速度场预测 v_pred
```

### 1.2 核心组件机制

#### (1) SkelNet 间架形变网（DeformSkel）
- **定位**：3.58M 参数的轻量 U-Net。解决“标准字形 $g_{std}$ 跨书家共享导致网络照抄”的核心瓶颈。
- **调制方式**：
  - **FiLM 卷积调制**：逐尺度特征缩放与平移；
  - **全局几何仿射**：预测 $2\times 2$ 仿射矩阵；
  - **坐标位移场 $(dx, dy)$**：重构字的重心与部首布局；
  - **门控笔画调制（Gated Stroke Mod）**：在前景距离场截断半径（$\text{radius} \le 0.25$）内收放笔画粗细，彻底杜绝背景造墨污染。
- **输出载体**：输出个性化骨架 $g'$，一分为四：
  1. 输入补丁直接相加：$x = x + 0.6 \cdot g_{tok}$；
  2. 中间 4 层通过 `ZeroAdaLNInjection` 逐层注入；
  3. 全局池化出 $e_{glyph\_vec} \in \mathbb{R}^{128}$ 供给主干条件；
  4. 显式接受真迹骨架损失监督：$\mathcal{L}_{deform} = \|g' - g_{inst}\|^2$（权重 $1.0$）。

#### (2) 双通道解耦注入机制
- **通道 A（间架结构轴 / SkelNet）**：掌控字的宏观体势（欹侧、收放、中宫、重心与粗细比例）；
- **通道 B（笔触质感轴 / Backbone adaLN）**：掌控墨法浓淡、枯笔飞白、毛边与微观笔锋动态。

#### (3) 现代生成底座与损失函数
- **Flow Matching 连续流匹配**：二阶 Heun-RK2 ODE 采样，训练使用 Logit-Normal 时间步采样；
- **全现代结构**：RMSNorm 归一化、SwiGLU 前馈网络、RoPE 旋转位置编码、注意力 QK-Norm；
- **联合损失函数**：
  $$\mathcal{L}_{total} = \mathcal{L}_{diff} + 0.03 \mathcal{L}_{repa} + 1.0 \mathcal{L}_{deform}$$
  其中 $\mathcal{L}_{repa}$ 在第 8 层与冻结的 DINOv2 ViT-S/14 特征对齐。

---

## 2. 实验、理论实证与结论记录 (Experiments & Empirical Findings)

### 2.1 关键物理机制与数学实证

#### 结论 1：标准骨架高度共享，无干预必导致风格门控失效
MCCD 数据集实测表明：按 MD5 严格查重，跨书家共享完全相同 $g_{std}$ 的样本占比达 **$43.5\%$**（如李邕与褚遂良共用同一标准骨架）。若仅依赖主网络去噪损失，网络将迅速退化为依赖 $g_{std}$ 的恒等描摹，风格通道被数学上自动门控置零。

#### 结论 2：SkelNet 自带梯度的自发性
形变标准骨架更贴近目标书家真迹，**可直接降低去噪重建损失**。在独立预训练中，SkelNet 达到 **$95.3\%$** 的风格跟随率（closure rate 44.9%）；在下游联合训练中，骨架潜变量相对形变率达到 **$53.05\%$**。

#### 结论 3：梯度正交性证明（扩散损失对骨架形变无几何偏好）
在 Step 5000 真实模型上利用自动微分探针对计算图分流求导，实测：
- 主干 adaLN 通道扩散梯度：$\|\nabla_{e_{dit}} \mathcal{L}_{diff}\| = 0.000276$
- 骨架形变通道扩散梯度：$\|\nabla_{e_{skel}} \mathcal{L}_{diff}\| = 0.005552$
- 骨架显式监督梯度：$\|\nabla_{e_{skel}} \mathcal{L}_{deform}\| = \mathbf{0.034526}$（强出 6.22 倍）
- **梯度余弦相似度**：
  $$\cos(\nabla_{e_{skel}}\mathcal{L}_{diff}, \; \nabla_{e_{skel}}\mathcal{L}_{deform}) = \mathbf{+0.0030} \approx 0$$
- **物理证明**：扩散去噪任务自身对间架细节形变呈现**完全正交**状态。若无 $\mathcal{L}_{deform}$ 的垂直锚定约束，骨架形变网络根本无法从扩散信号中自发学会历史大家的真实结体。

#### 结论 4：输入扰动消融验证双通道有效性
在 Step 5000 检查点上进行敏感度消融测试：
- **基准损失**：$\mathcal{L}_{diff} = 0.34680$
- **打乱主干 adaLN 风格（间架对，纹理错）**：Loss 恶化至 $0.35043$（**$+1.05\%$**），证明纹理通道强力生效；
- **关闭骨架形变（退化为 $g_{std}$）**：Loss 恶化至 $0.34936$（**$+0.74\%$**），证明间架形变具备确定性降低重建难度的能力；
- **双通道同时打乱**：Loss 恶化至 $0.35023$（$+0.99\%$）。

#### 结论 5：诊断出 `cond_fusion` 处的方差失衡瓶颈
在融合层输入处，实测 $\|e_{callig}\| = 9.86$ 而 $\|e_{glyph\_vec}\| = 60.90$（**数值相差 6.18 倍，方差相差 38 倍**）。全局 `LayerNorm(256)` 导致风格信号在主干中被骨架信号稀释约 $1/5$。后续版本将改用独立 LayerNorm 后拼接，预计可释放数倍风格敏感度。

---

### 2.2 证伪清单 (Falsification Registry：已实测排除的 8 大死胡同)

| 路线方案 | 实测证据 / 失败指标 | 机理归因与教训 | 归档文档 |
| :--- | :--- | :--- | :--- |
| **暴力扩张模型容量** (27M $\to$ 67M) | strict SSIM 全部停滞在 0.51~0.57，容量翻倍无任何收益 | 任务瓶颈在于“风格条件有效性”而非主干容量 | `docs/system/54` |
| **过度削减容量** (S/2 $\to$ XS/2 depth 8) | strict SSIM 跌落 $0.026$ (0.5603 $\to$ 0.5337)，LPIPS 变差 | Transformer 层数低于 12 会伤害基本去噪拟合 | `docs/system/62` |
| **12 通道多通道扩散** (aux 轮廓/骨架) | strict SSIM 仅 0.4993（比基线低 $0.06$） | 辅助通道分布异构，CFG 采样引导时多通道轨迹发散跑飞 | `docs/system/59` |
| **条件高斯噪声增强** (noise400k) | seen 暴跌 $-0.048$，strict 平台仅 0.4869 | 破坏了骨架条件像素级的高频几何空间确定性 | `docs/system/55` |
| **离散字符 ID 条件** (char_id) | 闭集严重，字典外字生成完全崩溃 | 无法支持书法万级生僻字/异体字的开集泛化 | `docs/system/60` |
| **无监督/弱监督端到端形变** | 风格跟随率退化至随机水平 ($\approx 50\%$) | 扩散 Loss 梯度与骨架几何严格正交，无自发几何引导 | `docs/922/96` |
| **未截断的全局笔画残差** | 背景白底大面积造墨污染、出现漂浮噪点 | 笔画粗细调节必须受前景距离变换场严格半径截断 | `docs/922/96` |
| **未对齐特征的 SupCon 风格表** | 风格可控比仅 1.2~2.5，与随机初始化无异 | 早期特征提取脚本存在尺寸不一致 bug，表征从未有效收敛 | `docs/919/10` |

---

### 2.3 实验代际演进谱系 (Timeline: v1 ~ v21)

| 代际代号 | 核心架构特征 | 关键干预与超参 | strict SSIM / 结论 |
| :--- | :--- | :--- | :--- |
| **v1 ~ v3** | 像素空间基础 DiT | 离散字符 ID + 书家 ID | 早期概念验证，闭集受限 |
| **v6 ~ v9** | 引入骨架潜变量条件 | 外部 ControlNet 架构 | 门控失效：网络恒等抄写骨架，风格被旁路 |
| **v10** | 确立开集标准骨架 $g_{std}$ | 单向量因子，彻底剥离 char ID | 0.5108；首次实现生僻字开集渲染 |
| **v11** | 现代化改造 + 12ch 探索 | RMSNorm, SwiGLU, RoPE, 12ch aux | 0.4993；证明 12ch 辅助目标失败，但现代骨干确立 |
| **v12** | 骨架全局向量池化 | `glyph_vec_cond` (mean pooling) | 0.5603；主干首次在 adaLN 中感知全局字形拓扑 |
| **v13** | 50k 清洗数据集基准 | 因子化拼接 `factorized_cat` | **0.5703**；确立像素级 SSIM 天花板 |
| **v14** | 类别细分 (87 pair) | 书家×书体细分，风格表全量微调 | 0.5691；证明纯类别扩充无法解决单向量模态塌缩 |
| **v15** | 多模态风格矩阵 | $K=4$ token 查表，DINO 质心初始化 | 建立多模态风格表达框架 |
| **v17** | 局部条件跨注意力与门控 | 时间门控 `glyph_gate_t` | 验证局部结构与全局纹理的时序分工 |
| **v18 ~ v20** | 骨架形变探索 | 引入可变形卷积，中间骨架初测 | 离线验证了形变网络具备梯度可用性 |
| **v21 (当前)** | **SkelNet + Gated Stroke Mod** | **离线 95.3% 注入 + 小 lr 联合训练 + $w=1.0$ 稠密监督** | **Step 10,000: seen 0.5408 / strict 0.5299，稳步爬升** |

---

### 2.4 当前活跃训练实时状态 (Live Run: v21_skelnet_200k)

- **硬件工况**：NVIDIA GeForce RTX 4090 24GB，单卡稳态吞吐 **$4.07\text{ steps/s}$**，功率 **$368\text{W}$**，显存稳定占用 **$18.65\text{GB}$**。
- **收敛曲线**：
  - Step 0 $\to$ Step 11,200+：
    - $\mathcal{L}_{diff}$: $0.4037 \to \mathbf{0.3141}$
    - $\mathcal{L}_{deform}$: $0.3155 \to \mathbf{0.2939}$（位移场均值稳定在 $0.422\text{ px}$）
    - $\mathcal{L}_{repa}$: $0.0059 \to \mathbf{0.0036}$
- **评测指标演进（每 5,000 步）**：
  - **Step 5,000**：`seen SSIM`: $0.5300$ \| `strict SSIM`: $0.5258$ \| `nn_ssim`: $0.5746$
  - **Step 10,000**：`seen SSIM`: $\mathbf{0.5408}$ ($\uparrow 0.0108$) \| `strict SSIM`: $\mathbf{0.5299}$ ($\uparrow 0.0041$) \| `target_spec`: $\mathbf{+0.0081}$
  - 生成海报及最近邻比对数据已实时回传至本地：`assets/results/v21_skelnet_200k/posters/`。

---

## 3. 全局全量文档索引地图 (Documentation Sitemap)

### 3.1 前沿技术决策与实验报告 (`docs/922/`)
- [`97_dual_channel_analysis.md`](docs/922/97_dual_channel_analysis.md)：**【核心必读】** 双通道机制审计、Step 5000 探针量化数据、梯度正交性证明与方差瓶颈分析。
- [`96_skelnet.md`](docs/922/96_skelnet.md)：SkelNet 骨架形变网完整原理、数据证据、门控笔画调制与对比学习设计。
- [`95_module_decision.md`](docs/922/95_module_decision.md)：技术路线选型裁定与历史试错复盘。
- [`94_style_supervision.md`](docs/922/94_style_supervision.md)：风格对比学习与难负例挖掘策略。
- [`93_samechar_nn_diag.md`](docs/922/93_samechar_nn_diag.md)：同字最近邻诊断与字形特异性量化体系。
- [`92_skel_condition_aug.md`](docs/922/92_skel_condition_aug.md)：骨架条件几何扰动增强实验记录。
- [`90_style_injection_summary.md`](docs/922/90_style_injection_summary.md)：历代风格注入机制横向对比总结。

### 3.2 系统级全景快照 (`docs/919/`)
- [`00_overview.md`](docs/919/00_overview.md)：任务定义、输入输出约定与结论清单总览。
- [`10_data.md`](docs/919/10_data.md)：数据资产全谱（Fame / TJ / 50k_v2、分片缓存、清洗与评测集定义）。
- [`20_model.md`](docs/919/20_model.md)：DiT-2Cond 核心模型拓扑、条件注入接口与证伪路线解释。
- [`30_training.md`](docs/919/30_training.md)：训练超参配方、Flow Matching 采样器设计与历史实验链。
- [`40_results.md`](docs/919/40_results.md)：评测总表、逐书家指标分布与书法质感分析。
- [`50_infra.md`](docs/919/50_infra.md)：运行环境、算力设施、显存瓶颈、编译优化与运维排错手册。

### 3.3 历史演进专论 (`docs/system/`)
- 包含从 `00` 到 `75` 篇系统演进专论，涵盖容量预算推导（`62`）、泛化本质分析（`72`）与风格排序显著性检验（`75`）。

---

## 4. 常用工具与操作指南 (Tooling & Quickstart)

### 4.1 双通道梯度与强度自动化探针
在任意检查点上一键测量风格双通路信号分布、形变量与梯度流：
```bash
python tools/probe_dual_channel.py \
    --config src/train/configs/v21_skelnet_200k.json \
    --ckpt assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0005000.pt \
    --batch-size 16 \
    --device cpu
```

### 4.2 主干训练启动 (v21 规范)
```bash
python -m src.train.train --config src/train/configs/v21_skelnet_200k.json
```

### 4.3 离线评测与可视化海报生成
```bash
python -m src.eval.in_process_eval \
    --ckpt assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0010000.pt \
    --eval-sets "seen:assets/eval_v13_seen.csv:20,strict:assets/eval_v13_strict.csv:250" \
    --cfg 0.7 \
    --steps 50
```
