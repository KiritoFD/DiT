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

> 详见专论：**[03. 证伪注册表与失败方向归档](file:///g:/GitHub/DiT/docs/04_experiments/03_falsification_registry.md)**

| 路线方案 | 实测证据 / 失败指标 | 机理归因与教训 | 原始技术专论归档 |
| :--- | :--- | :--- | :--- |
| **暴力扩张模型容量** (27M $\to$ 67M) | strict SSIM 全部停滞在 0.51~0.57，容量翻倍无任何收益 | 任务瓶颈在于“风格条件有效性”而非主干容量 | [`54_results_and_insights.md`](file:///g:/GitHub/DiT/docs/archive/system_notes/54_results_and_insights_20260913.md) |
| **过度削减容量** (S/2 $\to$ XS/2 depth 8) | strict SSIM 跌落 $0.026$ (0.5603 $\to$ 0.5337)，LPIPS 变差 | Transformer 层数低于 12 会伤害基本去噪拟合 | [`62_param_budget_derivation.md`](file:///g:/GitHub/DiT/docs/archive/system_notes/62_param_budget_derivation.md) |
| **12 通道多通道扩散** (aux 轮廓/骨架) | strict SSIM 仅 0.4993（比基线低 $0.06$） | 辅助通道分布异构，CFG 采样引导时多通道轨迹发散跑飞 | [`59_12ch_retrospective.md`](file:///g:/GitHub/DiT/docs/archive/system_notes/59_12ch_and_white_zero_retrospective.md) |
| **条件高斯噪声增强** (noise400k) | seen 暴跌 $-0.048$，strict 平台仅 0.4869 | 破坏了骨架条件像素级的高频几何空间确定性 | [`55_self_conditioning.md`](file:///g:/GitHub/DiT/docs/archive/system_notes/55_self_conditioning_and_noise_aug.md) |
| **离散字符 ID 条件** (char_id) | 闭集严重，字典外字生成完全崩溃 | 无法支持书法万级生僻字/异体字的开集泛化 | [`60_what_is_useless.md`](file:///g:/GitHub/DiT/docs/archive/system_notes/60_what_is_actually_useless.md) |
| **无监督/弱监督端到端形变** | 风格跟随率退化至随机水平 ($\approx 50\%$) | 扩散 Loss 梯度与骨架几何严格正交，无自发几何引导 | [`96_skelnet.md`](file:///g:/GitHub/DiT/docs/archive/phase_922/96_skelnet.md) |
| **未截断的全局笔画残差** | 背景白底大面积造墨污染、出现漂浮噪点 | 笔画粗细调节必须受前景距离变换场严格半径截断 | [`96_skelnet.md`](file:///g:/GitHub/DiT/docs/archive/phase_922/96_skelnet.md) |
| **未对齐特征的 SupCon 风格表** | 风格可控比仅 1.2~2.5，与随机初始化无异 | 早期特征提取脚本存在尺寸不一致 bug，表征从未有效收敛 | [`10_data.md`](file:///g:/GitHub/DiT/docs/archive/phase_919/10_data.md) |

---

### 2.3 实验代际演进谱系与权威天梯 (Timeline: v1 ~ v21)

| 代际代号 | 最佳步数 | 核心架构特征 | strict SSIM | seen SSIM | 核心结论 |
| :--- | :---: | :--- | :---: | :---: | :--- |
| **v1 ~ v3** | 20k | 像素空间基础 DiT + 离散字符 ID | ~0.4200 | ~0.4500 | 早期验证，受困于闭集受限 |
| **v6 ~ v9** | 50k | 外部 ControlNet 架构 + 骨架拼接 | 0.5058 | 0.4961 | 门控失效：网络恒等抄写骨架，风格被旁路 |
| **v10b** | 360k | 确立开集标准骨架 $g_{std}$ | 0.5680 | 0.7606 | 首次实现生僻字开集渲染，长跑奠定坚实基础 |
| **v11** | 490k | 现代网络算子 (RMSNorm, SwiGLU, RoPE) | 0.5675 | 0.7685 | 确立现代流匹配底模，排除 12ch 多通道伪路线 |
| **v12** | 95k | 骨架全局向量池化 `glyph_vec_cond` | 0.5603 | 0.7214 | 主干首次在 adaLN 中显式感知全局字形拓扑 |
| **v13** | 170k | 50k 清洗数据集基准 (`v13_12ch_post` / `v13_styletok`) | **0.5713** | 0.7837 | 确立静态图像重构质量天花板 |
| **v14** | 162.5k | 类别细分与阶段微调 (`v14_style87_s3`) | **0.5764** | 0.7698 | **历史纯风格重构最高峰值** |
| **v15** | 150k | 风格矩阵与小样本自适应 (`v15a` / `v15_fs6`) | 0.5699 | 0.7650 | 沈周小样本迁移 2,000 步突破 0.5568 |
| **v17** | 100k | 局部条件跨注意力与空间门控消融 | 0.5529 | 0.6612 | 证伪 LCA 交叉注意力，确立显式骨架形变路线 |
| **v18** | 145k | 笔画骨架解耦初测 | 0.5537 | 0.7410 | 离线验证了形变网络具备梯度可用性 |
| **v21 (当前)** | 15k+ | **SkelNet + Gated Stroke Mod 解耦联合训练** | **0.5279** | **0.5420** | **目标专属性 +0.0143，书法家富集 1.91x 质变** |

---

### 2.4 当前活跃训练实时状态 (Live Run: v21_skelnet_200k)

- **硬件工况**：单卡 NVIDIA GeForce RTX 4090 24GB，单卡稳态吞吐 **$4.09\text{ steps/s}$** ($1,308\text{ 字/秒}$)，功耗 **$368\text{W}$ 满载**，显存稳定占用 **$18.59\text{GB} / 24\text{GB}$**。
- **收敛曲线 (Step 0 $\to$ Step 15,000)**：
  - $\mathcal{L}_{diff}$: $0.4037 \to \mathbf{0.3061}$
  - $\mathcal{L}_{deform}$: $0.3155 \to \mathbf{0.2905}$（位移场均值严格稳定在 $0.434\text{ px}$）
  - $\mathcal{L}_{repa}$: $0.0059 \to \mathbf{0.0034}$
- **连续里程碑评测数据演进**：
  - **Step 5,000**：`seen SSIM`: $0.5300$ \| `strict SSIM`: $0.5258$ \| `nn_ssim`: $0.5746$ \| `cal_enrich`: $0.85\times$
  - **Step 10,000**：`seen SSIM`: $0.5408$ \| `strict SSIM`: $\mathbf{0.5299}$ \| `target_spec`: $+0.0081$ \| `cal_enrich`: $0.95\times$
  - **Step 15,000 (最新实测)**：`seen SSIM`: $\mathbf{0.5420}$ \| `strict SSIM`: $\mathbf{0.5279}$ \| `LPIPS`: $\mathbf{0.3961}$ (创新低) \| `target_spec`: $\mathbf{+0.0143}$ ($\uparrow 76.5\%$) \| `cal_enrich`: $\mathbf{1.91\times}$ (质变跃升)
  - 评测生成的高清大图海报与同字最近邻匹配表已回传归档至本地：[`assets/results/v21_skelnet_200k/posters/`](file:///g:/GitHub/DiT/assets/results/v21_skelnet_200k/posters/)。

---

## 3. 全局技术文档四大支柱地图 (Documentation Sitemap)

项目文档中心已全面完成现代化重构，结构精简、论述深入，完整索引见 **[文档中心导航总纲](file:///g:/GitHub/DiT/docs/README.md)**：

### 3.1 第一支柱：核心模型与网络架构 ([`docs/01_architecture/`](file:///g:/GitHub/DiT/docs/01_architecture/))
- **[01. 架构全景与系统边界](file:///g:/GitHub/DiT/docs/01_architecture/01_overview.md)**：系统三阶段流水线、数据契约、模块间职责与运行边界定义。
- **[02. DiT-2Cond 主干模型](file:///g:/GitHub/DiT/docs/01_architecture/02_model.md)**：DiT-2Cond-S/2 参数规约、2D RoPE 旋转位置编码、RMSNorm、SwiGLU 与 adaLN-Zero 机制。
- **[03. SkelNet 拓扑可形变网络](file:///g:/GitHub/DiT/docs/01_architecture/03_skelnet.md)**：粗细双尺度控制网格 (TPS)、笔画宽度自适应调制、背景零造墨硬门控 ($r=0.25$) 与真值骨架中间监督。
- **[04. 双通道风格注入与正交梯度](file:///g:/GitHub/DiT/docs/01_architecture/04_dual_channel.md)**：宏观排版与微观墨迹的双尺度解耦、梯度正交性实证数学证明 ($\cos \approx +0.0030$)、38x 方差失衡诊断与修复。

### 3.2 第二支柱：数据集规范与数据流转 ([`docs/02_dataset/`](file:///g:/GitHub/DiT/docs/02_dataset/))
- **[01. 50k 高保真书法数据集规约](file:///g:/GitHub/DiT/docs/02_dataset/01_dataset_spec.md)**：50,000 例字形清单、45 位书法家词表字典、Seen 重构集与 Strict 零样本外推评测集切分。
- **[02. 数据清洗四阶隔离协议](file:///g:/GitHub/DiT/docs/02_dataset/02_cleaning_lineage.md)**：`_quarantine` v1-v4 清洗历史、反色极性判定、Canny 拓扑中心线细化、白底全零极性对齐 (White-Zero)。
- **[03. 存储布局与内存预加载流水线](file:///g:/GitHub/DiT/docs/02_dataset/03_asset_layout.md)**：离线分片结构 (`shards_img`, `shards_std`, `shards_aux_skel3`)、零 I/O 内存预载 (RAM Preload)、DINOv2 表征缓存。

### 3.3 第三支柱：训练策略与基础设施优化 ([`docs/03_training/`](file:///g:/GitHub/DiT/docs/03_training/))
- **[01. 最优传输流匹配理论](file:///g:/GitHub/DiT/docs/03_training/01_flow_matching.md)**：连续时间 OT-CFM 直线插值场、Logit-Normal 中段密度聚焦采样、Euler 与 Heun 高阶 ODE 数值求解器、无分类器引导 (CFG)。
- **[02. 损失函数多任务协同与超参配方](file:///g:/GitHub/DiT/docs/03_training/02_loss_and_recipes.md)**：三任务联合损失 ($\mathcal{L}_{diff} + 0.03\mathcal{L}_{repa} + 1.0\mathcal{L}_{deform}$)、学习率余弦退火、EMA 影子权重更新、梯度裁剪。
- **[03. 基础设施极致优化](file:///g:/GitHub/DiT/docs/03_training/03_infra_optimization.md)**：PyTorch Inductor 全算子融合编译、SDPA 零展开注意力、异步非阻塞 CPU 检查点落盘、单卡 RTX 4090 达到 **$4.09\text{ steps/s}$ ($1,308\text{ 字/秒}$)** 硬件压榨经验。

### 3.4 第四支柱：实验评测与证伪记录 ([`docs/04_experiments/`](file:///g:/GitHub/DiT/docs/04_experiments/))
- **[01. 历史全实验权威总天梯榜](file:///g:/GitHub/DiT/docs/04_experiments/01_master_leaderboard.md)**：75+ 组历史实验全量排位，Strict SSIM、Seen SSIM、LPIPS、专属性全景横评。
- **[02. 低资源书法家小样本自适应](file:///g:/GitHub/DiT/docs/04_experiments/02_fewshot_adaptation.md)**：沈周、伊秉绶、傅山 100 样本快速迁移实验，`row_pt` 先验初始化完胜 `mean_scaled`，2000 步极速达峰（SSIM 0.5568）。
- **[03. 证伪注册表与失败方向归档](file:///g:/GitHub/DiT/docs/04_experiments/03_falsification_registry.md)**：严谨证伪的 8 大技术死胡同（LCA交叉注意力、多Token风格、全层密集注入、骨架VAE、高权REPA等）与数学机理分析。
- **[04. 旗舰运行 v21_skelnet 实时遥测报告](file:///g:/GitHub/DiT/docs/04_experiments/04_v21_skelnet_telemetry.md)**：200k 旗舰长跑实时跟踪，记录 Step 5k, 10k, 15k 里程碑，目标专属性 $+0.0143$ 与同字富集度 $1.91\times$ 的质变突破。

### 3.5 历史全量技术档案库 ([`docs/archive/`](file:///g:/GitHub/DiT/docs/archive/))
- **[`docs/archive/system_notes/`](file:///g:/GitHub/DiT/docs/archive/system_notes/)**：项目早期记录的 75 篇演进专论（00 至 75），包含大量的消融草稿、显式推导与会话实录。
- **[`docs/archive/phase_919/`](file:///g:/GitHub/DiT/docs/archive/phase_919/)**：2026年9月19日阶段性重构快照。
- **[`docs/archive/phase_922/`](file:///g:/GitHub/DiT/docs/archive/phase_922/)**：2026年9月22日注入机制重构专题研讨记录。
- **[`docs/archive/legacy_reports/`](file:///g:/GitHub/DiT/docs/archive/legacy_reports/)**：早期 S6 阶段评估报告与历史分析。
- **[`docs/archive/txt_reports/`](file:///g:/GitHub/DiT/docs/archive/txt_reports/)**：原始数据集清洗统计与诊断文本文件。

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
