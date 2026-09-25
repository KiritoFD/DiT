# Callig-DiT (马良) 研发全景文档索引

> **马良 (Callig-DiT)**：基于显式拓扑解耦与最优传输流匹配的高保真汉字书法生成大模型。
> 本文档中心为全项目的技术规范、实验结论、数据集流转与工程实现的总索引。

---

## 📚 核心文档四大支柱 (Four Core Pillars)

### 第一支柱：核心模型与网络架构 ([`docs/01_architecture/`](file:///g:/GitHub/DiT/docs/01_architecture/))
- **[01. 架构全景与系统边界](file:///g:/GitHub/DiT/docs/01_architecture/01_overview.md)**：系统三阶段流水线、数据契约、模块间职责与运行边界定义。
- **[02. DiT-2Cond 主干模型](file:///g:/GitHub/DiT/docs/01_architecture/02_model.md)**：DiT-2Cond-S/2 参数规约、2D 旋转位置编码 (RoPE)、RMSNorm、SwiGLU 与 adaLN-Zero 机制。
- **[03. SkelNet 拓扑可形变网络](file:///g:/GitHub/DiT/docs/01_architecture/03_skelnet.md)**：粗细双尺度控制网格 (TPS)、笔画宽度自适应调制、背景零造墨硬门控 ($r=0.25$) 与真值骨架中间监督。
- **[04. 双通道风格注入与正交梯度](file:///g:/GitHub/DiT/docs/01_architecture/04_dual_channel.md)**：宏观排版与微观墨迹的双尺度解耦、梯度正交性实证数学证明 ($\cos \approx +0.0030$)、38x 方差失衡诊断与修复。

### 第二支柱：数据集规范与数据流转 ([`docs/02_dataset/`](file:///g:/GitHub/DiT/docs/02_dataset/))
- **[01. 50k 高保真书法数据集规约](file:///g:/GitHub/DiT/docs/02_dataset/01_dataset_spec.md)**：50,000 例字形清单、45 位书法家词表字典、Seen 重构集与 Strict 零样本外推评测集切分。
- **[02. 数据清洗四阶隔离协议](file:///g:/GitHub/DiT/docs/02_dataset/02_cleaning_lineage.md)**：`_quarantine` v1-v4 清洗历史、反色极性判定、Canny 拓扑中心线细化、白底全零极性对齐 (White-Zero)。
- **[03. 存储布局与内存预加载流水线](file:///g:/GitHub/DiT/docs/02_dataset/03_asset_layout.md)**：离线分片结构 (`shards_img`, `shards_std`, `shards_aux_skel3`)、零 I/O 内存预载 (RAM Preload)、DINOv2 表征缓存。

### 第三支柱：训练策略与基础设施优化 ([`docs/03_training/`](file:///g:/GitHub/DiT/docs/03_training/))
- **[01. 最优传输流匹配理论](file:///g:/GitHub/DiT/docs/03_training/01_flow_matching.md)**：连续时间 OT-CFM 直线插值场、Logit-Normal 中段密度聚焦采样、Euler 与 Heun 高阶 ODE 数值求解器、无分类器引导 (CFG)。
- **[02. 损失函数多任务协同与超参配方](file:///g:/GitHub/DiT/docs/03_training/02_loss_and_recipes.md)**：三任务联合损失 ($\mathcal{L}_{diff} + 0.03\mathcal{L}_{repa} + 1.0\mathcal{L}_{deform}$)、学习率余弦退火、EMA 影子权重更新、梯度裁剪。
- **[03. 基础设施极致优化](file:///g:/GitHub/DiT/docs/03_training/03_infra_optimization.md)**：PyTorch Inductor 全算子融合编译、SDPA 零展开注意力、异步非阻塞 CPU 检查点落盘、单卡 RTX 4090 达到 **$4.09\text{ steps/s}$ ($1,308\text{ 字/秒}$)** 硬件压榨经验。

### 第四支柱：实验评测与证伪记录 ([`docs/04_experiments/`](file:///g:/GitHub/DiT/docs/04_experiments/))
- **[01. 历史全实验权威总天梯榜](file:///g:/GitHub/DiT/docs/04_experiments/01_master_leaderboard.md)**：75+ 组历史实验全量排位，Strict SSIM、Seen SSIM、LPIPS、专属性全景横评。
- **[02. 低资源书法家小样本自适应](file:///g:/GitHub/DiT/docs/04_experiments/02_fewshot_adaptation.md)**：沈周、伊秉绶、傅山 100 样本快速迁移实验，`row_pt` 先验初始化完胜 `mean_scaled`，2000 步极速达峰（SSIM 0.5568）。
- **[03. 证伪注册表与失败方向归档](file:///g:/GitHub/DiT/docs/04_experiments/03_falsification_registry.md)**：严谨证伪的 8 大技术死胡同（LCA交叉注意力、多Token风格、全层密集注入、骨架VAE、高权REPA等）与数学机理分析。
- **[04. 旗舰运行 v21_skelnet 实时遥测报告](file:///g:/GitHub/DiT/docs/04_experiments/04_v21_skelnet_telemetry.md)**：200k 旗舰长跑实时跟踪，记录 Step 5k, 10k, 15k 里程碑，目标专属性 $+0.0143$ 与同字富集度 $1.91\times$ 的质变突破。

---

## 🏛️ 历史档案与原始文献库 ([`docs/archive/`](file:///g:/GitHub/DiT/docs/archive/))

为了完整保留科研探索的真实心路历程，所有历史文档与原始记录均妥善归档：
- **[`docs/archive/system_notes/`](file:///g:/GitHub/DiT/docs/archive/system_notes/)**：项目早期建立的 75 篇连续演进备忘录（00 至 75），包含大量的每日消融试验草稿、显式推导与会话实录。
- **[`docs/archive/phase_919/`](file:///g:/GitHub/DiT/docs/archive/phase_919/)**：2026年9月19日阶段性重构快照文档集合。
- **[`docs/archive/phase_922/`](file:///g:/GitHub/DiT/docs/archive/phase_922/)**：2026年9月22日注入机制重构专题研讨文档集合。
- **[`docs/archive/legacy_reports/`](file:///g:/GitHub/DiT/docs/archive/legacy_reports/)**：早期的 S6 阶段评估报告、初版 ControlNet 设计草案与历史分析报告。
- **[`docs/archive/txt_reports/`](file:///g:/GitHub/DiT/docs/archive/txt_reports/)**：远端与本地运行产生的数据集分布诊断、统计清洗输出原始 `.txt` 文本文件。