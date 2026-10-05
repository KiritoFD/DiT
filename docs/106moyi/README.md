# 墨韵 (Moyi) 实验全系历史资产与容量增强基准总览

> **归档位置**：`docs/106moyi/`  
> **数据来源**：从机器 48 (`10.222.120.101`) 的 `/home/ds/Workspace/moyi/` 与 `/home/ds/Workspace/DiT/experiments/` 完整回传整理  
> **同步内容**：全部 733 个实验历史文件（包含所有 JSON 配置、评估指标、训练日志、全景评测海报及代表性样本，已剔除大权重与 Shards）  

---

## 一、 核心基准总评测表 (Benchmark Matrix)

本表格汇总了墨韵（Moyi）系列在黄金测试集 `eval200_fixed.csv`（包含 187 个严格字符与 20 个已见字符）上的标准化评测指标（完整数据见 [`moyi_experiments_summary.csv`](./moyi_experiments_summary.csv)）：

| 实验系列 / 代号 | 模型架构 | 通道 / 潜在空间 | 参数量 | 评测步数 | Strict SSIM ↑ | Strict LPIPS ↓ | Skel IoU ↑ | 核心结论与学术意义 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Moyi_Top10_RF_4ch** | Moyi-RF-4ch | SD-VAE (4ch) | 基线 | 50,000 | **0.5949** | 0.3678 | 0.0161 | 4通道传统 VAE 的 50k 高位基线 |
| **Moyi_Top10_RF_4ch** | Moyi-RF-4ch | SD-VAE (4ch) | 基线 | 80,000 | **0.6085** | **0.3489** | **0.0235** | ★ 突破 0.60 大关的最高分模型之一 |
| **Moyi_Top10_RF_16ch**| Moyi-RF-16ch | FLUX-VAE (16ch)| 基线 | 50,000 | **0.5935** | - | 0.0177 | 高维 16ch 潜在空间具备极高拓扑保真潜力 |
| **Phase_A_Exp1** | DiT-2Cond-S/2 | SD-VAE (4ch) | 33.8M | 10,000 | **0.5408** | 0.3926 | 0.0140 | 遭遇 33M 参数容量硬瓶颈（微观组件无法解救） |
| **Capacity_Tier2_Sp** | DiT-2Cond-Sp/2| SD-VAE (4ch) | 59.2M | 10,000 | **0.5494** | 0.3848 | 0.0150 | 容量扩展验证：同 10k 步单调提升 +0.0086 |
| **Capacity_Tier2_Sp** | DiT-2Cond-Sp/2| SD-VAE (4ch) | 59.2M | 20,000 | **0.5606** | 0.3747 | 0.0158 | ★ Sp/2 终测达到 0.5606，拉开与 S/2 差距 |
| **Capacity_Tier3_B**  | DiT-2Cond-B/2 | SD-VAE (4ch) | 131.0M| 5,000 | **0.5346** | 0.3994 | 0.0134 | 5k 步处于大模型初级爬坡期，Loss 持续破新低 |
| **Capacity_Tier3_B_Aug**| DiT-2Cond-B/2 | SD-VAE (4ch) | 131.0M| 48,000 | *训练中* | *训练中* | *训练中* | **当前运行中：10h 满载冲刺 + 7.8万对称数据增强** |

---

## 二、 实验历史体系复盘与演进脉络

### 1. 墨韵前置基线探索 (Moyi RF 4ch / 16ch)
- **探索重点**：在 `top10_style23`（10位名家传世字帖）上，对比标准 Rectified Flow 在 SD 4通道 VAE 与 FLUX 16通道 VAE 空间中的建模能力。
- **关键突破**：
  - `moyi_top10_rf_4ch` 在 80,000 步时 Strict SSIM 冲到了 **0.6085**，LPIPS 压至 **0.3489**，Skel IoU 提升到 **0.0235**；
  - 证明了以流匹配（Flow Matching）为骨架，在高质量标准字条件下完全具备突破 0.60 瓶颈的潜能。
- **归档资产**：
  - 50k 与 80k 步的所有评测海报及样本位于 `docs/106moyi/moyi/results/moyi_top10_rf_4ch/`
  - 16ch 探索位于 `docs/106moyi/moyi/results/moyi_top10_rf/`

### 2. Phase A 微观组件消融的终结 (Phase A Ablation)
- **探索重点**：在 S/2 小模型（33.8M）上系统性尝试微观组件升级（RMSNorm、QK-Norm、RoPE、SwiGLU）。
- **实测结论**：
  - `exp1_rmsnorm` 跑到 10,000 步（累计消化 780 万样本暴露），Strict SSIM 仅为 **0.5408**（较基准线未见本质突破）；
  - **根本裁定**：彻底终结微观组件琐碎消融，确认小模型处于严重的**特征容量瓶颈（Capacity Bottleneck）**，小模型哪怕换上所有现代 Trick，网络也无法承载万字级别的复杂骨架拓扑。

### 3. 模型容量阶梯爬坡 (Capacity Scaling Ladder)
放弃微观 Trick 后，将机器 48 的 48GB 显存推满（全局 Batch 384~580，显存占用 44~46GB），直接验证容量阶梯：
- **Tier 1 (S/2, 33.8M)**：10k 步停留于 0.5408；
- **Tier 2 (Sp/2, 59.2M)**：10k 步冲到 0.5494，20k 步稳健收敛至 **0.5606**；
- **Tier 3 (B/2, 131.0M)**：
  - Hidden=768, Depth=12, 12 Heads，参数量达 1.3 亿；
  - 训练损失 Diff Loss 自 1,000 步起便以 0.007~0.014 的幅度全面压制 Sp/2；
  - 5,000 步落盘指标为 0.5346，处于大模型构建全局笔势的起跑期。

---

## 三、 当前正在运行：Tier 3 (B/2) + 7.8万对称增强 10 小时总决战

### 1. v4 对称笔画粗细数据增强 (Symmetric Dilation / Erosion)
- **原理**：对 top10 的 26,002 张原图，按照确定性哈希生成严格对偶的 $\pm$同幅 变体：
  - `tp`（膨胀加粗）+ `tn`（腐蚀变细），带有细字断裂防护（$\ge 15\%$ 面积）与粗字糊块防护（$\le 3.0\times$ 面积）。
- **规模**：从 26,002 扩增至 **77,823 张图片**（3.0倍扩展），彻底解决字形拓扑与像素粗细的过度绑定。
- **清单**：`train_top10_aug_sym.csv`（77,823 行）。

### 2. 10 小时冲刺配置
- **模型**：`DiT-2Cond-B/2` (131.0M)
- **全局 Batch Size**：**384**（显存占用 ~46 GB，功耗 440W 满载）
- **总训练步数**：**48,000 Steps**（以 1.33 步/秒换算，恰好为 **36,000 秒 = 10.0 小时**）
- **预期目标**：验证“1.31 亿模型大容量 + 对称几何笔画增强”对突破 0.60 瓶颈的复合乘数效应！

---

## 四、 本地回传文件目录索引

```text
docs/106moyi/
├── README.md                           # 本白皮书总报告
├── moyi_experiments_summary.csv        # 全量实验指标结构化汇总表
├── DiT/
│   └── experiments/
│       ├── ablation_phase_a/           # Phase A 消融实验 (RMSNorm 等配置、日志、10k评测)
│       └── capacity_ladder/            # 容量阶梯 (Tier 2 Sp/2, Tier 3 B/2 配置、日志、评测)
└── moyi/
    └── results/
        ├── moyi_top10_rf/              # FLUX 16ch 流匹配历史评测、词表、海报
        ├── moyi_top10_rf_4ch/          # SD 4ch 流匹配 50k & 80k 突破 0.60 的历史评测与海报
        └── eval_full_metrics/          # 骨架 IoU 历史多模型对比评测
```
