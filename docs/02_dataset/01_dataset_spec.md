# 50k 高保真书法数据集规约与评测切分

## 1. 数据集概览

Callig-DiT 训练所依托的核心数据集为 **50k 高保真书法切片库**（元数据清单：[`assets/train_50k_v2.csv`](file:///g:/GitHub/DiT/assets/train_50k_v2.csv)）：
- **样本总数**：$50,000$ 例精细对齐的双通道书法字（汉字图像 + 标准规范骨架）；
- **书法家总数**：$45$ 位中国历代经典书法大家（王羲之、米芾、颜真卿、苏轼、黄庭坚、赵孟頫、文徵明、董其昌、伊秉绶、徐渭等）；
- **书体覆盖**：楷书（正楷、小楷、魏碑）、行书（行楷、行草）、草书、隶书四大正统书体；
- **图像格式**：$256 \times 256$ 灰度单通道（通过 VAE 编码为 $4 \times 32 \times 32$ 潜变量 Shards 存储）。

---

## 2. 书法家词表与词表映射

45 位书法家拥有独立的类别索引，统一映射定义于 [`assets/callig_id_map_50k.json`](file:///g:/GitHub/DiT/assets/callig_id_map_50k.json)。
- **嵌入层参数**：`num_calligraphers: 45`，`callig_embed_dim: 128`；
- **预训练权重**：初始化自自监督表征预训练权重 [`assets/callig_emb_pretrained_50k.pt`](file:///g:/GitHub/DiT/assets/callig_emb_pretrained_50k.pt)。在主干训练中保持冻结（`freeze_callig_table: true`），确保风格流形坐标系不发生漂移；下游任务通过线性投影头自适应适配。

---

## 3. 评测集与泛化基准设定

为了严谨衡量模型的真实泛化能力，杜绝“字形记忆过拟合”，系统划分了两个互斥的评估基准：

### 3.1 Seen 评测集 (In-Distribution Validation)
- **定义**：训练集中出现过的书法家与汉字组合，但抽取独立样本进行重构；
- **规模**：$N = 20$；
- **元数据**：[`assets/eval_v13_seen.csv`](file:///g:/GitHub/DiT/assets/eval_v13_seen.csv)；
- **目的**：测试模型对已知风格与字形的表达上限与重构保真度（Benchmark: SSIM $\approx 0.5420$, PSNR $\approx 21.5$）。

### 3.2 Strict 评测集 (Zero-Shot Out-of-Distribution Generalization)
- **定义**：**严格字形零样本外推**——该汉字在该书法家的训练历史中从未出现过；
- **规模**：$N = 250$；
- **元数据**：[`assets/eval_v13_strict.csv`](file:///g:/GitHub/DiT/assets/eval_v13_strict.csv)；
- **目的**：考核模型在面对陌生复杂汉字拓扑时，能否准确将书法家的笔势形变规律迁移到新字上（Benchmark: SSIM $\approx 0.5280 - 0.5760$）。

### 3.3 Few-Shot 迁移评测集
- 针对沈周、伊秉绶、傅山等低资源书法家，建立 100 样本的 Few-Shot 微调评测子集，评估模型在 1000 - 3000 步极低样本下的风格迁移收敛速度。
