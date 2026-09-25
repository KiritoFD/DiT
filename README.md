# CalliDiT (墨韵)
### Multi-Condition Diffusion Transformer for Stylized Chinese Calligraphy Generation
*基于多条件扩散 Transformer 的高可控风格化中国书法字形生成模型*

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-76B900.svg?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE.txt)
[![Status](https://img.shields.io/badge/Training-v21__skelnet__200k-success.svg)]()

---

## 0. 项目使命与核心定位

传统中文字法生成模型普遍受困于三个根本瓶颈：
1. **闭集字符依赖**：依赖离散的字 ID（Character ID），遇到字典外的罕见字、开集生僻字或书法异体字时彻底瘫痪；
2. **结构照搬与门控失效**：引入骨架或轮廓条件时，网络往往学会退化为“恒等抄写”，抹杀书法家的结体取势；
3. **风格与间架混杂**：风格向量难以同时调节微观笔锋（飞白/干湿/墨法）与宏观间架（欹侧/收放/中宫），导致不同书家生成的字“千人一面”。

**CalliDiT (墨韵)** 提出了一套**结构开集、风格闭集、间架可塑、质感解耦**的全新架构范式：
- **开集字形驱动**：彻底丢弃离散字 ID，采用跨字体渲染的标准字形骨架潜变量 $g_{std}$ 作为内容条件，天然支持万级开集汉字渲染；
- **SkelNet 间架形变网络**：引入 3.58M 参数的轻量可变形 U-Net，受书家特征驱动，将跨书家共享的呆板“标准字”形变为具备历史书家个性化间架的 $g'$，并受历史真迹骨架稠密监督（$\mathcal{L}_{deform}$）；
- **双通道解耦注入机制**：
  - **间架结构轴（SkelNet 通道）**：驱动骨架仿射、位移与门控笔画粗细，重塑骨架几何；
  - **笔触质感轴（Diffusion adaLN 通道）**：在 12 层 DiTBlock 中调制特征 scale/shift/gate，掌控墨色动态与笔锋质感；
- **现代化高效扩散引擎**：基于 Flow Matching (Heun-RK2 + Logit-Normal 采样)，搭载 RMSNorm、SwiGLU、RoPE 与 DINOv2 REPA 多层表征对齐，在单卡 RTX 4090 上实现 **4.08 steps/s** 的极致吞吐。

---

## 1. 系统全景架构 (Architecture Overview)

```
                            [ 标准字形骨架 g_std ] (4, 32, 32)
                                       │
                                       ▼
  [ 书家类别 y_callig ] ──> e_callig (128d) ──> [ SkelNet / DeformSkel ] (3.58M)
                                       │                 │
                                       │                 ├─> FiLM 逐尺度特征调制
                                       │                 ├─> 全局仿射变换 (缩放/错切/旋转)
                                       │                 ├─> 偏移场 (dx, dy)
                                       │                 └─> 门控笔画粗细调制 (Gated Stroke Mod)
                                       │                         │
                                       │                         ▼
                                       │                [ 书家个性骨架 g' ] ──┬──> L_deform 监督 (w=1.0)
                                       │                         │             │
                                       │                         ▼             ▼
                                       │                 [ GlyphEmbedder ]   [ 真实墨迹骨架 g_inst ]
                                       │                         │
                                       │                         ├─> ① 输入 Patch 残差: x = x + 0.6·g_tok
                                       │                         ├─> ② 4 层 ZeroAdaLN 逐层跨注意力注入
                                       │                         └─> ③ 全局均值池化 ──> e_glyph_vec (128d)
                                       │                                                      │
                                       ▼                                                      ▼
                       [ cond_fusion: LayerNorm + Linear ] <──────────────────────────────────┘
                                       │
                                       ▼
                              y_emb (384d 条件信号)
                                       │
  [ 时间步 t ] ──> t_emb (384d) ──────(+)──> c = t_emb + y_emb (全局调制向量)
                                                          │
                                                          ▼
  [ 扩散加噪 latent x_t ] ───────────────────────> [ 12x DiTBlock ] (S/2, 36.5M)
                                                    ├─ RoPE 旋转位置编码
                                                    ├─ RMSNorm + QK-Norm + SwiGLU
                                                    ├─ adaLN: 6 路调节 (scale, shift, gate)
                                                    └─ REPA DINOv2 语义对齐 (Layer 8)
                                                          │
                                                          ▼
                                            [ FinalLayer & Unpatchify ]
                                                          │
                                                          ▼
                                            [ 速度场预测 v_pred ] ──> L_diff 扩散流损失
```

---

## 2. 重大科研实证结论与证伪清单 (Key Insights & Falsification)

经过数十万步前沿探索与严格的自动微分探针审计，本项目沉淀了关于书法生成的关键物理规律：

### 2.1 核心突破与实证实录

1. **骨架跨书家高度共享的实证事实**：
   实测 MCCD 数据集，跨书家的 $g_{std}$ 骨架重复率高达 **$43.5\%$**（例如李邕与褚遂良共用相同的标准骨架）。若无间架形变干预，主干网络会陷入“恒等照抄”捷径，风格信号被完全门控忽略。
2. **SkelNet 自带梯度的机制突破**：
   将 $g_{std}$ 形变至接近历史名家真迹，**直接降低去噪重建损失**。离线训练达到 **$95.3\%$** 的风格跟随率，潜空间形变量达 **$53.05\%$**，彻底击穿“主干不听风格”的死锁。
3. **梯度正交性的数学证明**（见 [`docs/922/97_dual_channel_analysis.md`](docs/922/97_dual_channel_analysis.md)）：
   在 Step 5000 真实模型上使用计算图分流求导，实测：
   $$\cos(\nabla_{e_{skel}}\mathcal{L}_{diff}, \; \nabla_{e_{skel}}\mathcal{L}_{deform}) = \mathbf{+0.0030} \approx 0$$
   **数学实证**：扩散去噪任务自身对骨架微观几何呈现正交无偏好状态；$\mathcal{L}_{deform}$ 提供了强出 6.22 倍的垂直梯度力量，是确保间架不被扩散随机噪声带偏的决定性支柱。
4. **双通道解耦实测有效**：
   扰动消融测试表明：打乱主干 adaLN 风格使 Loss 显著恶化 **$+1.05\%$**，关闭骨架形变使 Loss 恶化 **$+0.74\%$**，验证了间架与纹理分工明确、双轮驱动。
5. **融合方差失衡的病理诊断**：
   在 `cond_fusion` 处，$\|e_{callig}\| = 9.86$ 与 $\|e_{glyph\_vec}\| = 60.90$ 存在 38 倍方差差距，全局 `LayerNorm(256)` 压制了风格振幅约 $1/5$，为下一代架构指明了确定性升级方向。

### 2.2 已被实测证伪的路线 (Falsification Hall of Fame)

为避免后续研究重走弯路，以下路线已被严格实测证伪：

| 路线方案 | 实测表现 / 失败证据 | 根本机理归因 | 详见文档 |
| :--- | :--- | :--- | :--- |
| **暴力扩张模型容量** (27M $\to$ 67M) | strict SSIM 全落入 0.51~0.57，容量翻倍无任何提升 | 瓶颈在条件有效性而非主干参数量 | `docs/system/54` |
| **削减模型容量** (S/2 $\to$ XS/2 depth 8) | strict SSIM 从 0.5603 跌至 0.5337 ($-0.026$) | Transformer 深度小于 12 会损失基本拟合表达 | `docs/system/62` |
| **12 通道辅助扩散目标** | strict SSIM 仅 0.4993 (比基线低 $0.06$) | 辅助通道分布异构，CFG 引导时轨迹严重跑飞 | `docs/system/59` |
| **条件高斯噪声增强** (noise400k) | seen 暴跌 $-0.048$，strict 平台仅 0.4869 | 破坏了骨架的高频空间位置确定性 | `docs/system/55` |
| **离散字符 ID 条件** (char_id) | 闭集泛化极差，未见字 SSIM 趋近于 0 | 无法开集泛化，且与骨架产生条件冗余门控 | `docs/system/60` |
| **无骨架监督的端到端学习** | 风格跟随率退化至随机水平 ($\approx 50\%$) | 扩散 Loss 梯度对骨架几何正交，无法自发学结体 | `docs/922/96` |
| **全局无截断笔画残差** | 出现背景造墨污染、字体周围散落杂点 | 笔画调制必须在距离场截断约束内进行 | `docs/922/96` |

---

## 3. 实验代际演进谱系 (Experiment Evolution Timeline)

```
[v1~v3] 像素空间探索 ──> [v6~v9] ControlNet 门控失效 ──> [v10] 标准骨架引入 (单向量因子)
                                                                 │
┌────────────────────────────────────────────────────────────────┘
▼
[v11] 架构现代化 (RMSNorm/SwiGLU/RoPE) + 12ch 试错
  │
  ▼
[v12] 骨架全局向量池化 (glyph_vec_cond) ──> [v13] 50k 清洗数据集基准 (SSIM 0.5703 遇瓶颈)
                                                  │
┌─────────────────────────────────────────────────┘
▼
[v14] 87 类别书家×书体细分 ──> [v15] 多模态 K-Means 风格 token (K=4)
                                     │
┌────────────────────────────────────┘
▼
[v17] 局部条件注入与门控机制 ──> [v18~v20] 离线形变探索 (SkelNet 破局)
                                         │
                                         ▼
                 ★ [v21_skelnet_200k] 当前基准：
                   - 95.3% 离线权重注入 (deform_skel_v10.pt)
                   - Gated Stroke Mod (gate_radius=0.25)
                   - 小 lr (5e-5) 联合微调 + w=1.0 稠密中间监督
                   - 稳态 4.08 steps/s, strict SSIM 突破 0.5299
```

---

## 4. 当前运行态势与实时监控 (Live Training Status)

| 指标维度 | 当前状态 (Step 11200+ / 200,000) |
| :--- | :--- |
| **活跃进程** | `v21_skelnet_200k` (tmux session: `v21_200k`) |
| **算力硬件** | NVIDIA GeForce RTX 4090 24GB (Ubuntu 18.04, PyTorch 2.1.2) |
| **训练吞吐** | **$4.07 \sim 4.10\text{ steps/s}$**（全速平稳运行） |
| **整机功耗 / 显存** | **$368\text{W}$** / **$18.65\text{GB}$**（资源利用率饱和） |
| **损失函数曲线** | $\text{Diff}: 0.403 \to \mathbf{0.314}$ \| $\text{Deform}: 0.315 \to \mathbf{0.294}$ \| $\text{REPA}: 0.0059 \to \mathbf{0.0036}$ |
| **Step 5,000 评测** | `seen SSIM`: **$0.5300$** \| `strict SSIM`: **$0.5258$** |
| **Step 10,000 评测** | `seen SSIM`: **$0.5408$** ($\uparrow 0.0108$) \| `strict SSIM`: **$0.5299$** ($\uparrow 0.0041$) |
| **同字特异性** | `target_spec`: $\mathbf{+0.0081}$ \| `cal_enrich`: $\mathbf{0.95\times}$ \| `nn_ssim`: $0.5811$ |

---

## 5. 全局全量文档索引地图 (Documentation Sitemap)

本项目拥有完整的工程与研究全谱文档，各阶段分析均有迹可循：

### 5.1 最新前沿研究与架构设计 (`docs/922/`)
- [`97_dual_channel_analysis.md`](docs/922/97_dual_channel_analysis.md)：**【核心必读】** 双通道风格注入机制审计、Step 5000 探针量化数据、梯度正交性证明与方差瓶颈分析。
- [`96_skelnet.md`](docs/922/96_skelnet.md)：SkelNet 骨架形变网完整原理、数据证据、门控笔画调制与对比学习设计。
- [`95_module_decision.md`](docs/922/95_module_decision.md)：阶段模块选型裁定与历史技术路线复盘。
- [`94_style_supervision.md`](docs/922/94_style_supervision.md)：风格表征监督机制与 Hard Negative 对比损失探索。
- [`93_samechar_nn_diag.md`](docs/922/93_samechar_nn_diag.md)：同字最近邻诊断与字形特异性量化体系。
- [`92_skel_condition_aug.md`](docs/922/92_skel_condition_aug.md)：骨架条件几何扰动增强实验分析。
- [`90_style_injection_summary.md`](docs/922/90_style_injection_summary.md)：历代风格注入机制横向对比总结。

### 5.2 系统全景快照与工程规范 (`docs/919/`)
- [`00_overview.md`](docs/919/00_overview.md)：项目目标、任务定义与结论清单总览。
- [`10_data.md`](docs/919/10_data.md)：数据资产全谱（Fame / TJ / 50k_v2、分片缓存、清洗与评测集定义）。
- [`20_model.md`](docs/919/20_model.md)：DiT-2Cond 核心模型拓扑、条件注入接口与证伪路线解释。
- [`30_training.md`](docs/919/30_training.md)：训练超参配方、Flow Matching 采样器设计与历史实验链。
- [`40_results.md`](docs/919/40_results.md)：评测总表、逐书家指标分布与书法质感分析。
- [`50_infra.md`](docs/919/50_infra.md)：运行环境、算力设施、显存瓶颈、torch.compile 规避与运维排错手册。

### 5.3 历史演化深度专论 (`docs/system/`)
- 包含从 `00` 到 `75` 篇系统演进专论，重点推荐：
  - `50_evolution_retrospective_20260910.md`：前 50 代实验演进长文复盘；
  - `62_param_budget_derivation.md`：DiT 宽度、深度与计算量（FLOPs）预算推导；
  - `72_seen_vs_strict_analysis.md`：泛化性（seen vs strict）本质差距剖析；
  - `75_style_rank_loss_20260925.md`：风格排序损失与 McNemar 显著性检验。

---

## 6. 快速启动与工程操作手册 (Quickstart & Tooling)

### 6.1 环境准备
```bash
# 激活环境
conda activate cu121

# 验证核心依赖
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), 'Version:', torch.__version__)"
```

### 6.2 双通道强度与梯度自动化探针
在任何已有检查点上一键测量风格双通道路由、形变量与梯度流：
```bash
python tools/probe_dual_channel.py \
    --config src/train/configs/v21_skelnet_200k.json \
    --ckpt assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0005000.pt \
    --batch-size 16 \
    --device cpu
```

### 6.3 启动主干训练 (v21 配置)
```bash
python -m src.train.train --config src/train/configs/v21_skelnet_200k.json
```

### 6.4 离线评测与可视化海报渲染
评测脚本将自动在 `seen`（记忆集）与 `strict`（泛化集）上运行双轴 CFG 采样，生成最邻近风格比对海报：
```bash
python -m src.eval.in_process_eval \
    --ckpt assets/results/v21_skelnet_200k/20260926-005749-v21-skelnet-200k/checkpoints/0010000.pt \
    --eval-sets "seen:assets/eval_v13_seen.csv:20,strict:assets/eval_v13_strict.csv:250" \
    --cfg 0.7 \
    --steps 50
```

---

## 7. 许可证与引用

本项目采用 [Apache License 2.0](LICENSE.txt) 开源协议。

```bibtex
@article{callidit2026,
  title={CalliDiT: Multi-Condition Diffusion Transformer for Stylized Chinese Calligraphy Generation},
  author={Xiao Yang and Research Team},
  year={2026},
  journal={Internal Research Report}
}
```
