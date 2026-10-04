# 55. 自条件推理 (Self-Conditioning) 与条件噪声增强（2026-09-13）

> **背景与核心矛盾**：
> 1. GT 实例骨架是作弊（推理期不可得），真实可部署条件下只能使用通用印刷体标准骨架 $g$（楷书等）。
> 2. 标准骨架与真实书法字的 GT 骨架存在天然结构偏差（$\text{cos}=0.902, \text{nmse}=0.197$）。由于标准骨架缺失书家个性化结体，可部署 strict SSIM 被锁定在 ~0.57 附近。
> 3. 12 通道联合扩散（`image(4ch) + canny(4ch) + skel(4ch)`）已在训练期让网络学习预测目标字形和骨架。推理期若直接将辅助通道丢弃，便浪费了模型自主预测的书家化骨架结构。

---

## 1. 自条件推理 (Self-Conditioning Inference)

### 1.1 原理与机制

在 12ch 联合扩散模型中，骨架通道（channels 8–11）的去噪目标是真实书家作品的骨架 latent。
模型在第一遍采样完成时，其输出的骨架通道实际上是：**模型预测该书家书写该字时的骨架形态**（即带书家结体风格的骨架）。

**两遍自条件流程**：

```
[Pass 1: 粗结构推断]
标准印刷骨架 g ──▶ 12ch Flow 采样 (可设步数 20~50) ──▶ 输出 x0_pass1 (12ch)
                                                        │
                                                        ▼
                                                提取 skel 通道 (8:12)
                                                = 预测的书家化骨架 g_pred
                                                        │
[Pass 2: 精细渲染]                                       ▼
混合骨架 g' = (1-α)·g_pred + α·g ──────────────▶ 12ch Flow 采样 (50步) ──▶ 取前 4ch 解码为最终图像
```

### 1.2 关键优势
- **零额外训练**：完全在推理侧实现，直接适配现有已收敛的 12ch checkpoint（如 `v11_pretrain_M432_adaln4_sym`）。
- **计算开销可控**：Pass 1 可使用较少 ODE 步数（例如 20 步 Heun），总延迟仅增加 ~1.4×；若两遍均为 50 步，开销为 2×。
- **可调节混合系数 $\alpha$ (`blend_alpha`)**：
  - $\alpha = 0.0$：完全信任模型预测骨架；
  - $\alpha = 0.3 \sim 0.5$：兼顾标准骨架拓扑稳定性与书家结体特征；
  - $\alpha = 1.0$：退化为单遍基线采样。

### 1.3 代码实现
- **推理函数**：[`src/eval/inference.py`](file:///g:/GitHub/DiT/src/eval/inference.py) 中的 `sample_latents_self_cond()`。
- **A/B 测试脚本**：[`tools/eval/eval_self_cond.py`](file:///g:/GitHub/DiT/tools/eval/eval_self_cond.py)。

---

## 2. 条件噪声增强 (Condition Noise Augmentation)

### 2.1 动机
标准骨架 $g$ 在数据集中是固定的离散字形 latent。模型在数十万步训练中极易对 $g$ 的精确连续数值产生记忆与过拟合。
一旦推理期输入的骨架与训练期标准模板产生细微差异（例如自条件生成的骨架、不同字体字帖骨架、手写草图），过拟合模型会表现出脆弱的泛化性能。

### 2.2 扰动策略
训练期对输入的条件张量 $g \in \mathbb{R}^{B \times 4 \times 32 \times 32}$ 施加随机扰动：
1. **连续高斯扰动**：
   $$g' = g + m \cdot \epsilon \cdot \sigma, \quad m \sim \text{Bernoulli}(p_{\text{noise}}), \quad \sigma \sim \mathcal{U}(0, \sigma_{\max}), \quad \epsilon \sim \mathcal{N}(0, I)$$
2. **空间 Patch Dropout**：
   $$g'' = g' \odot M_{\text{spatial}}, \quad M_{\text{spatial}} \sim \text{Bernoulli}(1 - p_{\text{drop}})$$
   迫使 DiT 块利用 RoPE 与全局自注意力补全局部缺失的骨架拓扑。

### 2.3 配置参数 (`src/train/train.py`)
| 参数 | 类型 | 默认值 | 推荐值 | 说明 |
|---|---|:---:|:---:|---|
| `glyph_noise_scale` | float | 0.0 | 0.10 | 噪声标准差上限 $\sigma_{\max}$ |
| `glyph_noise_prob` | float | 0.0 | 0.30 | 样本被加噪概率 $p_{\text{noise}}$ |
| `glyph_patch_drop` | float | 0.0 | 0.05 | 空间 token 丢弃概率 $p_{\text{drop}}$ |

默认全为 0.0，向后完全兼容已有全部配置文件。

---

## 3. 验证与 A/B 测试方案

### 3.1 远端 4090 快速验证
使用工具脚本直接评估已有 checkpoint：
```bash
python tools/eval/eval_self_cond.py \
    --ckpt assets/results/v11_pretrain_M432_adaln4_sym/checkpoints/0042500.pt \
    --config src/train/configs/v11_pretrain_M432_adaln4_sym.json \
    --eval-csv assets/eval_seen_v10.csv \
    --skel-dir data/skel/std_skel3_latents_fame_sym \
    --n 50 --device cuda \
    --first-pass-steps 20 \
    --blend-alphas 0.0,0.2,0.5
```
对比基线单遍采样与多组 $\alpha$ 的 SSIM、MSE 及逐样本改善率。
