# 41 — v10b 梯度诊断：REPA 劫持 + callig 链弱梯度（瓶颈定位）

> 2026-09-07。用 `tools/debug_v10b_gradients.py`（从 85000.pt 加载真模型 + 真分布
> 数据跑 4 batch 梯度）诊断 v10b 瓶颈。用户要求：先 debug 再设计改进。

## 诊断方法

- 从 v10b 85000.pt 加载 EMA 模型（miss=0 unexp=0），MCCDLatentDataset 真数据
- 4 batch × 16，flow matching MSE + REPA（w0.1 layers(8,)）反向传播
- 测：模块相对更新量 grad_norm/param_norm、REPA 梯度占比、loss 数值、g 注入作用

## 结果

### 模块相对更新量 (grad_norm/param_norm, 4 batch 均值)

| 模块 | grad_norm | rel | 解读 |
|---|---|---|---|
| x_embedder | 0.104 | 0.0242 | 输入层，正常最高 |
| **glyph_embedder** | 0.0735 | **0.0076** | 骨架注入中等 |
| t_embedder | 0.041 | 0.0043 | 正常 |
| final_layer | 0.067 | 0.0029 | 正常 |
| block00-11 | 0.044-0.081 | 0.0006-0.0014 | 正常 DiT 水平，随深度略降 |
| **callig_chain** | 0.0216 | **0.0013** | **条件链最弱（低 6 倍）** |
| **repa** | **0.342** | **0.0213** | **最大梯度模块** |

### 关键指标

- **REPA 梯度平方和占比 = 158.5%**（repa_grad_sq / model 全参数 grad_sq）
- **REPA/main loss 数值比 = 0.696**（repa loss 0.096 vs main 0.139）
- **g 注入作用 = 输出差 10.4%**（有/无 g 全网输出相对差，条件生效非死区）

## 瓶颈结论

1. **REPA 劫持优化（首要瓶颈）**：
   - `w_repa=0.1` 看似小，但 REPA 是**余弦 loss（量纲 0~1）**，MSE 是
     **velocity MSE（量纲小）**——两者梯度幅度不可比
   - 实测 REPA 投影头梯度平方和 = 全模型 158%，**优化方向被 REPA 主导**
   - 后果：主干被拉向"DINO 结构对齐"而偏离"velocity 拟合"——v10b base SSIM
     0.8404 上不去、v10a-dino 也受牵连的可能机制
   - **修正方向**：REPA 权重必须按梯度量纲归一（w_repa 0.1→0.01-0.03 或
     w×grad_scale 校准），或换 L2 量纲统一的损失形式

2. **callig 风格链弱梯度（次要瓶颈）**：
   - callig_chain rel=0.0013，比 glyph_embedder 低 6 倍、比 x_embedder 低 18 倍
   - 书家风格条件在训练中更新很慢——风格保真（美感）可能因此受限
   - 修正方向：callig_embed_dim 128→256、callig_scale 初值上调、或
     callig_proj 换更大 MLP

3. **g 注入生效但非主力**：输出差 10% 说明骨架条件在作用，但 glyph_embedder
   rel 0.0076 中等——注入通路不是死区，也不需要大改；重点是别让 REPA 抢走梯度

## 对 v10a-dino 的回看

v10a-dino（冻结 DINO 表 + ln_only）base 0.8348 < v10a 0.8476 的差距，
在 REPA 劫持视角下有了新解释：DINO 表特征秩低（34/384），REPA 对齐到这种
低秩 teacher 特征会把主干带偏更多。**REPA 劫持 + 低秩 teacher = 双重拖累**。

## 下一步（改进实验设计见 42 号文档）

- 实验 1：REPA 权重校准（w 0.1→0.02 + 梯度归一）
- 实验 2：callig 链增强（dim 256 + scale 上调）
- 实验 3：REPA 仅浅层/off（对照）
- 实验 4：两者组合

存档：`5script/results/v10b_debug_grad.json`，工具 `tools/debug_v10b_gradients.py`。