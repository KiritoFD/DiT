# P1 实现说明：FM + 图像域重建（冻结 Decoder）

**日期：** 2026-11-02  
**Spec 生成：** `tools/d_20261101/make_p1.py`  
**Objective：** `src/unet/joint/objective.py` 新增 `P1` arm  
**状态：** 已实现、已通过预检、**尚未启动训练**

## 目标函数

$$L = L_{\mathrm{FM}} + \lambda\, L_{\mathrm{image}}\big(D(\hat z_0),\, x_{\mathrm{GT}}\big)$$

- VAE **冻结**，U-Net **解冻**，图像损失**穿过冻结的 Decoder** 回传到生成器。
- $L_{\mathrm{image}}$ = **纯 L1**。不含 DINO、Gram、clDice、edge。
- 保留完整 FM 与 Writer 条件。

## 与 GARU 的区别（这是本臂的意义所在）

| | 生成器 | Decoder | 图像目标 |
|---|---|---|---|
| GAR | **冻结**（368 张量逐位不变）| 训练 | MSE+LPIPS+DINO+10×Gram |
| GARU | 解冻 | **也训练** | 同上（复杂）|
| **P1** | **解冻** | **冻结** | **纯 L1** |

GARU 同时训练 Decoder，因此它的任何改善都无法归因到生成器；
P1 冻结 Decoder，**唯一能动的就是生成器**，因果可归因。

同时 P1 刻意用**最简单的图像损失**：GAR/GARU 已经证明复杂感知栈修不了断笔，
所以先问最小的问题 —— **图像域重建监督能否到达生成器、能否改善结构**。

## 预检结果（`check_spec_init.py`）

```
spec P1.json
  transfer_style=True calibrate_gains=False
  freeze_backbone=False skip_fm=False
  joint.arm='P1'
  as-built from spec: 368 tensors, max|diff|=0.000e+00 at -, offenders=0
  trainable tensors: 362/365
OK: the arm starts exactly at K and has the intended trainable surface
```

**`max|diff|=0.000e+00`** 证明带上了 `transfer_style=True`，
没有重蹈 D1/D2/D3「K 主干 + 随机风格分支」的覆辙。
**362/365 可训练** 证明生成器确实解冻（GAR 是 0 个可训练生成器张量）。

## 代码改动

`src/unet/joint/objective.py`：

| 项 | 值 | 说明 |
|---|---|---|
| arm 白名单 | 加入 `'P1'` | |
| `self.image` | True | 走图像分支 |
| `self.train_decoder` | **False** | 冻结 Decoder |
| `self.detach_image` | False | 梯度必须回到生成器 |
| `self.train_generator` | **True** | 与 GARU 同 |
| `self.simple_image` | **True** | 仅 P1；`image_loss` 返回纯 L1 |
| `self.reconstruction` | False | 不做额外的 latent 重建项 |

## 权重 λ 不做猜测

`λ` 必须由**梯度审计**决定，不能凭 loss 数值小就加大。审计要测三件事（每 t）：

1. **可达性**：图像损失是否真的在 U-Net 参数上产生梯度（意外的 detach / no_grad 会静默给 0）；
2. **尺度**：`|g_image| / |g_FM|`，建议初始扫描 **5%–20%**；
3. **夹角**：`cos(g_image, g_FM)`。接近 0 表示它在补一个新方向（可能有用）；
   **强负相关表示它在和 FM 对抗** —— 那正是笔画模糊、多样性下降的机制。

**注意高 t 的陷阱：** 高 t 时噪声几乎不含 GT 信息，
强行施加图像 L1 会逼生成器预测条件均值（模糊 + 多样性下降）。
margin 大 ≠ 应该在 t=1 加强监督。

**不再照搬 `t≤0.3`：** 先前审计（`TS_WINDOW_AUDIT.md`）显示结构信号在最弱区间，
旧门限不能继承。时间窗必须由上面的审计数据决定。

## 验收要求（缺一不可）

按 `frag` 单独验收是**不安全**的 —— 擦掉笔画会降低连通块数、降低 frag，
看起来像修复（这正是 P0.5 首版被骗的机制）。所以必须同时报：

1. **frag 的逐字变化**，且分开 `component_excess` / `component_deficit`；
2. **墨量保留率** `ink_retention`（擦除的守卫）；
3. **OCR 与 E-G**；
4. **latent 优化距离**，并测**优化后 latent 的 VAE round-trip 稳定性**，
   排除「靠偏离正常 latent 流形」获得的虚假收益。

对照必须是**相同起点（K）、相同数据、相同步数的纯 FM**（严格配对）。

## 尚未做的事

- **未启动训练。** 需先完成 A/C/B 收敛对照与梯度审计。
- **梯度审计脚本**（`grad_audit_p1.py`）尚未跑通：它需要真实的条件
  （writer / script / STD），而这要走训练器的数据路径；
  独立脚本里造假条件正是早先 `GT/latent crop or VAE mismatch` 的成因。
  **正确做法**是在真实训练路径上加一个 `--probe-grad` 模式，而不是另写一个假数据脚本。
- **GARU 不启动**（结论未变，且 P1 是它的受控替代）。
