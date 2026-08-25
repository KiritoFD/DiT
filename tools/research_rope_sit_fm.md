# RoPE / SiT / Flow-Matching 调研结论（针对 DiT-2Cond-S/2, 46M, 256-token 场景）

## 场景特征
- 46M 小模型, 序列 16×16=256 token（固定、短、无外推需求）
- 当前: DDPM ε-prediction + DDIM 50 步采样 + 冻结 2D sincos PE + adaLN-zero
- 600K 步训练计划

## 1. RoPE — 收益最小，不建议
- 核心价值在长序列外推（LM 场景），我们固定 256 token 无外推需求
- 短 2D patch 序列上无权威证据优于 2D sincos（ViT/DiT 主流都不用 RoPE）
- 2D-RoPE 分组变体多，易写错，改完可能不升反降
- **结论：排除。继续用冻结 2D sincos**

## 2. SiT — 收益中等，作为采样器参考
- interpolant 框架本质 = Flow Matching 家族（DDPM/FM/RF 都是特例）
- ImageNet 256: SiT-XL FID ~2.04 vs DiT-XL 2.27（同参数量提升可观）
- 提升主要来自训练目标 + EDM 风格采样器，不需更多训练步
- 46M 小模型上的迁移性未验证
- **结论：不必单独做，作为 FM 的采样器调优参考**

## 3. Flow Matching / Rectified Flow — 最值得引入
- RF/FM 是当前 SOTA 事实标准（SD3、Flux 都用）
- 核心收益: 轨迹拉直 → 少步采样（25-50 步 Euler 即可高质量）+ 训练更稳定
- 收益不依赖模型规模（几何性质），比 SiT 更可信迁移到小模型
- 与 adaLN-zero / 2Cond / sincos 全正交，实现路径成熟（SD3/diffusers 参考多）
- **结论：最高性价比单点升级**

## 实施路径（分阶段）
- 阶段 0: 固定当前 DDPM+DDIM50 作 baseline 快照
- 阶段 1: 训练改 x_t=(1-t)ε+t·x0, v=x0-ε, MSE(net, v)；采样换 Euler ODE（50 步起步）
- 阶段 2: logit-normal 时间权重 + Heun 二阶采样器
- 阶段 3（可选）: SiT 式 EDM 参数化（c_skip/c_in/c_out），压步数到 10-20

## 引用
- RoPE: arXiv:2104.09864 (RoFormer)
- Flow Matching: arXiv:2210.02747 (Lipman et al.)
- Rectified Flow: arXiv:2209.03016 (Liu et al.)
- SD3: arXiv:2403.03206 (Esser et al.)
- SiT: github.com/willisma/SiT（arXiv 编号待核实）
- DiT: arXiv:2212.09748
- EDM: arXiv:2206.00364 (Karras et al.)
