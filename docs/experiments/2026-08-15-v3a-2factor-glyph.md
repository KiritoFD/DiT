# V3-A 二因子 glyph 实验（从头训练）

> 日期：2026-08-15  
> 状态：训练中  
> 决策：script 不是独立第三因子 → `glyph = script×character` 合并为 35130 类，模型降为 `calligrapher × glyph` 二因子

## 动机

- V1（factorized_add 3 因子，20k）：MSE 1.00497 / SSIM 0.45762。
- V2（V1-20k 继续 + latent Canny/Skeleton，24k）：MSE 1.00220 / SSIM 0.4503 —— 结构损失被否。
- 88.15% 的 `(calligrapher,character)` pair 只出现在一种 script；书家-书体互信息 1.527 bits（约 68.3% 的 script 熵）。
- 结论：script 应并入内容条件，而非作为第三个平级 lookup。用户指示：把 script×char 作为 char 处理，从头训练。

## 模型（V3-A 二因子版）

- 主干：`DiT-2Cond-S/2`（depth=12, hidden=384, heads=6, patch=2, learn_sigma）。
- 条件：calligrapher（1011 类）× glyph（35130 类），factorized_add 投影（callig 128 + glyph 192 → 384，LayerNorm+Linear，相加 /√2）。
- 4-way mask（训练时，model.train() 内）：
  - full 60%（callig+glyph 都保留）
  - callig-only 15%（drop glyph → style score s_A）
  - glyph-only 15%（drop callig → content score s_G）
  - unconditional 10%（全 drop → CFG 基准）
- 参数量：39,577,952。
- CFG 采样：`eps_uncond + cfg×(eps_full − eps_uncond)`；评估时另测 pair-composed `s_A + s_G − s_0`。

## 数据

- `5script/train.csv`（147,841 行）新增 `glyph_id = script_id×7026 + character_id` 列；
- eval 集（`clean_unseen_triple_100.csv` 等）同样加列；dataset 的 `y_char` 读 `glyph_id`（无列时回退 `character_id`）；
- factor-balanced sampler 的字符权重按 glyph 计数。

## 训练配置（exp_s_5script_v3a_glyph_cs.json，第二轮：结构损失回归）

用户决策：**不要结构损失子集采样；batch 降到 8，对每一张都算 pixel canny+skel loss，两个权重拉到相当高**。

| 项 | 值 |
|---|---:|
| batch | 8（每张都算结构损失） |
| epochs / steps | 500 / 39,600 |
| lr | 1e-4，cosine，warmup 500，min 0.1 |
| weight decay | 0.02（AdamW） |
| EMA | 0.9999 + warmup |
| sampler | factor_balanced（char/glyph α=0.35，callig α=0.15） |
| pixel canny loss | `w_canny=1.0`，全 batch 每张（Sobel 梯度 L1） |
| pixel skeleton loss | `w_skel=1.0`，全 batch 每张（on-skel + off-skel） |
| REPA | 关闭（w_repa=0） |
| VAE decode | fp32 + gradient checkpointing（`struct_subset=0` = 全量） |
| ckpt | 前 5000 每 1000，之后每 5000 |
| auto-eval | free-sampling DDIM 50 步 cfg=4.0，n=100，clean_unseen_triple_100 |
| 起点 | 从头（pretrained=null, use_lora=false） |

显存/速度 probe（batch 8，全量结构损失）：**峰值 15.69G，1.68 steps/s**（相比 b224 无结构 19.2G/3.53 steps/s——显存更低，速度约为 1/2，因 decode 是每 step 固定开销）。

## 执行日志

- 2026-08-15 19:35：V3-A 无结构损失版拉起（PID 333717），step 1000 MSE=2.0889/SSIM=0.2027，step 2000 MSE=1.6990/SSIM=0.2708。
- 2026-08-15 19:51：用户决策：结构损失回归，batch 8 全量 canny+skel，权重 1.0/1.0。停掉无结构版。
- 2026-08-15 19:56：b8 全量结构损失 probe 通过（15.69G / 1.68 steps/s）。
- 2026-08-15 20:02：正式拉起 exp_s_5script_v3a_glyph_cs.json（PID 336762），loss 快速下降中。

## 结果

（训练中，待补）
