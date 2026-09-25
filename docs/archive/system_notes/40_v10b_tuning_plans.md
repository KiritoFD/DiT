# 40 — v10b 调优实验方案（4 个，按信息量/成本排序）

> 2026-09-07。v10b（无 char 两因子：callig 风格 + skel-g 结构）现状与短板梳理，
> 给出 4 个可独立执行的调优实验。v10b 基线：base SSIM 0.8404@85k（未完，曲线仍升）、
> 新字遵循度 0.570（n=30 平台）、参数 32.7M。
> 上承 38（方向 A/B/C）、39（DINO 表否决）。

## v10b 基线配方（当前）

| 项 | 值 |
|---|---|
| 架构 | DiT-2Cond-S/2，use_char_cond=False（无 char 因子），use_glyph_cond=True |
| 骨架注入 | glyph_embedder Conv(4→384) patch2 → 256 token，**输入层 token-add（glyph_inject_layers=0）**，glyph_scale_init 0.4，glyph_drop 0.1 |
| 条件 | callig 128 维单因子（4-way 退化）+ skel-g |
| REPA | **w=0.1，layers=(8,)**（弱） |
| 训练 | batch 320 / compile / no-checkpoint / lr 1.5e-4 / warmup 3000 / 150k 上限 / 早停 |
| eval | pretrain_g（g=GT 骨架 latent，n=100，cfg 0.7） |
| 短板 | ① base 未完(85k)→净代价未定 ② 遵循度 0.57 平台 ③ REPA 弱 ④ 注入单层 |

---

## 方案 A：v10b-cont — 续训补齐净代价（成本最低，先做）

**假设**：v10b 85k 早停过早，曲线仍上行（0.8298→0.8404 持续升）；v10a 封顶在
127.5k，v10b 若续到 135k 可能追平或超越 v10a（0.8476）。

**改动**（config 微调 + resume）：
- resume `85000.pt`，早停放宽（patience 5→8 或 min_delta 0.002 保持，min_steps 10000）
- 其余全不变（batch 320 / lr 1.5e-4 / REPA 0.1 / 150k 上限）

**预期结果**：
- 追平/超越 v10a → **drop char 完全免费**：v10b 定案（开放词汇 + 少 13.8M + base 持平）
- 封顶 ~0.843 → 净亏 ~0.004，代价微小仍可接受

**成本**：~6-8h GPU（85k→135k，2.8 sps）。零代码改动。

---

## 方案 B：v10b-repa — REPA 强化到 v8e 成功配方（base 质量杠杆）

**假设**：v8e 两阶段 REPA w0.5 + layers(8,11) 是最强配方（22.5k 封顶 0.7761 且
早停到位）；v10b 只有 w0.1 layers(8,)，REPA 未充分发挥。REPA 让中间特征对齐
DINO 结构先验，理应加速收敛、提升 base。

**改动**：
- `w_repa`: 0.1 → **0.5**
- `repa_layers`: (8,) → **(8, 11)**（v10b 无 char 因子，与 v8e 两路条件完全可比）
- `repa_warmup` 保持 0（v10b 无）或设 2000（v8e 有）——选 2000 与 v8e 对齐
- 严格早停（v8e 教训：长训伤 base）：patience 5 / min_delta 0.002 / min_steps 10000

**预期结果**：base 峰值 +0.003~+0.008（vs v10b 基线），且收敛更快（50k 内到 0.82+）。
风险：REPA w 过高可能压主任务（v9a 曾 batch320 w0.1 稳定），w0.5 需盯 loss 构成。

**成本**：~8h 全量新训。改动 = config 三行。

---

## 方案 C：v10b-inject — 深层骨架注入 + scale 上调（遵循度/结构保真杠杆）

**假设**：v10b 遵循度 0.57 平台的热点原因——骨架只注入**输入层一次**（token-add），
主干深层无显式结构约束（glyph_inject_layers=0）；v8 系 ControlNet 用
ZeroAdaLNInjection 逐层调制更稳。且 glyph_scale 固定 0.4 偏保守。

**改动**：
- `glyph_inject_layers`: 0 → **4**（或 8，均匀分布在 12 层 block 后，与 ControlNet
  对齐的 ZeroAdaLNInjection 注入）
- `glyph_scale_init`: 0.4 → **0.6**（骨架信号给足）
- 显存：每层约 +150MB（batch192 口径），batch 320 下预计 +0.6-1.2G（18.3G→~19.5G，
  预算内）；若 OOM 降 batch 256

**预期结果**：遵循度 0.57 → **0.62+**（结构保真直接受益）；base 可能微升
（更有效利用 g）。风险：深层注入与 skel-g 训练耦合，需调 glyph_drop 配合
（0.1→0.05 防注入过强后无 g 退化）。

**成本**：~8h 全量新训。改动 = config 两行 + 模型已支持（glyph_inject_layers 参数已存在）。

---

## 方案 D：v10b-data — 骨架数据口径拓宽（遵循度/部署语义杠杆）

**假设**：v10b 训练用 1px GT 实例骨架 latent；遵循度测评用 3px 容差 IoU
（1px 严格 IoU 对细线中心线苛刻已被证明）。若训练数据本身用 3px 骨架或
**混合 1px/3px**，模型对"粗骨架到生成"的映射更稳，部署端（手绘/字库）更贴。

**改动**：
- 新增 3px 骨架 shard（`data/skel/final_skel_latents_fame_3px_v8`，若存在）或训练期
  对 1px latent 做形态膨胀增强（低成本，采样时随机 dilate 1-3px）
- eval g 条件也用 3px（与测评协议对齐）

**预期结果**：遵循度 +0.01~+0.03（3px 容差 IoU 直接受益）。风险：与 17 号文档
"1px GT 优于 3px（step2500 0.7355 vs 0.7288）"结论相反——那是两阶段 ControlNet
场景，v10 单阶段 skel-g 未测过，需 A/B。

**成本**：~1-2h 数据处理 + ~8h 训练。改动 = 数据处理脚本 + config。

---

## 优先级建议

1. **A（续训）**：零成本信息量最大，先做——直接回答 drop char 净代价是否为零
2. **C（注入）**：若 A 后遵循度仍是用户痛点的核心指标，C 是遵循度最大的单杠杆
3. **B（REPA）**：base 质量杠杆，可与 A/C 并行（GPU 空闲时新训）
4. **D（数据）**：A/B/C 结论后再定（3px 与 1px 之争需先小步 A/B）

> 提示：A 可立即 resume 跑（GPU 空闲）；B/C/D 是全新训练号，任一先跑 GPU 独占，
> 其余排队。若 GPU 只有一张，推荐 **A 先跑（8h 内出净代价结论），同时本地准备
> B/C 的 config，A 完直接起 C（遵循度优先）**。