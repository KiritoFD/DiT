# 17 · 训练层：代际演进与 V8 三阶段链

## 1. 预训练代际（基模）

| 代 | 模型 | 数据 | 扩散 | 结果（各自口径） | 教训 |
|---|---|---|---|---|---|
| s2-s11 | pixel DDPM S/B | top6/top30 | eps | 0.49-0.73（pixel 口径） | 结构损失有害；加宽无益 |
| s12-s14 | latent DDPM S/XS | 3top30 | eps | 0.489@60k | 转_latent 成功 |
| s15/s17 | latent flow S/WS | 3top30 | flow | 0.539/0.533 | flow > ddpm |
| s18 | latent flow S | top6 1.1万 | flow | 0.5476 | 数据太少 |
| s19 | latent flow S | mid-clean 11.9万(含增广) | flow | 0.5222 | 增广无益 |
| **s20/s21** | **v2 架构 rms/swiglu/qk_norm/RoPE** | **mid-common / fame** | **flow** | **0.5294 / 0.5121+** | **新架构 +0.013，用 2.4 分之 1 步数** |

架构升级三件套（RMSNorm + SwiGLU + QK-norm + 2D axial RoPE）的贡献：
+0.013 SSIM（s20 vs s19 同口径对比），且训练步数减半即超越。

## 2. V8 三阶段链（现行）

```
v8a_s30_base (预训练 132.5k, fame 数据)
  → A_main_final.pt
    → v8b_s31_ctrl (ControlNet, 1px 骨架条件)   ← 运行中
      → v8c_s32_repa (REPA finetune)            ← 排队
```

| 阶段 | 关键配置 | 状态/结果 |
|---|---|---|
| v8a | DiT-2Cond-S/2, batch 192, lr 1.5e-4 cosine, 132.5k early-stop | SSIM 0.5121（eval_fame_strict_clean_v8） |
| v8b | ControlNet modulate 注入, 1px 骨架条件, batch 72, lr 3e-4, resume 支持 | ΔSSIM +0.221 @7.5k（持续上行） |
| v8c | REPA finetune | 排队 |

### v8b 关键配置
- 条件: `cond_drop_all=0.05 / one=0.30 / which_glyph=0.85 / struct=0.1`
- 骨架条件: `final_skel_latents_fame_1px_v8/`（1px，非 3px）
- 注入: `injection=modulate`（ZeroAdaLN 零初始化）
- eval: `eval_fame_strict_clean_v8.csv` n=100, cfg0.7, 每 2500 步

## 3. ControlNet 代际

| 代 | 注入方式 | 骨架 | 结果 | 教训 |
|---|---|---|---|---|
| fame-ctrl (GT 3px) | ZeroAdaLN modulate | GT 书法骨架 | **SSIM 0.8045**, Δ+0.304 | 基线成功 |
| fame-ctrl-stdskel | 同上 | 标准字库骨架 | **未训通** Δ+0.0015 | **冗余条件被门控忽略** |
| **fame-1pix (GT 1px)** | 同上 | GT 1px 骨架 | **SSIM 0.7974@50k 仍上行** | **1px > 3px** |
| **v8b** | 同上 + v8a 新基模 | GT 1px 骨架 | **Δ+0.2997@50k（100k 上限）** | **现行最强** |

### ControlNet 三定理（实验实证）
1. **条件域匹配**：训练/推理骨架风格必须一致，跨域即失效。
2. **cfg ≤ 1.0**：骨架条件下 CFG>1 单调劣化（1.7→0.7: SSIM 0.683→0.752）。
3. **冗余条件被门控忽略**：骨架信息可由字 ID 推出时注入分支学不到——
   需要 char-dropout 大幅提高或结构信息不可推断的条件源。

## 4. 基础设施代际

| 设施 | 演进 |
|---|---|
| eval daemon | 按系列手工启动（路径 bug 3 次）→ **universal_metrics_daemon.py**（递归扫描 + supervisor 自动重启） |
| ckpt 管理 | ckpt_keep=0 全量堆积 → **best-2 收敛策略**（cleanup_old_experiments.py，两轮释放 ~200G） |
| 极性处理 | cleaner 内特调（不可靠）→ **数据层独立翻转 pass** + 人工标注兜底 |
| 前端 | flask_app → **gradio_fame_local.py**（参数滑块 + MCCD GT 对照 + ZERO-SHOT 标注） |
| 训练 resume | `--resume` + `max_steps` 可扩（cosine 按 new max_steps 重算） |

### 坑清单（已修复，防复发）
1. ckpt 只存 `ctrl_encoder.*` 丢 `injections.*` → 致命，已修
2. daemon 相对路径 → 永不消费 pending → 已修（绝对路径 + universal 递归）
3. eval CSV 组合泄漏 → fame eval 已改严格凸包
4. `np.where(ink,...)` change-detection 恒 False（全 0 changed）→ 已修
5. DataLoader collate numpy→tensor（astype 前需 .numpy()）→ 已修
6. tmux/pkill 自匹配 → 用 `[b]racket` 模式规避
