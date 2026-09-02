# Skeleton-ControlNet (1px GT skel) 实验线结果汇总

> 更新时间：2026-09-02
> 本文件记录 **skel-ctrl 1px 实验线** 的所有 checkpoint 位置、已有评测数据、关键结论与已知问题。
> 配套训练代码：`src/train/train_controlnet.py`；评测：`src/eval/eval_ctrl_metrics_daemon.py` + `src/eval/inference.py`。
> **独立 ckpt 评测**：`src/eval/eval_ctrl_ckpt.py`（GPU 采样 + 图片全盘落盘 + CPU 指标，支持任意 ckpt / 数据目录传入）。

## 1. 实验线图谱（base → ctrl）

| 阶段 | 实验 | base 主模型 | skel 条件 | 状态 |
|---|---|---|---|---|
| A | **S21** fame 预训练 | —（自身） | — | ✅ 已早停（best ssim 0.4670 @ 40k） |
| A' | **S30** fame 预训练（DINO真迹字表+char强化） | —（自身） | — | ✅ 已早停（best ssim **0.4841** @ 130k） |
| B | **ctrl_fame_1pix_v1**（旧，S21-based） | `s21/.../0030000.pt` | `final_skel_latents_fame_1px` | ✅ 训练完成（50k 步） |
| B' | **s31_ctrl_gt_skel_1px**（当前，S30-based） | `s30/.../0132500.pt` | `final_skel_latents_fame_1px`（训练）/ **`final_skel_latents_eval_1px`（eval，错误）** | ⚠️ 训练中，eval skel latent bug 已定位 |
| C | **s32 REPA 微调**（计划） | S30 + s31 | — | ⏳ 待 s31 完成 |

## 2. Checkpoint 位置

### S21 base
```
5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0030000.pt   # ctrl 挂载点
5script/results/s21_fame_flow_v2/20260829-232329-s21-fame-flow-v2/checkpoints/0040000.pt   # 最终（早停）
```

### S30 base（当前最优 base）
```
5script/results/s30_dino_char_strong_pretrain/20260901-052520-s30-dino-char-strong-pretrain/checkpoints/0132500.pt
```
- S30 best ssim = **0.4841** @ 130k（比 S21 高 **+0.0171 / +3.7%**）
- S30 ckpt 含 `_orig_mod.` 前缀（compile 存盘），跨脚本加载需剥前缀（`load_main_model` 已处理）

### 旧 S21-based 1px ctrl（ctrl_fame_1pix_v1）
```
5script/results/ctrl_fame_1pix_v1/20260830-205652-fame-ctrl-skel-1px-v1/checkpoints/0050000.pt   # 主批次（更优）
5script/results/ctrl_fame_1pix_v1/20260830-185731-fame-ctrl-skel-1px-v1/checkpoints/0022500.pt   # 早期批次
```
- 配置：`char_proj_mode=ln_only, freeze_char_table=true, batch=72, cond_drop_which_glyph_prob=0.5`
- eval：`eval_fame_strict.csv, n=100, cfg=0.7, steps=50, skel=final_skel_latents_fame_1px`
- 存盘：`ctrl` + `ema`（非 `main.*` 全存，含注入层）

### 当前 S30-based 1px ctrl（s31）
```
5script/results/s31_ctrl_gt_skel_1px/20260901-135832-s31-ctrl-gt-skel-1px/checkpoints/<step>.pt
```
- 配置：`char_proj_mode=mlp, freeze_char_table=true, batch=192, cond_drop_which_glyph_prob=0.85`
- 训练数据 skel：`final_skel_latents_fame_1px` ✅（覆盖训练集 51822）
- **eval skel latent bug**：config `gpu_eval_skel_latent_shards_dir=final_skel_latents_eval_1px`（仅 501 个，eval 500 个 img_id 覆盖 **0/500**）→ eval 时 skel latent 全零

## 3. 已有评测数据（旧口径 eval_fame_strict，n=100）

### 旧 ctrl_fame_1pix_v1（S21-based，**skel latent 正确**）
| step | ctrl.ssim | base.ssim | ΔSSIM | ctrl.mse | 趋势 |
|---|---|---|---|---|---|
| 2500 | 0.7355 | 0.4977 | +0.2378 | — | 上升 |
| 10000 | 0.7368 | 0.4977 | +0.2391 | — | 波动 |
| 20000 | 0.7641 | 0.4977 | +0.2663 | — | 上升 |
| 40000 | 0.7929 | 0.4977 | +0.2952 | — | 上升 |
| 50000 | **0.7974** | 0.4977 | **+0.2997** | — | 持续上升，**无回撤** |

> 关键：旧 ctrl eval 的 skel 条件真实（全量 latent），**ctrl.ssim 一路涨到 0.7974 无回撤**，ΔSSIM 高达 +0.30。

### 当前 s31（S30-based，**eval skel latent 为全零**）
| step | ctrl.ssim | base.ssim | ΔSSIM | ctrl.mse | 趋势 |
|---|---|---|---|---|---|
| 2500 | 0.5432 | 0.4946 | +0.0486 | 0.8716 | 上升 |
| 10000 | **0.5631** | 0.4946 | +0.0685 | 0.8066 | 峰值 |
| 17500 | 0.5434 | 0.4946 | +0.0488 | 0.7943 | 回撤 |
| 30000 | 0.5485 | 0.4946 | +0.0524 | 0.7950 | 波动 |
| 40000 | 0.536 | 0.4946 | +0.0412 | ~0.79 | 回撤 |

> ⚠️ **s31 的 eval ssim 全部不可信**（eval skel latent 为全零张量）。训练 loss 正常下降（0.3153→0.15），但 eval 未收到真实骨架条件。

## 4. 关键结论

1. **旧 ctrl（S21-based, 真实 skel latent）能达到 0.79 且无回撤** → 之前"回撤"是 **eval bug（skel latent 覆盖为 0）** 造成的假象，不是模型问题，也不是 SSIM 指标问题。
2. **SSIM 本身是好指标**：在正确条件下，ctrl.ssim 稳定提升、ΔSSIM +0.30，能清晰反映骨架引导的质量。
3. **base 性能接近，但 eval 口径必须统一**：S21 base eval ssim 0.4977（旧集）vs S30 base 0.4946（新集），两者都是"重建自条件"的口径，需用**同一 eval 集 + 同一 skel latent** 重评才能公平对比 ctrl 增益。
4. **修复动作**：s31 config `gpu_eval_skel_latent_shards_dir` 应改为 `final_skel_latents_fame_1px`（全量 51822），并重启 s31 使 eval 恢复真实条件；已产生的 eval/early-stop 判断无效。

## 4b. 2026-09-01 晚：s31 正确 eval（修复后）

**根因确认**：s31 config `gpu_eval_skel_latent_shards_dir=final_skel_latents_eval_1px` 只有 501 个 latent，与 eval 集 500 个 img_id **覆盖 0/500**。`make_eval_cache` 在 latent 命中失败时**静默保留零张量**，导致训练中 eval 全程用"零骨架条件"。修复为 `final_skel_latents_fame_1px`（全量 51822，覆盖 100/100）。

**用独立脚本 `src/eval/eval_ctrl_ckpt.py` + 正确 latent 重评（GPU, n=100, cfg=0.7, steps=50, eval_fame_strict_clean+final_skel_latents_fame_1px）**：

| 项 | 修复前 eval（全零 skel） | 修复后 eval（真实 skel） |
|---|---|---|
| ctrl.ssim @ 42500 | 0.536 | **0.8081** |
| ctrl.mse @ 42500 | ~0.79 | **0.1442** |
| ΔSSIM vs base | +0.04 | **+0.3135** |
| ΔMSE vs base | -0.12 | **-0.7672** |
| skel_iou | 0.006 | **0.3357** |

→ s31（S30-base）在正确 eval 下与旧 S21-based ctrl（0.7974）水平相当甚至更好，**之前的"回撤"与"低分"全是 eval bug 假象**。

**全 ckpt 批量重评**：`_sync_work/eval_all_s31.sh` 对 2.5k~42.5k 全部 17 个 ckpt 逐个落盘 base/ctrl/gt/skel 图 + metrics.json，汇总 → `manual_eval_correct_skel/summary.json`，用以判断收敛点与是否需续训。

**s31 正确 eval 收敛曲线（dit_batch=100 大 batch, n=100, cfg=0.7, steps=50, eval_fame_strict_clean + final_skel_latents_fame_1px）**：

| step | ctrl.ssim | ctrl.mse | skel_iou | ΔSSIM |
|---|---|---|---|---|
| 2500 | 0.7355 | 0.2489 | 0.217 | +0.241 |
| 10000 | 0.7685 | 0.1836 | 0.269 | +0.274 |
| 20000 | 0.7917 | 0.1621 | 0.314 | +0.297 |
| 30000 | 0.8021 | 0.1536 | 0.330 | +0.308 |
| 42500 | **0.8081** | **0.1442** | **0.336** | **+0.314** |

- 全程单调上升、**无回撤**；但 25k 后进入收益递减（最近5段平均增量 +0.0011/2500，仍在递减）
- **决策：不再续训 s31，直接进入 REPA（s32）**，基于 s31@42500 最优 ckpt

## 4c. REPA 训练（s32）启动记录

- **启动**：`_sync_work/launch_repa.sh`（tmux `s32repa`），2026-09-01 18:29
- **基础**：S30 base `0132500.pt`（主模型）+ s31 ctrl `0042500.pt`
- **配置**：`s32_repa_finetune.json`，w_repa=0.1, repa_layer=8, batch=128, compile, max_steps=20000
- **修正**：s32 config `gpu_eval_skel_latent_shards_dir` 已从 `final_skel_latents_eval_1px` 改为 `final_skel_latents_fame_1px`（避免同样的 eval bug）
- **状态**：main 加载 missing=0, ctrl 加载 missing=0, REPALoss(本地 DINOv2) 就绪, trainable=68.8M, 15.5G 显存, ~4 steps/s
- **早期行为**：step 250 时 Diff=0.1585（去噪能力保持，无灾难遗忘）、REPA=0.26（从 0.89 快速收敛）→ 双损失健康

## 4d. s32（弱 REPA w=0.1）cfg=0.7 统一口径评测

**修正 eval cfg**：s32 训练时误用 `gpu_eval_cfg=1.7`（与 s31 的 0.7 不一致）。已停掉，用 **cfg=0.7** 重评 s32 已有 ckpt（`eval_s32_cfg07.sh`，含 main override：REPA 对主模型的修改也纳入评测）。

**s32 弱 REPA（w=0.1, L8, cfg=0.7, eval_fame_strict_clean+final_skel_latents_fame_1px）**：

| step | ctrl.ssim | ctrl.mse | skel_iou | lpips | ΔSSIM |
|---|---|---|---|---|---|
| 7500 | 0.8158 | 0.1225 | 0.3741 | 0.1455 | +0.3183 |
| 10000 | 0.8162 | 0.1219 | 0.3813 | 0.1450 | +0.3171 |
| 12500 | 0.8175 | 0.1214 | 0.3888 | 0.1443 | +0.3167 |
| 15000 | **0.8181** | **0.1210** | **0.3956** | **0.1439** | +0.3144 |

**对比 s31 best（@42500, cfg=0.7）**：ssim=0.8081, mse=0.1442, skel_iou=0.336。
→ 弱 REPA 已带来小幅但真实的提升：**ssim +0.010、mse -0.023、skel_iou +0.060**。REPA 确实在帮助，但力量偏弱（w=0.1 后期 REPA 占比 ~7%）。

## 4e. s32b 强 REPA（w=0.3, L2 8&11）启动

**代码升级（支持更强 REPA + 更全指标）**：
- `src/model/controlnet.py`：forward 支持 `return_intermediate_layers`（多层捕获，注入前）
- `src/loss/losses.py`：REPALoss 支持共享 teacher（多层只加载一次 DINOv2）+ 多层输入 tuple
- `src/train/train_repa.py`：`--repa-layers "8,11"` 多层 REPA；多层各配一个 proj
- `src/eval/eval_ctrl_ckpt.py`：支持从 REPA ckpt 的 `main.*` 覆盖主模型（评 REPA 对主模型改动）
- `src/eval/metrics_png.py`：新增 `bg_uniformity`(背景均一度)/`ink_purity`(墨色纯度)/`ringing`(振铃) 等视觉质量指标

**启动**：`_sync_work/launch_s32b.sh`（tmux `s32b`），2026-09-01 20:41
- 配置 `s32b_repa_strong.json`：w_repa=**0.3**, repa_layers=**8,11**, batch=128, compile, max_steps=20000
- **eval cfg 统一 0.7**，dit_batch=100（大 batch 快速 eval）
- 基于 s31@42500 最优 ckpt + S30 base 0132500
- 早期行为：step 250 REPA(L2)=0.41（比弱版 0.26 收敛更深）、Diff=0.163 保持健康

## 4f. s32b 强 REPA 中间评测 + 视觉质量指标对比

**s32b eval 曲线（cfg=0.7, eval_fame_strict_clean + final_skel_latents_fame_1px, daemon 计算）**：

| step | ctrl.ssim | ctrl.mse | ΔSSIM |
|---|---|---|---|
| 2500 | 0.8100 | 0.1379 | +0.358 |
| 5000 | 0.8111 | 0.1350 | +0.351 |
| 7500 | 0.8130 | 0.1312 | +0.343 |
| 10000 | 0.8158 | 0.1259 | +0.340 |

**扩展视觉质量指标（`metrics_png.py`，训练前 s31@42500 vs s32b@10000, cfg=0.7）**：

| 指标 | 训练前 s31 | s32b@10k | Δ | 方向 |
|---|---|---|---|---|
| mse | 0.1442 | **0.1259** | -0.0183 | ✓ |
| psnr | 16.40 | **16.62** | +0.22 | ✓ |
| ssim | 0.8054 | **0.8136** | +0.0082 | ✓ |
| tv (总变差/噪) | 0.01865 | **0.01852** | -0.0001 | ✓ |
| saltpepper | 0.00175 | **0.00168** | -0.0001 | ✓ |
| bg_uniformity | 0.0087 | **0.0085** | -0.0002 | ✓ |
| ink_purity (Otsu) | 0.9592 | **0.9611** | +0.0019 | ✓ |
| ringing (振铃) | 0.3770 | **0.3912** | +0.0142 | ✓ |
| lap_var (清晰度) | 0.0083 | 0.0086 | +0.0003 | ~(笔画更锐) |
| hf_energy | 0.0153 | 0.0157 | +0.0004 | ~(细节更丰富) |
| edge_clean | 0.5691 | 0.5691 | ~0 | ≈ |

**解读**：8/11 指标改善、**所有噪点类指标下降**（tv/saltpepper/bg_uniformity），墨色纯度与振铃改善显著 → REPA 让画面更干净、边缘更利落。lap_var/hf_energy 微升来自笔画细节变丰富（非噪点，saltpepper 在降佐证）。
注：`ink_purity` 于 21:40 改为标准 **Otsu 分离度**（η=σ_b²/σ_t², 有界 [0,1]），数值量纲从 0.0058 → 0.959 是口径变化，非突变。

## 5. 待办 / 下一步

- [x] 修复 s31 config 的 `gpu_eval_skel_latent_shards_dir` 并重启 s31
- [x] 批量重评 s31 全 ckpt → 已确认收敛（best 0.8081@42500）→ 决定进 REPA
- [x] s32 REPA 训练已启动（2026-09-01 18:29）；cfg=1.7 误用已修正为 0.7
- [x] s32 弱 REPA cfg=0.7 重评完成（best 0.8181@15000, 较 s31 +0.010）
- [x] s32b 强 REPA（w=0.3, L2 8&11, cfg=0.7）已启动（2026-09-01 20:41）
- [x] 扩展视觉质量指标 + 训练前后对比（8/11 改善, 噪点全降）
- [x] s32b 训练完成 → cfg=0.7 终评 + 全指标三线对比（s31/s32/s32b）

## 6. 2026-09-02：s32c 长收敛 + s32d 超强 REPA（24h 串行链）

**设计**：s32b 证明"batch 小需更多 steps"→ 改为 24h 串行链，两实验**独立对照**（均从 s31@42500 冷启）：
- **s32c-repa-longconv**：w_repa=0.3, L2(8,11), **max_steps=90000**（长收敛，LR 1e-4→1e-5 退火）
- **s32d-repa-super**：w_repa=**2.0**, L3(4,8,11), max_steps=45000, lr=5e-5（超强 REPA 压噪点）

**s32c 完整 eval 曲线（cfg=0.7, eval_fame_strict_clean + final_skel_latents_fame_1px, daemon n=100）**：

| step | ctrl.ssim | ctrl.mse | skel_iou | base.ssim |
|---|---|---|---|---|
| 5000 | 0.8112 | 0.1359 | 0.3690 | 0.4604 |
| 10000 | 0.8150 | 0.1252 | 0.3885 | 0.4776 |
| 15000 | 0.8158 | 0.1250 | 0.4029 | 0.4823 |
| 25000 | 0.8180 | 0.1235 | 0.4188 | 0.4283 |
| 30000 | 0.8191 | 0.1226 | 0.4236 | 0.3931 |
| 40000 | 0.8204 | 0.1211 | 0.4313 | 0.3318 |
| 50000 | 0.8198 | 0.1204 | 0.4361 | 0.2903 |
| 60000 | 0.8198 | 0.1202 | 0.4388 | 0.2621 |
| 70000 | 0.8195 | 0.1202 | 0.4409 | 0.2451 |
| 80000 | 0.8192 | 0.1204 | 0.4414 | 0.2337 |
| **90000** | 0.8190 | 0.1206 | 0.4407 | 0.2264 |

**s32c 结论**：
- ctrl.ssim 峰值 **0.8204 @ 40000**，随后轻微回落（40k 后过拟合, 回撤 -0.0014）。
- **base.ssim 从 0.46 一路降到 0.23**：REPA 在长收敛下持续把主模型表征"拉开"（高 REPA 压力），主模型重建能力退化（对齐 DINO 表征的代价）。但 ctrl 注入仍保住 0.82（CtrlNet 承担了重建）。
- 与 s32b（20k, w=0.3）：s32b 峰值 0.8177，s32c 长收敛提升有限（+0.0027）且伴随 base 崩溃 → **纯长收敛无增益，且损伤 base**。

**s32d（w=2.0 超强 REPA）早期**：step 5000 ctrl.ssim=0.7217, step 10000=0.7227（远低于 s32b/s32c 同期 0.81）→ w=2.0 的 REPA 压力**挤压重建通道**，CtrlNet 明显回撤。运行中（~15k/45k）。

## 7. 数据清洗 v7 质量评估（值不值得重新预训练）

**清洗范围（51,822 张 fame 图）**：
- **bars（伪影横/竖条）**：11,287 张（21.8%）bars>0；**bars>=3 有 2,238 张**（4.3%，重度污染）；单张最多 160 条
- **removed_frac（缺失笔画占比）**：18,506 张（35.7%）>0；>0.01 仅 711 张、>0.05 仅 17 张（重度缺失极少，多数是轻微残缺）
- **inverted（反相）**：0 张（v7 扫描无新增反相；早前 flip 39 张已处理）
- **手动翻转**：39 张（encode_changed_ids.json）
- **变更合计**：**21,050 张（40.6%）** 有某种缺陷 → 其中 **11,287 张 bars>0（21.8%）** 是主要问题

**结论：值得重新预训练（有条件）**：
1. **污染比例高**：21.8% 的图带 bars 伪影、40.6% 有缺陷。这些噪声直接灌进 latent 训练，是 s31/s32 中"噪点类指标"（saltpepper/tv/bg_uniformity）的根源之一。清洗后**消除了 21,050 张缺陷样本**，等于去掉约 4 成的"坏样本"。
2. **清洗是"修改而非删除"**：bars 修复（去条/补笔）、removed 修复、39 张翻转都**保留了样本数**（51,822→51,822），不缩小数据集，只提升单样本质量 → 重新预训练**不损失数据量**，只获益。
3. **与 REPA 协同**：REPA 能"压噪"（噪点类指标全降）但**不能修复结构性伪影**（bars/缺失笔画属于结构性错误）。清洗负责源头，REPA 负责表征对齐，两者互补。
4. **代价**：S30 预训练约 150k 步 / 单卡数天。若**清洗后的 S30' 只训到等量 ssim**，则证明清洗省步数；若同等步数更高 ssim 则证明质量提升。**建议**：把 S30' 作为 base，重跑 s31→s32c 全链（或直接 s32 强 REPA），用同一 eval 集对比。

**待办（清洗落地）**：
- [ ] encode 写回（cpu_encode_v9.py 分阶段多进程已重写，正在跑 img→skel3→skel1→writeback）
- [ ] 验证 fame.npz + 各 latent shards 补丁正确（skel 一致性抽查）
- [ ] 基于清洗后 latent 决定是否重训 S30'（先跑 ~1 epoch 冒烟对比 loss 曲线）

## 5b. 待办 / 下一步（追加）

- [ ] s32d 训练完成（~45k）→ cfg=0.7 终评 + 全指标四线对比（s31/s32b/s32c/s32d）
- [ ] encode v9 writeback 完成 + skel 一致性抽查通过
- [ ] 清洗后重预训练 S30' 决策（冒烟对比）
- [ ] 最终 git 提交（代码/配置/文档）
