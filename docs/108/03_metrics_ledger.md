# 108-03 · 指标台账（2026-10-09 重抽，含在跑的 v72）

> 与既有 [`docs/experiments/mainline_20261009.csv`](../experiments/mainline_20261009.csv)（17 行 187 协议主表）**互为校验**：v68 旗舰 **0.6149@200k** 两侧一致 ✅。
> 本台账的不同之处：**逐评测点**（616 行）而非每 run 一行，且**纳入了在跑的 v72 与 5 个现代组件消融臂**。

## 1. 产物

| 文件 | 行数 | 内容 |
|---|---|---|
| [`assets/ledger/points_all.csv`](../../assets/ledger/points_all.csv) | 617（含表头） | **逐评测点 × 33 列** |
| [`assets/ledger/runs_all.csv`](../../assets/ledger/runs_all.csv) | 37 | 每 run 一行：摘要 + 配置快照（40 个 `cfg_*` 列） |
| [`assets/ledger/aligned_curve.csv`](../../assets/ledger/aligned_curve.csv) | 41 | step 对齐矩阵（12 条主线） |
| `assets/ledger/quarantine_manifest_20261009.csv` | 31 | 隔离清单（回滚用） |

**抽取范围**：`assets/runs`、`exp-std`、`experiments`、`baseline` 下全部 `eval_auto*.json` = **616 个**（解析失败 0），覆盖 **36 个 run**。
**生成脚本**：`scratch/build_ledger.sh`（体积/ckpt/末行索引）+ `scratch/build_ledger3.sh`（Python 抽取与落盘，纯 `json/csv`，**不 import torch、不碰 GPU**）。

## 2. `points_all.csv` 字段

| 组 | 列 |
|---|---|
| 定位 | `family, run, run_key, instance, json` |
| 评测元信息 | `set`（`eval200fix` / `seen` / `(no-set)` 老 run）、`step`、`n`、`cfg`、`ddim_steps`、`engine`、`elapsed_s` |
| 主指标 | `ssim`、`ssim_med`、`ssim_q25`、`ssim_q75`、`ssim_p10`、`ssim_p90`、`ssim_std` |
| 像素指标 | `mse`、`mse_std`、`lpips` |
| 结构/墨迹 | `ink_ssim`、`ink_iou`、`skel_iou`、`frag`（破碎度）、`hole`、`hole_gt` |
| 参照/风格 | `nn_ssim`、`nn_mean`（同字最近邻）、`tgt_spec`（风格专属性）、`cal_enrich`（书家富集度） |
| strict | `strict_ssim`、`strict_n` —— **本批数据全为 `None`**（见 §5 口径提醒） |

## 3. 主线 step 对齐（`eval200fix`, N=187）

| step | v68 (aug+SD-VAE) | **v72 (noaug+Calli-VAE)** | Δ | v66 (150k 线) | v54 (字表基线) | v70 (stdskel) | v68_pix |
|---|---|---|---|---|---|---|---|
| 5k | 0.5225 | 0.5247 | +0.0022 | 0.5140 | — | 0.5341 | 0.5306 |
| 10k | 0.5300 | **0.5414** | +0.0114 | 0.5341 | 0.5204 | 0.5349 | 0.5298 |
| 15k | 0.5355 | **0.5543** | +0.0188 | 0.5407 | 0.5337 | 0.5354 | 0.5290 |
| 20k | 0.5429 | **0.5650** | +0.0221 | 0.5397 | 0.5365 | 0.5429 | 0.5365 |
| 25k | 0.5456 | **0.5746** | +0.0290 | 0.5456 | 0.5419 | 0.5408 | 0.5387 |
| 30k | 0.5495 | **0.5844** | +0.0349 | 0.5504 | 0.5459 | 0.5457 | 0.5414 |
| 40k | 0.5621 | **0.5945** | +0.0324 | 0.5602 | 0.5521 | 0.5466 | 0.5511 |
| 50k | 0.5701 | **0.5999** | +0.0298 | 0.5661 | 0.5613 | 0.5496 | 0.5588 |
| 60k | 0.5791 | **0.6032** | +0.0241 | 0.5729 | 0.5674 | 0.5535 | 0.5664 |
| 65k | 0.5833 | **0.6034** | +0.0201 | 0.5767 | 0.5689 | 0.5535 | 0.5706 |
| … | | | | … | | | |
| 150k | 0.6118 | — | | **0.5955** | **0.5960** | — | — |
| 200k | **0.6149** | — | | | | | — |

### 关键读数

1. **v72 在 50k（0.5999）就超过了 v54/v66 跑满 150k 的终点（0.5960/0.5955）** —— 在"字表 S/2 + 黄金路由"这条线上，换用新 Calli-VAE 潜空间 + 无增广语料后，**用约 1/3 的步数达到旧线终点水平**。
2. **v68 的曲线形态完全不同**：v68 是"慢热长跑"（到 200k 才 0.6149），v72 是"早熟"（65k 已 0.6034）。两者 65k 的差距 +0.0201 在统计上可观。
3. **v70（纯 std-skel 开集路线）明显最弱**：65k 仅 0.5535，且在 60k 后停滞（0.5535→0.5535），并在 **78,750 步被停机**（未跑到 200k）——与 `docs/experiments/mainline_20261009.csv` 记的 0.5535@65k 一致。
4. **v68_pix（像素域变体）反超预期**：110k 达 0.6002，高于 v68 同步数。

### ⚠ 三个必须标注的限定

1. **不是干净单变量**：v72 = noaug 语料 + 新 Calli-VAE；v68 = 3x 增强 + SD-VAE + C2OT。v72 的**正确对照臂**是 48 上的 `noaug_c2ot`（SD latent + noaug，见 [决策文档 §5](../experiments/2026-10-09-frozen-vae-vs-repa-e-decision.md)），**该臂数据不在 4090**，因此**不能把全部增益归因给 VAE**。
2. **口径**：本批为**远程 in-mem eval，`cfg=1.0`、无 SDE**。仓库默认口径已是 `cfg 0.8 + γ 0.5`（[02](02_sync_reconciliation.md) §5 指出训练机没有 SDE 代码）→ **跨口径不可直接比**。
3. **单 run、20k~65k 趋势读数**：中低步数区间绝对值偏低（LR 未退火），适合读趋势与相对差，不适合当作终点结论。

## 4. 消融台账（20k 同频，README §5 的机器可读版）

| 臂 | 20k `eval200fix` SSIM | Δ vs 基准 | 结论 |
|---|---|---|---|
| 基准（v68 配置 @20k） | **0.5429** | — | — |
| 去 RoPE | 0.5408 | −0.0021 | 正贡献（噪声边缘） |
| 去 RMSNorm（→layer） | 0.5405 | −0.0023 | 正贡献（噪声边缘） |
| 去 QK-Norm | 0.5375 | −0.0054 | **正贡献（扎实）** |
| 去 SwiGLU（→gelu） | 0.5366 | −0.0063 | **正贡献（最扎实）** |
| 表内容：40 万切片 | 0.5453 | **+0.0024** | 建表语料边际 ≈ 0 |
| 表内容：像素表 | 0.5365 | −0.0064 | 负 |
| 表内容：完全随机 | 0.5323 | **−0.0106** | **最大杠杆：学过 vs 随机** |

→ 与 README §5 完全一致：**四个现代组件全部正贡献（且都主要买感知质量，LPIPS 一致恶化 +0.004）；"表内容学过 vs 随机"是全场最大杠杆（~0.011）**。

## 5. 数据坑与复现

1. **`strict_ssim` 全为 `None`**：`eval_auto_*.json` 里只有 `origin.strict` / `strict` 嵌套字段，本批 616 个文件中**没有一个**带 strict 数值 → strict 口径的分数在**离线 eval 产物**里（如 `assets/eval200_*_exact_metrics.csv`、48 侧 `eval_strict/*.json`），不在本台账范围内。
2. **老 run 无 `set` 字段**：`v10b`/`v11_pretrain` 系列的 JSON 没有 `set` 键（字段为 `step/elapsed_s/engine/origin/self_cond/ssim/mse/delta_seen_ssim`）。抽取器已做回退：无 `eval200fix` 标记时按全部点取 best/last，并在 `sets_present` 列标注（`v12`~`v18`、`moyi` 等 36 个 run 中多数属此类）。
3. **run_key 层级**：以路径中的 `checkpoints` 段为锚回溯两级得到 `assets/runs/<family>/<run>`；`exp-std/*` 下为 `exp-std/<group>/<run>`。
4. **重建命令**（远程，CPU-only）：
   ```bash
   bash scratch/build_ledger.sh      # 索引体积/ckpt/末行（写 /tmp 缓存）
   bash scratch/build_ledger3.sh     # 抽台账（读缓存 → _sync_work/ledger/*.csv）
   ```
   随后 `scp 4090:.../_sync_work/ledger/*.csv assets/ledger/`。
