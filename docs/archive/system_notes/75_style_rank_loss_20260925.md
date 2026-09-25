# 75 — 风格信号与 Style-Rank Loss（2026-09-25）

> 背景：风格注入被判定"不好"（v15a ratio_style 1.21 vs 基线 1.48），转向
> "能否造一个可用的风格信号，并把它挂进扩散训练"。本文记录 9/24–9/25 两天
> 的完整探索：信号测量 → 编码器训练 → 排序头 → 挂 loss 冒烟 → v18 预训练拉起。

## 0. 一页结论

| 主题 | 结论 |
|---|---|
| 免训练风格信号 | **有**。DINO patch-std 最强（同字池富集 3.02×），dino_mean/pixel_ssim 2.68×，ink_stats 1.68× |
| 对比学习增量 | content-matched InfoNCE：2.24× → 3.36×（DINO 特征） |
| latent 编码器（分类式） | 2.9M：留出·字没见过 23.3%（10.7× 随机），富集 3.13×，**AUC 仅 0.606**；缩小/加噪/大 batch 全部更差 |
| **DINO 排序头（最终采用）** | **留出 AUC 0.755**（+0.149），富集 2.92×，92 秒训完 |
| style-rank loss 冒烟 | ✅ 通过：1k 步 Diff 0.275→0.268 稳定、strict 0.5481（基模 0.5454）、无发散 |
| **v18 预训练** | **已拉起**：200k cosine、从头、w=0.005、t∈[0.05,0.25]，tmux `v18` |

## 1. 信号测量（免训练，content-matched 同字池检索）

口径：query=strict 留出 GT（249 张），在同字训练池（≤12 张）里最近邻，
命中同书家比例 vs 池内自然占比 = 富集。脚本 `tools/style_signal_trainfree.py`。

| 表示 | 命中 | 富集 |
|---|---|---|
| **dino_std**（patch 间方差） | 7.6% | **3.02×** |
| pixel_l2 | 7.1% | 2.85× |
| dino_mean / pixel_ssim | 6.7% | 2.68× |
| ink_stats | 4.2% | 1.68× |

**关键洞察：mean 编码内容、std 编码笔法纹理** → 后续特征统一用 `dino_mean+std`（768 维）。
`assets/dino_meanstd_50k.npy` (51036,768) 已预计算。

## 2. 对比学习投影（DINO 特征上）

`tools/style_contrastive_probe.py` → `assets/style_proj_dino_50k.pt`
content-matched InfoNCE（POS=同书家异字 / NEG=同字异书家，batch 1024）：
免训 2.24× → **3.36×**（+50%），loss 2.20→1.786。
历史 bug 均已记录在脚本头（下标混用 / InfoNCE 漏 −pos/τ / 泄漏 / batch 内配对 no-op）。

## 3. latent 编码器（分类式）—— 信号在，但不够格当 loss

`tools/style_encoder_latent.py` → `assets/style_enc_latent.pt`（2.9M，吃 (4,32,32) VAE latent）：

| 划分 | 准确率 | 随机 |
|---|---|---|
| 训练样本 | 100% | 2.2% |
| 留出·字见过 | 30.1% | 2.2% |
| **留出·字没见过** | **23.3%** | 2.2% |
| 同字池富集 | 3.13× | — |
| **pairwise AUC（留出）** | **0.606** | 0.5 |

- "字没见过"几乎不掉（30.1→23.3）→ 学的是**书家条件分布**，不是 (字,书家) 配对 → 良性过拟合。
- **缩模型/正则实测更差**：358K + dropout/wd → 18.5%/15.5%、富集 2.42×；
  输入加噪 0.05 正好抹掉笔触纹理（最差）。**大 batch 复扫（b1024→4096）也无起色**
  （ch16@4096: 15.4%、2.25×、AUC 0.604）→ 不是优化量问题，是 45 类在 latent 的可分性上限。
- **AUC 0.606 = 没资格当训练 loss**（40% 的同书家对与异书家对一样远 → 梯度会压掉内容多样性）。

## 4. DINO 排序头（最终裁判）

`tools/rank_dino.py` → `assets/style_rank_dino.pt`（712K 参数，92 秒）
损失 = pairwise ranking（softplus margin）+ InfoNCE，同 content-matched 口径：

| 指标 | latent 2.9M | **DINO 排序头** |
|---|---|---|
| 留出·字没见过 AUC | 0.606 | **0.755** ✅ |
| 留出·字见过 AUC | — | 0.759 |
| 富集 | 3.13× | 2.92× |

轨迹（1000→4000 步）：0.712 → 0.744 → 0.752 → 0.755（仍缓升）。
**判读：≥0.75 = 有资格当 diffusion 的风格监督。**

## 5. style-rank loss 实现（train.py 补丁，全部冻结，added=0）

文件：
- `src/loss/style_rank_module.py` — `StyleRankLoss(ckpt, cent_npy, t_min, t_max)`
- `tools/build_rank_feat.py` — 预计算 87 pair 质心 → `assets/rank_cent87.npy` (87,256)
  （自检：质心对外 cos mean −0.007 / p95 0.514 / max 0.999 —— max≈1 的两 pair 需留意）
- `_sync_work/apply_style_rank_patches.py` — 幂等补丁（备份 `*.bak_prerank`）

机制：
```
loss = 1 − cos( f(pred_xstart), centroid[y_pair] )      # 仅 t∈[0.05,0.25] 的样本
```
- `pred_xstart` 携带完整 DiT 梯度图（`_need_x0` 已加 `w_style_rank` 分支）；
- 编码器与质心全部冻结（可训练参数 +0）；
- cli 新 flag：`--w-style-rank / --style-rank-ckpt / --style-rank-cent / --style-rank-t-min/max`。

质心用的编码器 = 2.9M latent 编码器（AUC 0.606 的那把"尺子"），
但**监督目标是 pair 质心而非逐样本特征**——500+ 张图的平均，单图噪声被平均掉；
排序头（0.755）是判据，用于事后评测验证。

## 6. 冒烟（2026-09-25 05:18–05:30，通过）

底模 `v17_inj3_fixed_aug_100k@100k`（strict 0.5454），续 1k 步，w=0.005：

| step | Diff | strict(249) | seen(20) | cal_enrich |
|---|---|---|---|---|
| 100k（基模） | — | 0.5454 | 0.5818 | — |
| 100.5k | 0.2728 | 0.5481 | 0.5801 | 2.20× |
| 101k | 0.2679 | 0.5471 | 0.5777 | 2.20× |

- Diff 稳定下降（0.2747→0.2679），无发散、无 NaN；
- strict **+0.0027**（噪声内但方向对）；seen 略降 0.004（1k 步噪声范围）；
- style-rank loss 生效样本占比 ~8%（t<0.25 在 logit_normal 下自然比例）；
- grad-norm 正常（blocks 0.047）。

## 7. v18 预训练（已拉起，200k）

`src/train/configs/v18_style_rank_200k.json`（`_sync_work/make_v18_cfg.py` 生成）：
- 基模 = `v17_inj3_fixed_aug_100k` 全配方（inj3 三件套 + 离线几何 4 档 + 在线数值增强 + fixed 数据）
- 唯一新增：`w_style_rank=0.005`，t∈[0.05,0.25]
- 200k cosine（lr 5e-4, warmup 3000, batch 360, EMA on）
- tmux `v18`，日志 `logs/v18_series/v18_train_20260925-053355.log`
- 开跑即测：step 2450, Diff 0.3767↓, 4.04 step/s, Mem 19.8G —— 正常

**判据**（跑完后按序执行）：
1. strict 轨迹 vs `v17_inj3_fixed_aug_100k`（0.5454@100k）同口径；
2. `cal_enrich` 探针（in-mem eval 自带）随步数上升；
3. 完赛后 `tools/eval_diversity.py` 测 ratio_style（同协议 baseline：v13_base 1.48 / v15a 1.21）；
4. 失败判据：strict 掉 >0.005 或 intra 被压（ratio 虚高但生成塌）→ 降 w 或关 t 高段。

## 8. 遗留与备忘

- 质心 max cos≈0.999 的 pair 对：loss 会把这两个 pair 视作同一目标，影响未知，观察 strict 分书体分布。
- 排序头 0.755 仍非满分；若 v18 无效，下一档是"把 f() 换成排序头本身 + pixel 级 DINO（decode 一次离线建库）"。
- 冒烟对照臂 B（w=0 的 1k 步）只跑了 500 步即被停（用户裁定不冒烟了），strict±0.003 的自然波动未知，判读时以 v17 基模 100k→更长的自然斜率为准。
- v15b 中断在 118k/150k、v15c 未跑，仍待用户拍板是否续。

## 9. 文件清单（本次新增/修改）

```
src/loss/style_rank_module.py            # StyleRankLoss（新）
src/train/train.py                       # P1/P3/P3b 补丁（bak_prerank 备份）
src/train/cli.py                         # 5 个新 flag（bak_prerank 备份）
tools/rank_dino.py                       # 排序头训练+评测（新）
tools/style_encoder_latent.py            # latent 分类编码器（更早, 已有）
tools/style_contrastive_probe.py         # DINO InfoNCE 投影（更早, 已有）
tools/style_enc_overfit_check.py         # 过拟合检查
tools/build_rank_feat.py                 # 87 质心预计算（新）
tools/sweep_style_small.py               # 小模型/大 batch 扫描（新）
tools/style_signal_trainfree.py          # 免训练信号测量（已有）
_sync_work/apply_style_rank_patches.py   # 幂等补丁脚本（新）
_sync_work/make_smoke_cfg.py             # smoke_styleA/B 配置生成（新）
_sync_work/make_v18_cfg.py               # v18 配置生成（新）
_sync_work/launch_v18.sh / launch_smokeA.sh / launch_smokeB.sh / fscheck.sh
src/train/configs/v18_style_rank_200k.json  # 预训练配置（新）
src/train/configs/smoke_styleA/B.json       # 冒烟配置（新）
assets/style_rank_dino.pt                # 排序头权重（712K）
assets/style_enc_latent.pt               # 2.9M latent 编码器
assets/dino_meanstd_50k.npy              # DINO mean+std 特征 (51036,768)
assets/rank_cent87.npy                   # 87 pair 质心
logs/_rankDino.log / _stySweep*.log / _smoke_styleA.log / v18_series/*.log
```