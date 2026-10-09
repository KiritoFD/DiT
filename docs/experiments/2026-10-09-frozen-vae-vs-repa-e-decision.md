# 决策：冻结 Calli-VAE 训练 vs REPA-E 端到端（2026-10-09）

> 判据不是 PSNR，而是**"冻结的潜空间能否被扩散模型直接当目标学"**。本文件给出三个轴的实测，并给出决策与升级触发条件。
> 工具：`tools/vae/probe_latent_prior.py`（先验对齐 / 先验采样可解码性 / 潜空间敏感度）、
> `tools/vae/eval_calli_vae.py`（重建 + latent 诊断）、`eval_calli_vae_trend.py`（趋势）。
> 测量对象：Calli-VAE `step_12500`（kl1e6 版）vs `sd-vae-ft-ema`，32 张 `train_clean` 抽样，fp32，4090。

---

## 1. 三个轴的实测

### A. 先验对齐（潜变量逐维统计 vs N(0,1)）

| 指标 | **Calli-VAE** | sd-vae-ft-ema |
|---|---|---|
| 逐维 std 均值 / 中位 | **1.005 / 0.965** | 4.098 / 3.611 |
| 逐维 std p5 / p95 | 0.777 / 1.409 | 1.338 / 7.909 |
| 逐维 \|均值\| 中位 | **0.124** | 3.220 |
| 整体 z.std / 维间相关 | 1.005 / — | 6.256 / 0.178 |
| **DiT 目标 (z×sf) 逐维 std 中位** | **0.176** | 0.658 |

→ **Calli 潜空间几乎就是标准正态**（std≈1、均值≈0.12）；SD-VAE 则远离高斯（std≈4.1、均值≈3.2）。
从"扩散先验假设"角度，**Calli 比 SD-VAE 更规范** —— 这正是它能被标准 `decode(sample)` 直接解码、而 SD 需要 0.18215 缩放因子的原因。
⚠ 但两者**目标尺度差 3.7 倍**（0.176 vs 0.658）：现有 DiT 配置的噪声日程/初始化是按 SD 目标标定的，换 Calli 后应做一次全局重标定（见 §3 行动项 1）。

### B. 先验采样可解码性（z~先验 → decode）

| 采样 | Calli 墨迹覆盖 | Calli→真实图 DINO cos | SD 墨迹覆盖 | SD→真实图 DINO cos |
|---|---|---|---|---|
| z~N(0,1) | 0.036 | 0.316 | 0.982 | 0.291 |
| z~N(μ,σ²)（后验） | **0.224** | **0.953** | **0.223** | 0.952 |
| 真实图参考 | 0.223 | — | 0.223 | — |

→ **两者行为一致**：随机高斯解码都不在真迹流形上（Calli 偏空、SD 偏黑），但**后验采样解码完美**（墨迹覆盖与真实图几乎相同）。
这是所有 VAE 的常态，**不构成缺陷**：扩散模型学的是潜空间的条件流形，不是边缘高斯。

### C. 潜空间敏感度（decode(z + ε·noise)，noise 按各 VAE 逐维 std 缩放）

| ε | Calli L1 | Calli SSIM | Calli ink_IoU | SD L1 | SD SSIM | SD ink_IoU |
|---|---|---|---|---|---|---|
| 0 | 0.0142 | 0.9832 | 0.9872 | 0.0180 | 0.9801 | 0.9793 |
| 0.25 | **0.0275** | **0.9464** | 0.9553 | 0.0418 | 0.9309 | 0.9486 |
| 0.5 | **0.0481** | **0.8890** | 0.9108 | 0.0802 | 0.8233 | 0.9051 |
| 1.0 | **0.1119** | **0.7752** | 0.8041 | 0.1794 | 0.5232 | 0.8022 |

→ **Calli 潜空间对扰动更鲁棒**（同 ε 下损伤明显更小，ε=1.0 时 SSIM 0.775 vs 0.523）。
扩散预测永远不完美，潜空间越鲁棒 → 对 DiT 精度要求越低 → **冻结合成训练更可行**。

### D. 参考：已有下游证据（冻结路线）

v71（冻结 **Calli** latent + DiT）vs v68（冻结 **SD** latent + DiT），同步数 30k、同评测集 eval200fix(187)：
SSIM 0.5731/0.5495、MSE 0.8739/0.9635、ink_SSIM 0.4112/0.3898、ink_IoU 0.2826/0.2359、
skel_IoU 0.0169/0.0149、frag_ratio 1.488/2.036、nn_SSIM 0.6297/0.6120 **全面领先**，仅 LPIPS 落后（0.424/0.384）。
→ 冻结 Calli 潜空间**已经能训、且优于冻结 SD**。

---

## 2. 决策

**走冻结 VAE 路线（Route B）**，REPA-E 降级为**备选**（不是默认）。

理由：
1. 潜空间与高斯先验的对齐度优于 SD-VAE（std 1.005 / mean 0.124）；
2. 潜空间鲁棒性更好（扰动损伤曲线全面优于 SD）；
3. 冻结路线已有正面下游证据（v71 > v68，结构类指标全胜）；
4. REPA-E 的代价是现实的：encoder 解冻 + rsample + 两速优化器（VAE lr 2e-6）+ 在线 DINO 正则，
   显存与步速成本显著，且**有把 encoder 的重建能力带坏的风险**（高频退化）；
5. 冻结路线的工程复杂度、可复现性、对照性都更好（v68/v71 都是冻结，天然可做单变量对照）。

## 3. 行动项（走冻结路线必须做的三件事）

1. **潜空间重标定**：现有 DiT 配置的噪声日程按 SD 目标 std≈0.658 标定，Calli 为 0.176。
   → 建议把离线编码的 latent 乘一个常数 `k = 0.658/0.176 ≈ 3.7`（等价于用 `sf_calli = sf/3.7 ≈ 0.049`），
   使 DiT 目标尺度与既有配置一致；或在 config 里显式调 noise schedule。**二选一，但必须显式处理**。
   （v71 未做此标定也能训起来 → 说明不致命，但属"免费收益"）
2. **离线编码用一次性 sample 并缓存**（标准做法）：`z = posterior.sample()` 落盘，不要每个 epoch 重采样
   —— 保留细节随机性，同时避免目标抖动；注意不要退回 `mode`（会丢 8% 的重建细节）。
3. **保留 mode 解码的兼容**：探针显示 `mode` 与 `sample` 的解码 L1 已只差 8%，推理端可直接用 `decode(pred/sf)`
   （不再需要 `calli_decode_noise` 补丁）。

## 4. 升级到 REPA-E 的触发条件（预先定义，避免主观）

满足**任一**即启动 REPA-E 对照臂：
- **T1**：冻结 Calli 的 flow loss 在前 20k 步的平台显著高于冻结 SD 基线（同预算、同配置）；
- **T2**：生成结果出现与 VAE 重建失败模式一致的系统性伪影（如固定位置的笔画断裂/糊化）；
- **T3**：在同等步数下，冻结 Calli 的 eval200fix 指标**低于**冻结 SD（当前 v71 已超过 → 该条不会触发）。

若三条都不触发 → **不做 REPA-E**（省下大量工程与算力）。

## 5. 便宜的判定实验（建议下一步）

同一 DiT 配置、同等预算（20k 步），三臂对照：

| 臂 | 内容 | 状态 |
|---|---|---|
| A | 冻结 Calli latent（按 §3.1 重标定） | **待跑**（必跑） |
| B | 冻结 SD latent | 已有（v68 曲线 / README §2.5） |
| C | REPA-E（encoder 解冻 lr 2e-6） | 仅当 T1/T2/T3 触发才跑 |

看 flow loss 曲线 + 20k 处 eval200fix。A 是决策的关键路径，B 免费复用，C 按需。

---

## 6. 执行记录：v72 已启动（2026-10-09 14:22）

**决策落地**：VAE 训练于 step 12,800 主动停机（GPU 让给 v72）；用 step 12,500 权重冻结、预编码、起 v72。

### 三套 latent 的区分（★ 命名防混淆）

| shards | VAE | 存盘约定 | 解码约定 | 用于 |
|---|---|---|---|---|
| `exp-std/data/shards_img_aug` | sd-vae-ft-ema | `mode()*sf` | `decode(lat/sf)` = `decode(mode)` | v68、消融 4 臂 |
| `exp-std/data/shards_img_aug_calli`（48 上） | 旧 calli (step 25000, 非标准) | 裸 `sample()` | `decode(sample/sf)`（须注噪补丁） | v71 |
| **`exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample`** | **新 calli (kl1e6, step 12500, 标准)** | **`sample()*sf`** | **`decode(lat/sf)` = `decode(sample)`** | **v72** |

- 冻结制品：`experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv/`（含 `MD5SUMS.txt`、`PROVENANCE.md`）
- 编码自检（`tools/vae/verify_calli_latent_shards.py`）：16 shards / 77,823 行 / names 齐全 / img_id 唯一；
  存储 latent std **0.1955 ≈ sf 0.18215**（证明存的是 `sample*sf`）；
  `decode(lat/sf)` **L1 = 0.0072**，对照 `decode(lat)` = 0.6646 → 约定唯一正确 ✓

### v72 运行配置（要点）

- 语料：**无增广原版 26,002 行**（`exp-std/csv/train_top10_noaug.csv`，由 `tools/derive_noaug_csv.py` 从 3x 增强 CSV 派生，覆盖率 100%）
  → 与消融 `noaug_c2ot`（SD latent + 无增广）构成**单变量对照**（只换 VAE）；shards 是 77,823 超集，改一行 `data_csv` 即可切增强版
- **评测 VAE 已同步切换**：`vae_path` = `eval_vae_path` = 冻结新 VAE；
  `calli_decode_noise: false` → 评测走标准 `decode(lat/sf)`（`src/eval/in_mem_eval.py:984/1021`）
- 启动器：`tools/launch_v72.sh`（预检 + 清单 `logs/v72_launch_manifest.json`），
  内置「**语料一致性断言**」（config 的 `data_csv` 必须与 `CORPUS` 选择一致——曾因两者不一致误跑了增强语料，已修）

### 实测（14:25）

step 400 时 Diff 1.247→0.366、REPA 0.030→0.020、**3.68 steps/s**、显存 21.4G；
预估 200k 步 ≈ 15h（明早 ~05:30 完训），首个评测点 5k ≈ 15:40（eval200fix 187）。

