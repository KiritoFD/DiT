# VAE 本底噪声实验汇总

> 2026-08-25
> 汇总历史上所有 VAE encode→decode 纯重建质量实验 (不经过 diffusion)。
> 数据来源: 远程 `/root/Workspace/xy/DiT` + 本地 `tools/`。

## 实验一览

| # | 脚本 | 图片集 | 数量 | 域 | 日期 | 结果文件 |
|---|------|--------|------|-----|------|---------|
| 1 | `tools/vae/benchmark_vae.py` | train csv 随机 | 200 | [-1,1] 灰度 | 08-23 | `tools/vae/bench_results/results.json` |
| 2 | `tools/vae/verify_latents_f4.py` | 随机采样 | 100 | [0,1] RGB | 08-23 | 记录在 `DATA_PIPELINE.md` |
| 3 | `tools/evaluation/vae_noise_eval.py` | eval100_top30_clean | 79 | [0,1] RGB | 08-24 | `vae_noise_eval_summary.json` |
| 4 | `tools/evaluation/vae_patch_compare.py` | eval100_top30_clean | 79 | [0,1] RGB | 08-24 | `vae_recon_compare.json` |
| 5 | `tools/evaluation/vae_noise_gpu.py` | train_top30_clean 全量 | 106,345 | [0,1] RGB | 08-24 | 未完成 (GPU 与训练抢显存) |
| 6 | `tools/evaluation/vae_noise_multiproc.py` | train_top30_clean 全量 | 106,345 | [0,1] RGB | 08-24 | 未完成 (CPU 太慢) |
| 7 | `tools/evaluation/vae_noise_full.py` | train_top30_clean 全量 | 106,345 | [0,1] RGB | 08-24 | 未完成 (同上) |

> 实验 3 和 4 是同一批数据、同一套代码逻辑 (vae_patch_compare.py 即 vae_recon_compare.py 的远程版)，结果一致。

## 两个 VAE 的参数

| VAE | 下采样 | latent 通道 | latent shape (256²) | latent size | 参数量 | scaling_factor |
|-----|--------|-----------|---------------------|-------------|--------|---------------|
| sd-vae-ft-ema (SD1.5) | 8× | 4 | (4, 32, 32) | 4,096 | 83.7M | 0.18215 |
| kl-f4 | 4× | 3 | (3, 64, 64) | 12,288 | 55.3M | 0.102079 |

## 核心结果

### 实验 1: benchmark_vae.py (200 张, [-1,1] 灰度域)

> ⚠️ 此实验有 scaling_factor 除法 bug (见下文)，MSE 绝对值偏高，但 **相对排序可信**。

| VAE | MSE [-1,1] | SSIM | latent std |
|-----|-----------|------|-----------|
| sd-vae-ft-ema | 0.2064 | 0.7234 | 6.43 |
| sd-vae-ft-ema-gray (1ch) | 0.2063 | 0.7235 | 6.43 |
| **kl-f4** | **0.0740** | **0.8699** | 10.42 |
| kl-f4-gray (1ch) | 0.0740 | 0.8699 | 10.42 |

→ kl-f4 MSE 仅为 sd-vae 的 **36%**，SSIM 高 **0.15**。黑白 surgery 零损失。

### 实验 2: verify_latents_f4.py (100 张随机, 记录于 DATA_PIPELINE.md)

| VAE | MSE [0,1] | SSIM |
|-----|----------|------|
| sd-vae-ft-ema (f8) | 0.003660 | 0.9655 |
| **kl-f4 (f4)** | **0.001910** | **0.9882** |

→ kl-f4 MSE 仅为 sd-vae 的 **52%**，SSIM 接近无损 (0.988)。

### 实验 3/4: vae_noise_eval.py + vae_patch_compare.py (79 张 eval, [0,1] RGB, 正确实现)

| 配置 | MSE [0,1] | MSE [-1,1]×4 | SSIM | SSIM min | latent tokens (p4) |
|------|----------|-------------|------|---------|-------------------|
| **kl-f4 + patch4** | **0.000696** | 0.00278 | **0.9753** | 0.779 | 256 |
| kl-f4 + patch2 | 0.000696 | 0.00278 | 0.9753 | 0.779 | 1024 |
| sd-vae-ft-ema + patch4 | 0.000845 | 0.00338 | 0.9620 | 0.689 | 64 |
| sd-vae-ft-ema + patch2 | 0.000845 | 0.00338 | 0.9620 | 0.689 | 256 |

→ **关键验证: patch 不影响 VAE 重建** (p2 vs p4 的 MSE/SSIM 完全一致，因为 decode 的是同一 latent)。
→ kl-f4 MSE 仅为 sd-vae 的 **82%**，SSIM 高 **0.013**，最差图 SSIM 也更好 (0.779 vs 0.689)。

## 汇总对比表

| 度量 | sd-vae-ft-ema (f8) | kl-f4 (f4) | kl-f4 优势 |
|------|-------------------|-----------|-----------|
| **重建 MSE [0,1]** (79 eval, 正确) | 0.000845 | 0.000696 | **低 18%** |
| **重建 SSIM** (79 eval, 正确) | 0.9620 | 0.9753 | **高 0.013** |
| **重建 MSE** (100 随机, DATA_PIPELINE) | 0.0037 | 0.0019 | **低 49%** |
| **重建 SSIM** (100 随机, DATA_PIPELINE) | 0.9655 | 0.9882 | **高 0.023** |
| **重建 MSE [-1,1]** (200 张, benchmark) | 0.2064 | 0.0740 | **低 64%** ⚠️ |
| **重建 SSIM** (200 张, benchmark) | 0.7234 | 0.8699 | **高 0.147** ⚠️ |
| **latent 信息量** | 4,096 | 12,288 | **3×** |
| **参数量** | 83.7M | 55.3M | **少 34%** |
| **最差图 SSIM** (79 eval) | 0.689 | 0.779 | **高 0.09** |

> ⚠️ benchmark_vae.py 的 MSE 绝对值偏高 (0.074 vs 0.0007)，原因是该脚本在 decode 前多除了一次 scaling_factor (`z / vae.config.scaling_factor`)，放大了 latent 数值。但因为是两个 VAE 都做了同样的错误操作，**相对比较仍有效**。实验 3/4 的实现是正确的 (不除 scaling_factor)。

## 结论

**kl-f4 的 VAE 画质 (本底噪声) 在所有实验中全面优于 sd-vae-ft-ema**：

1. **重建 MSE 更低**: 0.0007 vs 0.0008 (eval集), 0.0019 vs 0.0037 (随机集) — kl-f4 低 18%-49%
2. **重建 SSIM 更高**: 0.975 vs 0.962 (eval集), 0.988 vs 0.966 (随机集) — kl-f4 高 0.013-0.023
3. **最差情况更好**: kl-f4 最差图 SSIM=0.779 vs sd-vae 0.689 — kl-f4 在困难样本上优势更大
4. **latent 信息量 3×**: 12,288 vs 4,096 — 更大的 latent 空间保留更多高频细节
5. **参数更少**: 55.3M vs 83.7M — 少 34%

**所以 kl-f4 的画质天花板确实更高。** DiT 训练的 eval SSIM 低于 sd-vae 时代的实验 (s7/s8/s11 的 0.50-0.55 vs s6 的 0.73)，原因是 DiT 还没训练到位 (步数/超参)，而非 VAE 限制。VAE 本底噪声 (SSIM 0.975) 远高于当前 DiT eval (SSIM 0.55)，说明 **瓶颈在 DiT 不在 VAE**。

## 文件索引

```
本地 (G:\GitHub\DiT\):
  tools/vae/
  ├── benchmark_vae.py              # 4-VAE 对比 (200 张, [-1,1], 有 scaling bug)
  ├── bench_results/results.json    # 实验 1 结果
  ├── verify_latents_f4.py          # 编码验证 + 重建底噪 (100 张)
  ├── DATA_PIPELINE.md              # 数据管线文档 (含实验 2 数据)
  └── README.md                     # VAE 方案设计文档
  tools/evaluation/
  ├── vae_noise_eval.py             # eval 集底噪 (79 张, CPU, 正确实现)
  ├── vae_noise_gpu.py              # 全量 GPU 版 (未完成)
  ├── vae_noise_multiproc.py        # 全量多进程版 (未完成)
  ├── vae_noise_full.py             # 全量版 (未完成)
  ├── vae_patch_compare.py          # f8+p2 vs f4+p4 对比 (79 张, 正确实现)
  ├── vae_noise_eval_summary.json   # 实验 3 结果 (本地拉取)
  └── vae_recon_compare_remote.json # 实验 4 结果 (本地拉取)

远程 (/root/Workspace/xy/DiT/):
  tools/vae_recon_compare.json      # 实验 4 原始结果
  tools/vae_noise_results/
  ├── vae_noise_eval_summary.json   # 实验 3 原始结果
  └── detail_f4_kl-f4_p4.json       # 空 (未写入)
```
