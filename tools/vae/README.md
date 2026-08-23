# VAE 方案论证与设计

> 2026-08-23 (updated)
> 目标：评估 kl-f4 VAE、sd-vae-ft-ema (f8) 微调、黑白 VAE 三种方案的可行性。

---

## 1. 现状

| 属性 | sd-vae-ft-ema (旧) | kl-f4 (新) |
|------|---------------------|----------------|
| 下采样倍率 | f8 (256→32) | f4 (256→64) |
| latent channels | 4 | 3 |
| input channels | 3 (RGB) | 3 (RGB) |
| latent shape (256²) | (4, 32, 32) | (3, 64, 64) |
| latent size | 4,096 | 12,288 (3× 信息量) |
| 参数量 | 83.7M | 55.3M (少 34%) |
| 架构 | AutoencoderKL (diffusers) | AutoencoderKL (ldm→diffusers 转换) |
| scaling_factor | 0.18215 | 0.102079 |
| 重建底噪 MSE | 0.0037 | 0.0019 (减半) |
| 重建底噪 SSIM | 0.9655 | 0.9882 |

### 关键发现
- kl-f4 重建底噪 MSE 仅为 sd-vae 的 52%，SSIM 接近无损 (0.988)
- kl-f4 参数更少 (55M vs 84M)，latent 信息量更大 (3×)
- 黑白 VAE (1ch surgery) 重建质量与 3ch 完全一致，但未采用 (省不了 latent ch)

## 2. 三种方案

### 方案 A: 直接用 kl-f4 VAE ✅ (已采用)
- **做法**：把 kl-f4 的权重转成 diffusers AutoencoderKL 格式，替换当前 VAE
- **改动**：
  1. latent shape 从 (4,32,32) → (3,64,64)
  2. DiT 的 `x_embedder` in_channels 从 4 → 3
  3. `final_layer` out_channels 从 4 → 3
  4. patch_size 从 2 → 4（64/4=16 → 16×16=256 tokens, 与 S/2 相同）
  5. 所有 latent 相关代码（dataset, train, eval）改 latent_channels=3, spatial=64
- **优点**：零 VAE 训练成本，直接验证 f4 是否更好
- **缺点**：kl-f4 也是自然图像训练的，不一定完全适配书法
- **状态**：已完成转换 + 编码 + 验证 + DiT 训练已启动

### 方案 B: 微调 sd-vae-ft-ema decoder (未采用)
- **做法**：冻结 encoder，只微调 decoder
- **优点**：保持 f8 latent 兼容（现有 ckpt 可直接用）
- **缺点**：f8 瓶颈不变，信息量仍只有 4096
- **状态**：备选方案，如 kl-f4 效果不佳再考虑

### 方案 C: 训练黑白专用 VAE (备选)
- **做法**：从 kl-f4 架构出发，改 input/output=1ch，在 MCCD 全量上从头训
- **优点**：完全适配书法，latent 紧凑高效
- **缺点**：训练成本高（~24h），且 DiT 要从头训
- **状态**：黑白 surgery 已验证 (零质量损失)，但未采用 — 不省 latent channels

## 3. 已完成的工作

### VAE 转换 (convert_klf4.py)
- ldm (pytorch-lightning) → diffusers AutoencoderKL
- 204 keys 完美匹配 (0 missing, 0 unexpected)
- 关键 remap: down/up block 编号、mid attention q/k/v→to_q/to_k/to_v、1×1 conv→linear

### 黑白 VAE 手术 (convert_grayscale_vae.py)
- conv_in: `weight.sum(dim=1, keepdim=True)` → [C_out, 1, K, K]
- conv_out: `weight.mean(dim=0, keepdim=True)` → [1, C_in, K, K]
- 验证: MSE/SSIM 与 3ch 完全一致 (零质量损失)
- `register_to_config(in_channels=1, out_channels=1)` 持久化 config

### Benchmark (benchmark_vae.py)
- 4 VAE 全面对比: sd-vae / sd-vae-gray / kl-f4 / kl-f4-gray
- 指标: MSE, SSIM, latent stats, 对比网格图
- 黑白 vs 3ch: 完全一致 (MSE 差异 <1e-6)

### 全量编码 (encode_latents_klf4.py)
- 128,842 张 → 26 shards, (5008, 3, 64, 64) fp16
- scaling_factor=0.102079 (encode 前 1/std 估计)
- 强制 resize 到 256×256 (处理 ~3% 非标尺寸图片)

### 验证 (verify_latents_f4.py + _verify_sdvae.py)
- 全量 latent 统计: kl-f4 mean=-0.056, std=0.984; sd-vae mean=0.318, std=1.131
- 重建底噪: kl-f4 MSE=0.0019, sd-vae MSE=0.0037
- 修复: fp16 overflow (用 float64 sum), SSIM 多通道 (per-channel 递归)

### DiT 训练 (s7_klf4_top30_diffonly)
- DiT-2Cond-S/4, 从头训练, 41.8M params
- batch=224, bf16 autocast, EMA on
- max_steps=600000, early stop (patience=6, min=60k)
- 3.51 steps/s, 19.7G VRAM (4090)
- auto_eval_cpu 独立进程轮询评测 (不阻塞 GPU)

## 4. 文件清单

```
tools/vae/
├── convert_klf4.py              # ldm → diffusers 格式转换 (204 keys)
├── convert_grayscale_vae.py     # 黑白 VAE 外科手术 (sum+mean)
├── estimate_scaling_factor.py   # 从 N 张图估计 scaling factor
├── encode_latents_klf4.py       # 全量编码 → shard_XXXXX.npz
├── verify_latents_f4.py         # 编码后验证: 统计 + 重建底噪
├── benchmark_vae.py             # 4-VAE 全面对比 (MSE/SSIM/latent)
├── eval_vae.py                  # 快速重建对比
├── test_grayscale_vae.py        # 黑白 VAE 快速测试
├── train_vae.py                 # VAE 微调/从头训练 (方案 B+C)
├── DATA_PIPELINE.md             # 数据管线完整文档
├── README.md                     # 本文档
└── bench_results/                # benchmark 结果
    ├── results.json
    └── vae_comparison.png
```

## 5. 代码改动 (根目录)

### models.py
- 新增 `DiT_2Cond_S_4` (patch=4) 到 `DiT_2Cond_models` dict
- DiT-2Cond-S/4: patch=4, hidden=384, depth=12, heads=6, input_size=64

### train.py
- 新增 args: `--vae-downscale`(8), `--latent-channels`(4), `--vae-in-channels`(3), 
  `--vae-out-channels`(3), `--vae-scaling-factor`(0.18215)
- `latent_size = args.image_size // vae_downscale`
- Model 构造传 `in_channels=getattr(args, 'latent_channels', 4)`
- MockVAE 使用动态 `_vae_ds/_vae_lc/_vae_oc/_vae_sf`
- 所有 `0.18215` → `_vae_sf` 变量
- 修复: `use_latent` referenced before assignment

### latent_dataset.py
- Auto-detect latent shape from first shard: `self.latent_channels`, `self.latent_spatial`
- Preload array 使用动态 shape
- `g_t = torch.zeros(self.latent_channels, self.latent_spatial, self.latent_spatial)`

### eval_auto.py
- `eval_gen_in_memory` / `eval_in_memory` 接受 `latent_channels`, `latent_spatial`, `scaling_factor` 参数
- Noise shape 动态
- 所有 `0.18215` → `scaling_factor` 参数
- 所有 `4, 32, 32` → 动态

### auto_eval_cpu.py
- `build_model` 使用 `vae_downscale` 算 latent_size
- 传 `in_channels` 给 model 构造
- `_vae_params()` helper 提取 VAE 参数
- cfg dict 包含 latent_channels/latent_spatial/scaling_factor (从 ckpt_args)
