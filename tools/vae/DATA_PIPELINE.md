# VAE 数据管线文档

> 2026-08-23  
> kl-f4 VAE latent 编码 + 验证流程

## 数据目录结构 (远程)

```
/root/Workspace/xy/DiT/
├── final_images/          329,715 张 PNG (256x256 RGB, 已 resize)
├── final_imgs_256/        329,715 张 (与 final_images 内容完全相同, md5 一致)
├── final_latents/         66 shards, sd-vae-ft-ema 编码, (5000, 4, 32, 32) fp16
├── final_latents_f4/      26 shards, kl-f4 编码, (5008, 3, 64, 64) fp16 [NEW]
├── final_skeleton/         329,715 张骨架图
├── final_skeleton_d3/      329,715 张 d3 骨架图
├── final_canny/            329,715 张 canny 边缘图
├── final_manifest.json    329,715 条目 (img_id → 元数据)
├── 5script/
│   ├── train_top30.csv    128,842 行 (top30 calligraphers, 6952 chars)
│   ├── eval100_top30.csv  100 行 (eval set)
│   ├── train_top6.csv     10,866 行 (top6 calligraphers, 3154 chars)
│   └── eval100_top6.csv   100 行
└── pretrained_models/
    ├── sd-vae-ft-ema/     f8, 4ch latent, 83.7M params (旧)
    └── kl-f4/              f4, 3ch latent, 55.3M params (新)
```

## 图片数据说明

- **final_images/** 和 **final_imgs_256/** 内容完全相同 (md5 一致), 均为 256×256 RGB
- 采样 500 张检查: 96.8% 为 256×256, ~3% 为其他尺寸 → encode 时强制 resize 到 256×256
- CSV 的 `image_path` 列指向 `final_images/{id}.png`

## VAE 对比 (本地 200 张书法图 benchmark)

| VAE | IO | Params | MSE | SSIM | Latent shape | Latent size |
|-----|-----|--------|-----|------|-------------|-------------|
| sd-vae-ft-ema | 3ch | 83.7M | 0.2064 | 0.7234 | (4, 32, 32) | 4,096 |
| sd-vae-ft-ema-gray | 1ch | 83.6M | 0.2063 | 0.7235 | (4, 32, 32) | 4,096 |
| **kl-f4** | 3ch | 55.3M | **0.0740** | **0.8699** | (3, 64, 64) | 12,288 |
| kl-f4-gray | 1ch | 55.3M | 0.0740 | 0.8699 | (3, 64, 64) | 12,288 |

kl-f4 的 MSE 是 sd-vae 的 36%, SSIM 高 0.15, 参数还少 34%.

## VAE 底噪 (encode→decode 重建, 100 张随机图)

| VAE | latent shape | latent size | 重建 MSE | 重建 SSIM |
|-----|-------------|-------------|----------|-----------|
| sd-vae-ft-ema (f8) | (4, 32, 32) | 4,096 | 0.003660 | 0.9655 |
| **kl-f4 (f4)** | (3, 64, 64) | 12,288 | **0.001910** | **0.9882** |

kl-f4 底噪: MSE 0.0019, SSIM 0.988 — 接近无损重建。

## DiT 训练 eval 对比 (旧 sd-vae latent)

| 实验 | 数据集 | VAE | 最终 step | eval MSE | eval SSIM | 底噪 MSE | **MSE 净增值** |
|------|--------|-----|-----------|----------|-----------|----------|--------------|
| s6 top6 diff-only | top6 (10k) | sd-vae f8 | 195k | 0.432 | 0.732 | 0.0037 | **0.428** |
| s5 top30 diff-only | top30 (129k) | sd-vae f8 | 70k | 0.841 | 0.520 | 0.0037 | **0.837** |

> eval MSE = DiT 生成质量 + VAE 底噪。底噪占比极小 (sd-vae: <1%)。
> 换 kl-f4 后底噪从 0.0037 降到 0.0019 (减半), eval 的 MSE 天花板更低。
> 但注意: 换 VAE 后 latent 分布完全不同, DiT 需从头训练, eval MSE 不可直接跨 VAE 比较。

## Scaling Factor

全量 128,842 张 latent 统计 (verify_latents_f4.py + _verify_sdvae.py):

| VAE | scaling_factor (使用) | latent mean | latent std | 1/std (理论) |
|-----|----------------------|------------|------------|------------|
| sd-vae-ft-ema | 0.18215 | 0.3178 | 1.1305 | 0.885 |
| kl-f4 | 0.102079 | -0.0559 | 0.9838 | 1.016 |

> 注: sd-vae 的 0.18215 并非 1/std(1.1305)=0.885，而是 SD 原始论文的经验值。
> latent 在 encode 后乘 scaling_factor 存储, decode 时除回。
> kl-f4 用 0.102079 是编码前估计的值, 全量统计后 std≈0.98 接近 1, 说明 scaling 基本合理。

## 编码流程

```bash
# 1. 编码 (远程 tmux)
tmux new-session -d -s encode_klf4 \
  'python tools/vae/encode_latents_klf4.py \
    --csv 5script/train_top30.csv \
    --img-root final_images \
    --vae pretrained_models/kl-f4 \
    --out final_latents_f4 \
    --shard-size 5000 --batch 16 \
    --scaling-factor 0.102079'

# 2. 验证 (encode 完成后)
tmux new-session -d -s verify_klf4 \
  'python tools/vae/verify_latents_f4.py \
    --shards final_latents_f4 \
    --vae pretrained_models/kl-f4 \
    --img-root final_images \
    --n 100 --batch 8'
```

## Shard 格式

```
shard_XXXXX.npz:
  latents: (5000, 3, 64, 64) float16   # scaled latent (z * scaling_factor)
  img_ids: (5000,) int64                # 对应 final_images/{id}.png
```

注意: 最后一个 shard 可能 < 5000 张 (128842 = 25*5000 + 3842).

## 训练配置 (s7_klf4_top30_diffonly.json)

| 参数 | 值 | 说明 |
|------|-----|------|
| model | DiT-2Cond-S/4 | patch=4, hidden=384, depth=12, heads=6 |
| vae_downscale | 4 | f4 (256→64) |
| latent_channels | 3 | kl-f4 latent ch |
| vae_scaling_factor | 0.102079 | 1/std (待全量验证) |
| latent_shards_dir | final_latents_f4 | 新编码的 shards |
| vae_path | pretrained_models/kl-f4 | kl-f4 VAE |
| global_batch_size | 64 | 待 VRAM 测试后调整 |
| image_size | 256 | 输入图片尺寸 |
| latent_size | 64 | 256/4 (DiT input_size) |
| DiT tokens | 16×16=256 | 64/4=16, 16²=256 (与 S/2 相同) |

## 文件清单

```
tools/vae/
├── convert_klf4.py              # ldm → diffusers 格式转换
├── convert_grayscale_vae.py     # 黑白 VAE 外科手术 (sum+mean)
├── estimate_scaling_factor.py   # 从 N 张图估计 scaling factor
├── encode_latents_klf4.py       # 全量编码 → shard_XXXXX.npz
├── verify_latents_f4.py         # 编码后验证: 统计 + 重建底噪
├── benchmark_vae.py              # 4-VAE 全面对比 (MSE/SSIM/latent)
├── eval_vae.py                   # 快速重建对比
├── test_grayscale_vae.py         # 黑白 VAE 快速测试
├── train_vae.py                  # VAE 微调/从头训练
├── README.md                     # 设计文档
└── bench_results/                # benchmark 结果
    ├── results.json
    └── vae_comparison.png
```
