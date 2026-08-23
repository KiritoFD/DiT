# VAE 方案论证与设计

> 2026-08-23  
> 目标：评估 kl-f4 VAE、sd-vae-ft-ema (f8) 微调、黑白 VAE 三种方案的可行性。

---

## 1. 现状

| 属性 | sd-vae-ft-ema (当前) | kl-f4 (新下载) |
|------|---------------------|----------------|
| 下采样倍率 | f8 (256→32) | f4 (256→64) |
| latent channels | 4 | 3 |
| input channels | 3 (RGB) | 3 (RGB) |
| latent shape (256²) | (4, 32, 32) | (3, 64, 64) |
| 参数量 | ~84M | ~? (待测) |
| 架构 | AutoencoderKL (diffusers) | AutoencoderKL (ldm, pytorch-lightning) |
| 训练数据 | 自然图像 | ? (可能也是自然图像) |

### 关键问题
- 书法是 **黑白图** (1 channel)，但 VAE 的 conv_in 是 3 channel → 当前把灰度复制 3 份
- f8 的 latent 只有 4×32×32=4096 个值，**信息瓶颈太紧**，书法细节丢失
- f4 的 latent 有 3×64×64=12288 个值，**3 倍信息量**，可能保留更多笔画细节

## 2. 三种方案

### 方案 A: 直接用 kl-f4 VAE（零成本验证）
- **做法**：把 kl-f4 的权重转成 diffusers AutoencoderKL 格式，替换当前 VAE
- **改动**：
  1. latent shape 从 (4,32,32) → (3,64,64)
  2. DiT 的 `x_embedder` in_channels 从 4 → 3
  3. `final_layer` out_channels 从 4 → 3（或 6 如果 learn_sigma）
  4. patch_size 从 2 → 4（64/16=4 个 token → 不够，用 patch=2 → 32×32=1024 token）
  5. 所有 latent 相关代码（dataset, train, eval）改 latent_channels=3, spatial=64
- **优点**：零训练成本，直接验证 f4 是否更好
- **缺点**：kl-f4 也是自然图像训练的，不一定适合书法
- **风险**：latent space 分布不同，需要重新训练 DiT

### 方案 B: 微调 sd-vae-ft-ema（f8 保持，适配书法）
- **做法**：冻结 encoder，只微调 decoder；或全微调
- **改动**：
  1. 把输入从 3ch 改成 1ch（conv_in.weight[:, :1] 取第一个通道初始化）
  2. decoder.conv_out 改成 1ch 输出
  3. 在 MCCD 全量数据上微调 10-50 epoch
- **优点**：保持 f8 latent 兼容（现有 195k/70k ckpt 可直接用）
- **缺点**：f8 瓶颈不变，只是 decoder 更懂书法
- **关键**：encoder 不改 → latent 不变 → DiT 不用重训

### 方案 C: 训练黑白专用 VAE（f4, 1ch）
- **做法**：从 kl-f4 架构出发，改 input/output=1ch，在 MCCD 全量上从头训
- **改动**：
  1. conv_in: 128×3×3×3 → 128×1×3×3 (复制 mean 初始化)
  2. decoder.conv_out: 3→1
  3. 从头训练 encoder + decoder（KL + L1 + perceptual loss）
  4. latent 3×64×64, 适合书法
- **优点**：完全适配书法，latent 紧凑高效
- **缺点**：训练成本高（需要全量 MCCD, ~24h），且 DiT 要从头训
- **loss**: L1 recon + KL(0.5) + perceptual(VGG) + GAN（可选）

## 3. 推荐路径

```
Step 1 (方案 B): 微调 sd-vae decoder → 快速验证书法 decode 质量
  - 1h 训练, 不影响现有 DiT
  - 如果 decode 质量明显提升 → 值得

Step 2 (方案 C): 训练黑白 f4 VAE → 长期最优
  - 需要 tools/vae/train_vae.py
  - MCCD 全量 (329k 图), f4, 1ch, KL+L1+perceptual
  - 训完后 DiT 需要重新 encode latents + 重训

Step 3 (方案 A): 直接试 kl-f4 → 如果方案 C 训完发现 f4 好
  - 转格式, 重 encode latents, 重训 DiT
```

## 4. 文件

```
tools/vae/
├── convert_klf4.py        # kl-f4 ldm → diffusers AutoencoderKL
├── train_vae.py           # VAE 微调/从头训练 (方案 B+C)
├── eval_vae.py            # VAE recon 质量 (MSE/SSIM/LPIPS)
└── README.md              # 本文档
```
