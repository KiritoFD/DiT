# 51. moyi 参考实现的 aux 结构通道：逐行核对与对齐

> 目的：确认 `ref/moyi` 的 "edge/skeleton 辅助任务" 到底是什么，我们的实现是否**完全对齐**。
> 结论先行：**机制对齐（预测 skel/canny 的 VAE latent，与图像 latent 拼成多通道路径，
> 对 latent 算 diffusion MSE；推理时全部通道从噪声生成、只取图像 4ch 解码）**，
> 但有 **5 处差异**，其中最关键的一处（loss 的 grey/black-white 变体）**实现文件不在 ref 拷贝里**，
> 无法字节级核对——需要补 `ref/moyi/utils/diffusion.py`。

---

## 1. 参考实现逐行核对

### 1.1 训练目标：12ch latent 拼接（`train_moyun2.py:289-327`）

```python
for image, edge, skeleton, y, stroke in loader:      # 都是 256×256 RGB
    image = image.to(device); edge = edge.to(device); skeleton = skeleton.to(device)
    if (args.use_12channel == 1):                    # ← .sh 未显式传, argparse default=1
        image    = vae.encode(image).latent_dist.sample().mul_(0.18215)     # (N,4,32,32)
        edge     = vae.encode(edge).latent_dist.sample().mul_(0.18215)      # (N,4,32,32)
        skeleton = vae.encode(skeleton).latent_dist.sample().mul_(0.18215)  # (N,4,32,32)
        # custom_zero=0 -> 不减白底
        x = torch.cat((image, edge, skeleton), dim=1)   # (N,12,32,32)  ← 扩散目标
    else:
        gray_image    = rgb_to_grayscale(image)          # (N,1,256,256)
        gray_edge     = rgb_to_grayscale(edge)
        gray_skeleton = rgb_to_grayscale(skeleton)
        x = torch.cat([gray_image, gray_edge, gray_skeleton], dim=1)  # (N,3,256,256)
        x = vae.encode(x).latent_dist.sample().mul_(0.18215)          # (N,4,32,32) 单通道融合 latent
    t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
    loss_dict = diffusion.training_losses(model, x, t, dict(y=y, stroke=stroke))
    loss = loss_dict["loss"].mean()
```

**两点确认**：
1. **是"预测 skel/canny 的 VAE latent，然后 latent MSE"**——edge/skeleton 各自 VAE 编码为
   4ch latent，与图像 latent 拼接成 **12ch 扩散目标**，loss 在 latent 上算。
2. 另一条 3ch 分支是把 `[gray_image, gray_edge, gray_skeleton]` 当 3 通道图 **VAE 编码成一个
   4ch latent**（R/G/B 分别是 image/canny/skel 灰度）——单 latent 低成本的变体。

### 1.2 模型（`moyun_2.py`）

工厂 `moyun_12channel`（训练脚本 `.sh` 用 `test-models-nofeature-12channel`，二者同构）：

| 项 | 值 |
|---|---|
| depth / hidden / heads / patch | **24 / 1024 / 16 / 2**（latent 32→16×16 token）|
| in_channels | **12** |
| `learn_sigma`（Moyun 默认） | **True → 输出 24ch**（12 均值 + 12 sigma）|
| `use_stroke` | **False**（无 stroke cross-attn）|
| `use_feature` | **False**（`y_embedder = LabelEmbedder(num_classes=4793)`，单类 id）|
| RoPE | False |
| 条件 | 仅 `y`（LabelEmbedder）→ adaLN + `t` |

→ **edge/skeleton 不是条件，只是扩散目标**；模型里没有任何骨架注入通路。

### 1.3 loss（`train_moyun2.py:206-210`）

```python
diffusion = create_diffusion(
    timestep_respacing="",
    use_black_white_mse_loss=(args.use_black_white_mse_loss == 1),
    use_grey_mse_loss=(args.use_grey_mse_loss == 1),
)  # default: 1000 steps, linear noise schedule
```

- **DDPM（1000 步线性），eps-prediction**，标准 DiT `training_losses` 的 `mean_flat(mse)`
  → 对 (C,H,W) 求均值 → **12ch 等权**。
- `.sh` 里 `--use_grey_mse_loss 1` —— **loss 被额外改写**，实现位于 `utils/diffusion.py`
  （**本 ref 拷贝缺失**，无法核对是"灰度空间 MSE"还是"灰/黑白色重加权"）。
- `use_black_white_mse_loss` 默认 0（本 run 关闭）。

### 1.4 推理（`moyun/generate.py:47-59`）

```python
z = torch.randn(n, 12, latent_size, latent_size, device=device)   # 12ch 全噪声
...
samples = diffusion.p_sample_loop(model.forward_with_cfg, z.shape, z, ...)
samples, _ = samples.chunk(2, dim=0)      # CFG
samples = samples[:, 0:4, :, :]           # ← 只取图像 4ch
samples = vae.decode(samples / 0.18215).sample
```

→ **aux 通道在推理时也完全从噪声生成、没有任何输入条件**，生成后直接丢弃。
（因此它起的是"训练期辅助表征/多任务正则"的作用，不是条件。）

### 1.5 训练超参（`train_moyun2.sh`）

`batch 256 / lr 1e-4(恒定) / epochs 80000 / ckpt-every 9000 / num-classes 4793 /
data=256png_SAM_small/training_set(~1.9M) / 2 GPU / use_grey_mse_loss 1 / use_12channel 默认 1`。

---

## 2. 我们实现的映射（现状）

| ref 做法 | 我们的实现 | 状态 |
|---|---|---|
| 12ch latent target = cat(image, canny, skel) | `aux_latent_shards_dirs` → `x=cat(image, aux...)` | ✅ 对齐 |
| edge/skeleton 来自目标图本身 | `tools/data/build_aux_latents_fame_e.py`（skeletonize + Canny(GT) → VAE latent）| ✅ 对齐 |
| 对全部通道算 latent MSE | flow-matching velocity MSE（**flow** 而非 DDPM）| ⚠️ 类型不同 |
| 推理 12ch 全噪声、只取 4ch 解码 | 采样器按 `model.in_channels` 自动扩噪声；解码/评测取前 4ch | ✅ 对齐 |
| 等权 loss | 我们加了 `aux_loss_weight`，当前设 **0.1** | ❌ 偏离 |
| `learn_sigma=True`（24ch 输出）| 我们 `learn_sigma=False` | ⚠️ 差异 |
| 纯 label 条件（无骨架条件）| 我们额外有 **std-skel-g 条件**（token-add + xattn/style）| ⚠️ 差异 |
| `use_grey_mse_loss` 改写 loss | 无（该实现文件缺失）| ❓ 无法对齐 |

---

## 3. 要"完全学习对齐 ref"，需要做的 5 件事

1. **loss 权重回 1.0**：ref 是 12ch 等权（`mean_flat`）。→ `aux_loss_weight: 1.0`。
   （我们之前设 0.1 是"图像主导"的自创，不是 ref。）
2. **补 `ref/moyi/utils/diffusion.py`**：必须看到 `create_diffusion` / `GaussianDiffusion.training_losses`
   才能确认 `use_grey_mse_loss`（及其 black_white 变体）如何改写 loss。
   **这是目前唯一的硬缺口。**
3. **DDPM vs flow（口径）**：ref 是 eps-DDPM；我们是 flow velocity。
   两者都是 latent MSE，但 t 域（0..1000 vs 0..1）、目标（eps vs v）不同。
   要严格复刻需 `diffusion_type=ddpm`（会换掉 heun/flow 采样链）。
4. **`learn_sigma`**：ref True（输出 24ch，其中一半是 sigma）。对齐可开。
5. **条件**：ref **没有骨架条件**（只有 label+t）。我们若保留 std-skel-g 条件，
   就不是"纯 ref 复刻"；若要做"aux 机制本身是否有效"的干净对照，应去掉 skel 条件。

---

## 4. 待补文件清单（请提供 `ref/moyi/utils/`）

| 文件 | 用途 |
|---|---|
| `utils/diffusion.py` | `create_diffusion` + `training_losses`（含 grey/black-white mse 实现）**最关键** |
| `utils/MultiLabelNestedDataset.py` | edge/skeleton 的**来源与极性**（Canny 参数、骨架提取方式）|
| `utils/config.py` | `text2label` / 4793 类的组织方式 |
| `utils/stroke.py` | `get_ch_strokes_tensor`（仅 use_stroke=True 时需要，本 run 未用）|

---

## 5. 口径提醒（回答"0.61-0.62 是不是现在这个 eval"）

- 0.6185（fame3 base **S/2** @50k，doc 44）用的是 **eval_seen_v10 n=10，cfg 0.7**。
  与现在的 seen n=10 **同一评测集**，但 **cfg 不同**：当前新跑多用 **cfg 2.0**，
  同一 ckpt cfg 2.0 比 0.7 高 **+0.04~0.06**（E4b 实测）→ 不可直接比。
- 另外 0.61-0.62 那条线是 **S/2 + 1013 书家 + 输入层 token-add**；
  "41 冻结表 + xattn" 是后来的 **Sp(59M)** 线（c41x）。

---

## 6. 对齐后建议的最小实验

| 方案 | 配置 | 想回答的问题 |
|---|---|---|
| **R0** | S/2、纯 label 条件、4ch、flow、从零 | 基线 |
| **R1** | R0 + 12ch(image+canny+skel) 等权 MSE | **ref 的核心 claim：aux 是否有效** |
| **R2** | 我们现有 std-skel-g 配方 + 12ch 等权 | aux 在"有骨架条件"时是否还有增量 |

> 现阶段的 12ch 实验（w_aux=0.1、4 层 xattn）**既不是 R1 也不是 R2**（权重/层数都改了），
> 结论不能用来判断 ref 方法本身。
