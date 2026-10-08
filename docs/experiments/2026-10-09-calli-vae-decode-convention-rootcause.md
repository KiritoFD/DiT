# Calli-VAE 解码约定 Bug 根因报告（2026-10-09）

> **一句话**：Stage-1 Calli-VAE 训练时把 **`/scaling_factor` 放在了 decoder 的输入上**（`decode(z/sf)`），
> 又把 **DiT 离线编码的重参数化噪声丢了**（存 `mode*sf`），两个错误在推理端相叠，
> 使标准解码路径 `decode(mode)` 落在 0.88 的**灰图**上。第三个静默 bug 是 DINO 权重命名
> 在 transformers 4.x 下**整网随机初始化**。全部已定位、复现、修复。

---

## 0. TL;DR

| # | Bug | 位置 | 症状 | 修复 |
|---|---|---|---|---|
| B1 | 解码尺度错位：`decode(z/sf)` | `tools/vae/train_dino_calli_vae.py` | decoder 只认 `sample/sf`，标准约定 `decode(sample)` 直接崩（L1 0.85） | 训练改 `decode(z)`；已训权重可**解析修正**（见 §4） |
| B2 | 重参数化噪声丢失：存 `mode*sf` | 离线编码 + DiT 目标 | 后验 `std≈0.865`，`decode(mode/sf)`=0.67，必须给 `sample` | 重编码时用 `sample()` |
| B3 | 配置项未被注册 | `calli_decode_noise` 未进 CLI | config 里的开关被**静默丢弃**，补丁形同虚设 | 注册 `--calli-decode-noise`（train/eval 两处） |
| B4 | DINO 权重命名硬改 | `load_dino_model()` | transformers 4.x 下主干权重 **全 miss**、整网随机初始化 | 按目标模型键自适应命名 + 缺失即 raise |

---

## 1. 三个环节的约定必须一致

一个 latent-diffusion 系统里，同一个"潜变量约定"出现在 **三** 个地方；只要有一处不一致，就会在推理端爆掉：

| 环节 | 标准写法（SD-VAE） | 本次实际 |
|---|---|---|
| ① 训 VAE 时的 decode | `x_recon = vae.decode(z)`，`z = posterior.sample()` | `x_recon = vae.decode(z / sf)` ❌ |
| ② 给 DiT 离线编码 | `z = posterior.sample()`，存 `z * sf` | `latent_dist.mode() * sf` ❌ |
| ③ 推理解码 | `x = vae.decode(pred / sf)` → 即 `decode(z)` | `pred = mode*sf` ⇒ `decode(mode)` ❌ |

`sf = vae.config.scaling_factor = 0.18215`。

- ① 把 decoder 的输入尺度**放大**了 `1/sf ≈ 5.49` 倍 → 训出"只认 `sample/sf`"的非标准解码器。
- ② 丢掉了重参数化噪声（`sample = mode + std·ε`），DiT 学到的是**去噪后的均量** `mode*sf`。
- ③ 于是推理喂给 decoder 的是 `mode`，而该 decoder 想要的是 `sample/sf` —— **尺度错 + 噪声缺**，双错叠加 → 灰图。

> 注：随后的 `calli_decode_noise`（`src/utils/calli_decode_noise.py`）在解码端用 `mode` 预测逐元素
> `std` 再注噪，是对**已训 v71 ckpt**的补救（自检恢复 99%）。但那是补丁，不是"约定正确"。

---

## 2. 实测铁证：同一个探针，base VAE vs Calli-VAE 的约定完全相反

工具：`tools/vae/check_vae_convention.py`（对同一批真图枚举 4 种解码输入，谁 L1 最低谁就是真实约定）。

**base `sd-vae-ft-ema`**（4090，8 张 `UNIFIED_RAW` 真图）：

| decode 输入 | L1 | PSNR(dB) |
|---|---|---|
| `sample` | **0.0106** ✅ | 35.54 |
| `sample/sf` | 0.4173 ❌ | 26.64 |

**Calli-VAE `step_25000`**（48，8 张 evalset GT 真图）：

| decode 输入 | L1 | PSNR(dB) |
|---|---|---|
| `sample` | 0.8479 ❌ | 7.17 |
| `sample/sf` | **0.0378** ✅ | 19.37 |
| `mode` | 0.8816 ❌ | 6.96 |
| `mode/sf` | 0.6702 | 9.16 |

结论：
1. **标准约定是 `decode(sample)`**（base VAE 如此；喂 `sample/sf` 直接废掉）。
2. **Calli-VAE 的约定被训练反了**（只认 `sample/sf`）→ B1 成立。
3. 即便尺度修正到 `/sf`，`decode(mode/sf)=0.670` 仍远差于 `decode(sample/sf)=0.038` → B2（噪声缺失）独立成立。
   后验 `std≈0.865`（very wide），decoder 实质上是个**随机去噪器**，必须喂真实 `sample`。

---

## 3. 为什么 v71 的已存评测全是灰图

- v71 的 DiT 目标 = `mode*sf`（B2 的产物，已烙进 shard `exp-std/data/shards_img_aug_calli`）；
- in-mem eval 解码 = `vae.decode(lat / sf)` = `decode(mode*sf/sf)` = `decode(mode)`；
- 而 Calli decoder 期待 `sample/sf` ⇒ 输入既**尺度小 5.49 倍**、又**缺噪声** ⇒ 输出近似常量灰。
- 实测：修前 `SSIM 0.514 / MSE 0.783 / LPIPS 0.840 / ink_IoU 0.000`；
  修后（`--calli-decode-noise true`）`SSIM 0.573 / MSE 0.874 / LPIPS 0.424 / ink_IoU 0.283`。

---

## 4. 解析补救：零训练成本把 VAE 改成标准约定（逐元素恒等）

因为 decoder 相对标准的**唯一**偏差就是输入尺度，而 `AutoencoderKL.decode` 实现为
`dec = self.decoder(self.post_quant_conv(z))`，所以令

```
post_quant_conv.weight  ←  post_quant_conv.weight / sf        # bias 不动
```

便有

```
decode_new(z) = decoder(post_quant_conv_old(z / sf)) = decode_old(z / sf)
```

**逐元素恒等**，encoder / 潜空间 / 后验一个参数不动。

工具：`tools/vae/fix_vae_decode_convention.py`。实测（48，8 张 GT，CPU）：

```
[before] posterior.std=0.8653  L1: sample=0.8474  sample/sf=0.0381  mode=0.8816  mode/sf=0.6702
[fix]    post_quant_conv.weight 缩放 1/0.18215 = 5.4900
[after ] posterior.std=0.8653  L1: sample=0.0381  sample/sf=0.6450  mode=0.6702  mode/sf=0.4128
[fix]    校验: |after.decode(sample) - before.decode(sample/sf)| = 0.000000 (✓ 恒等)
[conv]   ✓ 该 VAE 符合标准约定 decode(sample)
```

产出：`experiments/calli_vae_dino/calli_vae_step_25000_stdconv`（标准 drop-in：
`encode→sample()`，扩散目标 `z*sf`，推理 `decode(pred/sf)`，**不再需要 `calli_decode_noise`**）。

---

## 5. 另外两个静默 bug

### B3 — `calli_decode_noise` 配置项根本没生效
`src/eval/in_mem_eval.py` 用 `getattr(args, "calli_decode_noise", False)` 读取，但该 key
**从未在 `src/train/cli.py` / `src/eval/cli.py` 注册**，于是 config 里的 `"calli_decode_noise": true`
被配置加载器**静默丢弃** → 分支永远不执行 → 修完"看起来没变"。
修复：两处 CLI 注册 `--calli-decode-noise`。

### B4 — DINO 权重在 transformers 4.x 下整网随机初始化
`load_dino_model()` 过去**无条件**把权重键改成 transformers 5.x 命名（`attention.q_proj`），
但 4.x 用的是 `attention.attention.query`。在 4090 的 transformers 4.36.2 上 ⇒ 主干权重全 miss：
```
[dino warn] missing keys: ['encoder.layer.0.attention.attention.query.weight', ...]
```
`strict=False` 只打印一行 warning，训练照跑，**感知损失彻底失效**（最危险的静默错误）。
修复：按目标模型自己的 `state_dict()` 键决定是否改名，且主干权重缺失即 `raise`。

---

## 6. 修复后的实现约定（自检即契约）

`tools/vae/train_dino_calli_vae.py` 现在：

1. 训练解码 **标准约定**：`x_recon = vae.decode(posterior.sample())`；
2. **开机自检**（`verify_decode_convention`）：实测 base VAE 的 `decode(sample)` /
   `decode(sample/sf)` / `decode(mode)` / `decode(mode/sf)` 四种 L1，判定"缩放约定"是否标准，
   并与本脚本的训练约定对照；
3. **完训自检**：训练后必须回到 `decode(sample)`；
4. 周期日志新增 `posterior.std`（本次微调最关键的可观测量）。

> 注意（重要的方向性更正）：`KL=½(σ²+μ²−1−lnσ²)` 在 `σ=1` 取极小，所以 KL 项是把 `σ` **推向 1**
> 而非 0。当前 `σ≈0.865`（≈1）说明 KL 在主导、稀疏笔迹图的重建信号偏弱。要收紧后验应当**降** `w_kl`
> （或加权重建），**不是**升 `w_kl`。此结论待微调日志（`Std` 列）实测确认，勿凭直觉调节。

---

## 7. 附：v71 vs 同期 SD-VAE 双胞胎（同步数对照）

v71 config 声明为**单变量实验**（相对 `v68_aug_sp_c2ot` 只换目标 latent 的 VAE，其余逐字相同）。
同步数 `step=30000`、同评测集 `eval200fix`(n=187)：

| 指标 | v68_aug_sp_c2ot (SD-VAE) | v71_callivae_sp_c2ot (Calli-VAE) | 胜 |
|---|---|---|---|
| SSIM ↑ | 0.5495 | **0.5731** | v71 |
| MSE ↓ | 0.9635 | **0.8739** | v71 |
| LPIPS ↓ | **0.3836** | 0.4239 | v68 |
| ink_SSIM ↑ | 0.3898 | **0.4112** | v71 |
| ink_IoU ↑ | 0.2359 | **0.2826** | v71 |
| skel_IoU ↑ | 0.0149 | **0.0169** | v71 |
| frag_ratio ↓ | 2.036 | **1.488** | v71 |
| nn_SSIM ↑ | 0.6120 | **0.6297** | v71 |

即：同步数下 Calli-VAE 在**结构/像素/笔迹**类指标全面领先，仅**感知 LPIPS** 落后。
注意两者都只到 30k（15%），`v68_aug_sp_c2ot` 在 200k 达 `SSIM 0.6149 / LPIPS 0.3249`，故本表只说明
"同预算下谁领先"，不代表收敛点。

---

## 8. 运维政策

- **所有 VAE 类训练统一在 4090 主机进行**（网络/数据本地化考虑）；48 保留给重算力消融。
- Calli 权重正本：本地 `...\Temp\opencode\CalligVAE_pull\CalligVAE_upload`（MD5 校验通过）
  与 ModelScope `ArcherFD/CalligVAE`。

## 9. 相关文件

- `tools/vae/check_vae_convention.py`（实测某 VAE 的解码约定）
- `tools/vae/fix_vae_decode_convention.py`（解析修正到标准约定，恒等校验）
- `tools/vae/train_dino_calli_vae.py`（约定修正 + DINO 命名修正 + 自检 + Std 日志）
- `tools/re_eval.py`（对已训 ckpt 单独重跑评测，走已修好的 in-mem eval 路径）
- `src/train/cli.py`, `src/eval/cli.py`（注册 `--calli-decode-noise`）
- 前置补丁提交：`5bd564d`（decode 侧注噪 + 正确 Calli 编解码 wrapper）
