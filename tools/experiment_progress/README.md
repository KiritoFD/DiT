# 实验汇总与海报

海报说明：每张海报中，**一行 = 一个 ckpt**。每行上方是一条**黑底白字大字的标签行**，把该 ckpt 的 **step 和评估指标（MSE / SSIM）写在同一行**；标签行下方是 10 张样本（show5 unseen + seen5 train，各按 `img | canny | skel` 三列铺开）。最底部一条 GT 标签行 + 真值图。

- canny：灰度 Rec.601 → Sobel 幅值 >150 二值
- skel：灰度 >127 二值（均值>127 取反适配白/黑底）→ Zhang-Suen 细化
- 指标来源：远程 CPU 评估 `eval_auto_<step>.json`（eval100）。`SSIM` 为表内最后一档有评估结果的 step 值。

---

## 1. S_kailishu_noloss_2factor — `s2_fromscratch_2factor.png`
![对比度海报](s2_fromscratch_2factor.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**S**/2（from scratch，`pretrained=null`） |
| 数据 | 楷书单书 `kailishu_train.csv` |
| 条件 | 纯两因子：`callig(128)` × `char(256)` class，`factorized_add` |
| 标准字形条件 | ❌ 无（`w_glyph_cond=false`） |
| 中间步监督 | ❌ 无（`w_std_mid=0`） |
| 结构损失 | ❌ 无（无 canny/skel） |
| batch / steps | 128 / 50k（last） |
| 最终指标 | MSE 0.702 · **SSIM 0.577** |

> 基线之一：楷书单书、纯两因子、无任何结构辅助。

---

## 2. S_kailishu_mid — `s2_fromscratch_glyphmid.png`
![对比度海报](s2_fromscratch_glyphmid.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**S**/2（from scratch） |
| 数据 | 楷书单书 `kailishu_train.csv` |
| 条件 | 两因子 + **标准字形条件**（`w_glyph_cond=true`, `glyph_scale_init=0.4`） |
| 中间步监督 | ✔ **标准字形中间监督**（`w_std_mid=0.8`） |
| 结构损失 | ❌ 无 |
| batch / steps | 128 / 50k（last） |
| 最终指标 | MSE 1.284 · SSIM 0.453 |

> 对比基线 1：加了标准字形条件 + 中间步字形监督，SSIM 反而低于纯两因子基线。

---

## 3. S_top30_noloss — `s5_2factor_top30.png`
![对比度海报](s5_2factor_top30.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**S**/2（from scratch） |
| 数据 | **5 书 top30** `assets/train_top30.csv`（128842 行） |
| 条件 | 纯两因子 `callig × char`，`factorized_add` |
| 标准字形条件 / 中间步 | ❌ / ❌ |
| 结构损失 | ❌ 无 |
| batch / steps | **192**（大 batch）/ 200k max，eval 至 70k |
| 最终指标 | MSE 0.841 · **SSIM 0.520** |

> 5 书（楷篆草行隶）× top-30 书家；大 batch 纯两因子主基线之一。

---

## 4. S_top30_pixel_cs_post50k — `s5_2factor_struct_post50k.png`
![对比度海报](s5_2factor_struct_post50k.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**S**/2（from scratch） |
| 数据 | 5 书 top30 `train_top30.csv` |
| 条件 | 纯两因子 + **像素 canny + skel 结构损失**（`w_canny=0.5`, `w_skel=0.5`） |
| 结构解码 | 像素域（256px） |
| batch / steps | 12（内存受限小 batch）/ 50k 起继续 |
| 最终指标 | MSE 0.969 · SSIM 0.508 |

> 结构后置版：50k 后才叠加像素结构损失。

---

## 5. B_top30_latentC — `s5_2factor_B_latentstruct.png`
![对比度海报](s5_2factor_B_latentstruct.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**B**/2（from scratch，hidden 768） |
| 数据 | 5 书 top30 `train_top30.csv` |
| 条件 | 两因子 + **latent 域 canny 结构损失**（`w_latent_canny=0.5`, `latent_struct_max_t=500`） |
| 结构解码 | latent 域 |
| batch / steps | 64 / 200k max，eval 至 130k |
| 最终指标 | MSE 0.883 · **SSIM 0.507** |

> B 模型 + latent 结构损失，收敛到与 S 相近的 SSIM。

---

## 6. B_top30_latentC_pixelsk_opt — `s5_2factor_B_latentstruct_pixelsk_opt.png`
![对比度海报](s5_2factor_B_latentstruct_pixelsk_opt.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**B**/2（from scratch） |
| 数据 | 5 书 top30 `train_top30.csv` |
| 条件 | 两因子 + **像素 skel 结构损失**（`use_skel`, `w_skel=1.0`，无 canny） |
| 结构解码 | 像素域，**bf16**，`struct_subset=16` |
| batch / steps | 64 / 400k max，eval 至 115k |
| 最终指标 | MSE 0.871 · SSIM 0.391 |

> 仅像素骨架监督的优化版，SSIM 最低。

---

## 7. B_top30_cs_bf16 — `s5_2factor_B_canny05_pixelsk.png`
![对比度海报](s5_2factor_B_canny05_pixelsk.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**B**/2（from scratch） |
| 数据 | 5 书 top30 `train_top30.csv` |
| 条件 | 两因子 + **像素 canny + skel**（`w_canny=0.5`, `w_skel=1.0`） |
| 结构解码 | 像素域，**bf16**，`struct_subset=16` |
| batch / steps | 64 / 400k max，eval 至 325k |
| 最终指标 | MSE 0.782 · **SSIM 0.482** |

> B + 完整像素 canny/skel（bf16 解码），训练较深。

---

## 8. B_top30_cs_fp32 — `s5_2factor_B_pixelfp32.png`（运行中）
![对比度海报](s5_2factor_B_pixelfp32.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**B**/2（from scratch） |
| 数据 | 5 书 top30 `train_top30.csv` |
| 条件 | 两因子 + **像素 canny + skel**（`w_canny=0.5`, `w_skel=1.0`） |
| 结构解码 | 像素域，**fp32**（`struct_decode_bf16=false`, `struct_decode_scale=1.0`, `struct_subset=8`） |
| batch / steps | 32 / 300k max，eval 至 20k（**仍在训练**） |
| 最终指标（20k） | MSE 1.407 · SSIM 0.156（未收敛，仍上升） |

> ⚠️ 朴素 fp32 像素解码（大显存占用，subset8 在 24.5G 4090 上安全）；目前只到 20k，指标仍在爬升，未到收敛。

---

## 9. v3b_XL_glyph_kailishu — `v3b_xl_glyphcond.png`
![对比度海报](v3b_xl_glyphcond.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**XL**/2（预训练 `DiT-XL-2-256x256.pt` + **LoRA** r=16） |
| 数据 | 楷书单书 `kailishu_train.csv` |
| 条件 | 两因子 + **标准字形条件**（`w_glyph_cond=true`），融合 `xl_highdim` |
| 中间步 / 结构损失 | ❌ / ❌ |
| batch / steps | 16 / 30k（last） |
| 最终指标 | MSE 0.992 · **SSIM 0.489** |

> XL 预训练 + LoRA + 标准字形条件。

---

## 10. v3c_XL_midstep_kailishu — `v3c_xl_glyphcond_midstep.png`（未完整评估）
![对比度海报](v3c_xl_glyphcond_midstep.png)

| 项 | 值 |
|---|---|
| 模型 | DiT-2Cond-**XL**/2（预训练 + LoRA r=16） |
| 数据 | 楷书单书 `kailishu_train.csv` |
| 条件 | 两因子 + 标准字形条件 + **中间步标准字形监督**（`w_std_mid=0.8`） |
| batch / steps | 16 / 30k max |
| 指标 | 仅 1 档 show5 已保存，**eval100 未跑完，无 MSE/SSIM** |

> v3b 的中间步版本；目前只跑了 1 档 ckpt 的显示样本，评估尚未出指标。

---

## 备注
- 海报与指标来自本地缓存 `_exp_cache/<exp>/`（远程拉回的一次性缓存，不会重复下载）。
- 指标（MSE/SSIM）为 eval100 上最末一档有评估结果的 ckpt；`ssim` 数值越小表示生成图与真图结构差异越大（注意：这里 SSE 方向，SSIM 高才好，表格中以 **粗体** 标出各实验最终 SSIM）。
- 运行中实验：`B_top30_cs_fp32`（第 8 项）；`v3c`（第 10 项）评估未完成。
