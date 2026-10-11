# Calli-VAE 训练配方（Recipe）与两轮实验结论

> 2026-10-11 立。本文档说明 **VAE 该用什么 loss、为什么、怎么训、怎么验收**。
> 与 [`vae_latent_contract.md`](docs/vae_latent_contract.md) 配套：**那份管"接口/尺度约定"（latent 契约），本文管"训练配方"**。
>
> 一句话结论：**VAE 必须是纯物理压缩器 —— 用 L1 + LPIPS + PatchGAN + 极小 KL + 像素域高频项；
> 绝不能挂 DINO（或任何语义特征）**。REPA/DINO 的位置在 **DiT 主干**（REPA / REPA-E）。

---

## 1. 为什么必须改：上一轮（DINO 版 Calli-VAE）的结果与三个疑点

### 1.1 上一轮做了什么

| 项 | 值 |
|---|---|
| 初始化 | `experiments/calli_vae_step_25000_stdconv`（更早一轮 DINO 微调的产物） |
| 语料 | `/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv`（**393,486 张**） |
| 损失 | **`1.0·L1 + 1e-6·KL + 0.1·DINO_struct + 0.5·DINO_style`** ← **无 LPIPS、无判别器** |
| 优化 | batch 8 × accum 4（等效 32）、lr 5e-5 cosine（eta_min 1e-6）、20,000 步、0.35 opt-step/s ≈ 16h |
| 冻结 | step **12500** → `experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv` |
| DINO 用途 | `struct` = DINO patch 特征 MSE；`style` = DINO 特征跨 patch 的 std MSE（注释写"强迫保留干枯飞白"） |

### 1.2 它到底好不好？——**重建很好，但高频比 SD 还差**

同一批 187 张字帖、同一口径实测（往返 = encode→decode）：

| VAE | 往返 PSNR | 往返 SSIM | **顶带高频份额** | 墨量 | **后验相对宽度** | raw latent std / mean |
|---|---|---|---|---|---|---|
| SD-VAE (ft-ema, 4ch) | 26.80 | 0.9738 | **0.816×** | 0.999× | **4e-5（点质量）** | 5.87 / — |
| **我们 DINO 版 Calli-VAE** | **31.21** | **0.9822** | **0.783×** ← 更差 | 1.001× | **0.372（宽 4 个数量级）** | 1.085 / −0.1155 |
| FLUX-16ch（换尺寸，未采用） | — | — | 0.97× | — | 2e-4（点质量） | — |

**关键异常**：**PSNR/SSIM 都更好，高频却更差**。重建"像素上更准"，但"细节更糊"。

### 1.3 下游暴露的三个疑点

| # | 现象（实测） | 说明 |
|---|---|---|
| 1 | **DiT 只用到 VAE 天花板的 76~79%** | v72 HF/GT 0.622（天花板 0.783 → 利用率 0.794）；v76 HF/GT 0.629（天花板 0.828 → **0.759**）→ 抬高天花板**不会**自动转化为端到端收益 |
| 2 | **SDE 采样器对 DINO 版完全无效**（γ0→γ0.5：SSIM **−0.0025**、frag **+12%**） | 而它对 SD-VAE 版（v68）贡献 SSIM **+0.0148**、frag **−53%** |
| 3 | **latent 尺度自由漂移**：raw std 5.87 → 1.085、mean 0 → −0.1155 | 因为 L1/DINO/KL **都不约束 z 的全局尺度**（解码器可吸收任意常数缩放）→ 直接导致契约 E5（存盘 std 只有 0.198 而应先 ~1.0） |

**疑点 2 有两个候选机理，都可分辨**（留给下一轮实验判决）：

| 候选 | 内容 | 支持证据 | 判别方法 |
|---|---|---|---|
| **(A) 流形被语义化** | DINO 逼 VAE 对齐语义 → 隐空间局部平滑性被破坏 → 注入 SDE 噪声被解码成伪影 | **尚无直接证据** | 干净 VAE 训好后：若**基线 frag 仍低、但 SDE 增益回归** → 支持 (A) |
| **(B) 无事可修** | DINO 版基线本来就断线少（frag 2.57 vs v68 的 5.06）→ SDE 的"连通性修复"无从发挥 | SDE 对 v68 的增益几乎全是 frag −53% | 若干净 VAE 下 SDE 仍无效 → 支持 (B) |

---

## 2. 为什么"在 VAE 里挂 DINO"是错的（三条独立证据）

1. **DINO 特征被设计成对低级纹理"不变"** —— 这是自监督学习的卖点。用一把**看不见纹理的尺子**去监督一个**像素级压缩器**，监督信号与目标错位。
2. **工业界的 VAE 配方里没有语义特征**：
   - **SD1.5（kl-f8, LDM/Taming）**：`L1 + LPIPS(VGG) + KL(1e-6) + PatchGAN(Hinge, disc 0.5 + 自适应权重)`；
     官方 model card 明写 ft-EMA 用 `L1 + LPIPS`、ft-MSE 用 `MSE + 0.1·LPIPS`（盘上可验证）。
   - **FLUX（16ch / **8×** 下采样；注意"16ch/4×"这个组合不存在）**：同族配方，判别器更强（多尺度/特征匹配），
     并加**频域损失**与 **LeCam 正则**；config 里带 `shift_factor=0.1159`（= 我们实测到的均值偏移来源）。
   - → **我们把 SD 那两项"纹理压力"（LPIPS + GAN）整个换成了 DINO**，这正好解释了"PSNR 更好但高频更差"。
3. **我们自己的消融**：唯一把高频拉起来的是**像素域**项 —— 加 `topband`（直接优化顶带份额）后
   往返顶带 **0.783 → 0.828**（4090 口径 0.831→**0.902**）；而 DINO 的 struct/style **从未**把 HF 拉起来。

**结论：DINO 的用途是"给生成主干一个语义引导"，位置在 DiT（REPA / REPA-E），不在 VAE。**

---

## 3. 本轮配方（正在跑：`calli_vae_physics`）

### 3.1 总损失

```
L = 1.0 · L1                      # 重建（低频结构）
  + 1.0 · LPIPS(VGG)              # 感知（SD1.5 官方权重）
  + 0.5 · LaplacianEdge           # 书法特化：边缘位置
  + 1.0 · topband                 # 书法特化：顶带高频份额（我们唯一验证有效的 HF 杠杆）
  + 1e-6 · KL                     # 极小，仅保证流形不爆（与 SD 同量级）
  + 0.5 · λ_adv · Hinge_G         # PatchGAN 对抗；λ_adv 自适应；前 5k 步为 0
  − DINO_struct / DINO_style      # ★ 删除（w_struct = w_style = 0，且完全不加载 DINO 模型）
```

### 3.2 权重来自**实测**，不是拍脑袋

方法：拿冻结 VAE 的往返当"好重建"，量各项在**收敛态**的真实量级（187 张字帖）：

| 项 | 收敛态量级 | 采用权重 | 加权后 |
|---|---|---|---|
| L1 | 0.0108 | 1.0 | 0.0108 |
| LPIPS | 0.1185 | **1.0** | 0.1185 |
| Laplacian 边缘 | 0.0519 | **0.5** | 0.0260 |
| topband | 0.00343 | **1.0** | 0.0034 |

> ⚠ **量级坑**：`topband` 只有 0.0034、`focal`(FFL) 只有 6.9e-5 —— **盲用 0.1/0.5 等于没加**。
> 任何新增辅助损失，**先量量级再定权重**（本项目已因此误判过一次）。

### 3.3 判别器（PatchGAN，**从零联合训练**）

| 项 | 设置 | 理由 |
|---|---|---|
| 结构 | `NLayerDiscriminator`：5 层全卷积，base 64，**输出 30×30**（感受野 ~70px），**2.76M 参数** | **只做局部**：它不管"这是什么字"，只判"这块黑白交界的边缘像不像毛笔" |
| 初始化 | **从零**（与 VAE 交替训练） | 判别器必须挑**本 VAE 特有的伪影**（边缘发虚/特定频率马赛克），**预训练判别器不认识** |
| 损失 | **Hinge**：`0.5·(relu(1−D_real) + relu(1+D_fake))` | 真图 >1、假图 <−1 后不再产生梯度 → 防"过度自信"与梯度消失 |
| 稳定性 | ① **`disc_start=5000` 热身**；② **自适应 λ**；③ 可选谱归一化；④ **LeCam 待命** | 见 §3.4 |

### 3.4 四条稳定性机制（每一条都对应一个已知的翻车方式）

| # | 机制 | 实现 | 防什么 |
|---|---|---|---|
| **① 热身** | `--disc-start 5000`：**前 5k 步 `λ_adv = 0`**，只跑 L1+LPIPS+KL+边缘+高频 | | 一上来就同训 → 判别器一眼看穿"你画的是垃圾" → 梯度全是噪声 → VAE 权重爆掉 |
| **② 自适应 λ** | `λ = clamp(‖∇L_rec‖ / (‖∇L_GAN‖ + 1e-4), 0, 1e4)`，梯度取在 **`vae.decoder.conv_out.weight`** | LDM/VQGAN 祖传 | 判别器梯度过大 → 为造"真实毛刺"而扭曲字形（**本项目当年正是因此弃用 GAN**） |
| **③ Hinge + LeCam** | Hinge（已用）；LeCam（默认 0，可开）：维护真实 logits 的 EMA，**假图分低于真实历史均值就罚** | | 训练中后期判别器突然变强 → 生成器崩溃（FLUX/MagViT-v2 家族的解法） |
| **④ 只用局部 PatchGAN** | 30×30 patch 判定 | | 整图判别器容易钻漏洞（角落排版等），且显存大；局部判别把 GAN 职责严格限定在**高频纹理** |

**冒烟已验证**：`λ` 在 **5.7 → 1.9 → 9.3** 之间自适应跳动；热身期 `D=0, λ=0` ✓。

### 3.5 训练配置

| 项 | 值 |
|---|---|
| 初始化 | **`pretrained_models/sd-vae-ft-ema`**（干净、**无 DINO 印记**） |
| 语料 | `/root/Workspace/xy/UNIFIED_RAW/meta/train_clean.csv`（393,486 张） |
| 架构 | **4ch / 8× 下采样**（保持不变 → 只换 loss，单变量） |
| 优化 | batch 8 × accum 4（等效 32）、lr 5e-5 cosine、20,000 步、**梯度检查点** |
| 硬件/速度 | 4090，18.7G/24.5G、**0.66 opt-step/s**（比 DINO 版快近一倍）→ **ETA ≈ 8.4h** |
| 命名 | `experiments/calli_vae_physics/`，日志 `logs/calli_vae_physics_*.log` |

> 为何不用现 Calli-VAE 初始化：它带着 DINO 印记；**主臂用干净底座**才能干净归因。
> 若想验证"能否洗掉印记"，可另起对照臂 `--vae-base experiments/vae_fix_topband/calli_vae_final`。

---

## 4. 验收标准（训完必须逐条过，见契约 §1/§4）

| # | 判据 | 阈值 |
|---|---|---|
| I1 | 往返保真 | `L1 ≤ 0.005`；PSNR ≥ 30 dB；**顶带高频份额 ≥ 0.85× GT**（目标 ≥ **0.90**） |
| I2 | 存盘分布 | `std ∈ [0.8,1.25]` **且 mean 经 shift 归零**；**sf / shift 必须由实测拟合** |
| I3 | 后验宽度 | 记录 `σ/|z|`；> 0.1 则固定用 `sample()` 并写入 PROVENANCE |
| I4 | 解码约定自检 | `decode(z_stored/sf)` 的 L1 比 `decode(z_stored)` 小一个数量级 |
| I5 | DiT 三处一致 | `shards.std` ↔ `config.vae_scaling_factor` ↔ `config.vae_path`（**编码与解码同一 VAE**） |
| **I6（本轮新增）** | **往返 LPIPS 不退化** | 防止 GAN 用"噪点换顶带份额"（顶带份额可被高频噪声刷高） |

**编码已自动化**：`tools/encode_aug_latents.py --scaling-factor auto`（默认）会
**从数据拟合 `sf = 1/std`、`shift = mean`**（跨全表等距抽样标定），并**自己写 PROVENANCE.json** → E5 不会再无声发生。

---

## 5. 训完之后的流程

```
新 VAE 验收(I1-I6)
   └─ 重编码 shards（auto sf/shift + PROVENANCE）
        └─ DiT **从头训**（换 VAE = 新数据变体，不可 warm-start；见契约 §5.3）
             └─ 同 step 对比 v72/v76（cfg1.0/γ0, N=187）
                  ├─ 判据: HF_dit/HF_rt 与 HF/GT（v72 为 0.622 / 利用率 0.794）
                  └─ ★ SDE 判决（A vs B）: 干净 VAE 下 SDE 增益是否回归
```

**第二阶段（DINO 归位 = 真正的 REPA-E）**：冻结 VAE **decoder**、解冻 **encoder（1e-6 级极小 LR）**
+ DiT 正常 LR，**DINO 对齐作用在 DiT trunk**（框架 `src/train/configs/v73_repae_calli_joint.json` 已备）。
⚠ 联合训练会让 latent 漂移 → **预编码 shards 失效、解码约定与尺度必须重验**（正是 E3/E5/E9 那一类坑）。

---

## 6. 结果（2026-10-11 13:24 训毕 20,000 步，I1–I6 验收）

**同一批 187 张字帖、同一脚本同口径实测**：

| 项 | **新 VAE（纯物理）** | baseline（旧 DINO 版） | SD-VAE（参照） | 判读 |
|---|---|---|---|---|
| L1 | **0.00756** | 0.01085 | — | −30% |
| **PSNR** | **34.04 dB** | 30.56 | 26.80 | **+3.48 dB**；**优于 SD** |
| SSIM | **0.9949** | 0.9852 | 0.9738 | **优于 SD** |
| **顶带高频（census 同口径）** | **0.901** | 0.783 | 0.816 | **+0.118；优于 SD** ✅ |
| **往返 LPIPS** | **0.0135** | 0.1281 | 0.0182 | **−89%；优于 SD** ✅✅ |
| 墨量 | 1.000 | 1.001 | — | 不漂 |
| 后验宽度 | 0.277 | 0.391 | 4e-5 | 仍偏宽 → 用 `sample()` ✓ |
| raw std / mean | 1.2052 / **−0.0254** | 1.1002 / −0.1079 | 5.87 / — | 均值更接近 0 |
| 拟合参数 | **sf=0.8298, shift=−0.0254** | sf=0.9090 | — | 交给 `encode --scaling-factor auto` |

**径向谱剖面（低→高 6 带，重建/GT 能量比）** —— 用来区分"真锐"与"用噪点换顶带份额"：

```
GT  : 0.9091  0.0471  0.0198  0.0147  0.0076  0.0015
新  : 1.006   0.981   0.970   0.871   0.842   0.800
对照: 1.010   0.994   0.879   0.748   0.697   0.667
```
- 各带全面更接近 1；**比值随频率平滑下降 → 无"顶带虚高"的噪点指纹** ✓
- 残余仍在（顶两带保留 84%/80% 能量），但已从"严重缺失"变为"轻微不足"

**肉眼（`recon_step_19000.png`，上排真迹/下排重建）**：碑刻剥蚀颗粒、篆书发丝细笔、枯笔飞白、
灰底阶调**全部还原**，无"塑料感"。→ **像素域配方已把纹理做出来**。

### 6.1 三条必须写明的保留

1. **这次胜利不能归给 GAN**：`λ ≈ 0.001`、`D loss 0.02~0.32` → **对抗基本未生效**，
   赢的是像素域项（L1+LPIPS+Laplacian+topband）。PatchGAN 的贡献**尚未被验证**。
2. **上一版 GAN 调参要点（下一版用）**：判别器太强（D loss 极低）→ 需**降 D 的学习率 / 启 LeCam / 给 λ 设下限**，
   否则自适应 λ 会把对抗压成噪声级（这正是 `λ≈0.001` 的含义）。
3. **I1 的 `L1 ≤ 0.005` 阈值是坏阈值**：**我们在用、并据此训过所有 DiT 的旧版基线 L1 = 0.0109，同样不满足**
   → 该阈值从设定起就无任何版本满足过。**记为"待重标定"**（改判据必须单独 commit 并说明依据，
   不得为了让结果好看而移动门槛）。其余 I1 条件（PSNR、顶带）与 I2/I3/I6 **全部通过**。

### 6.2 下一步

1. 用 `encode --scaling-factor auto` 重编码 shards（自动拟合 sf/shift + 自写 PROVENANCE）；
2. **DiT 从头训**（换 VAE = 新数据变体，不可 warm-start）；
3. 同 step 对比 v72/v76（`HF_dit/HF_rt`、`HF/GT`、SSIM/LPIPS/frag/墨量）；
4. **SDE 判决（A vs B）**：干净 VAE 下若基线 frag 仍低而 SDE 增益回归 → 支持"DINO 弄崎岖了流形"；
   若仍无效 → 支持"断线少、无事可修"；
5. （可选、低优先）`--w-gan 0` 消融，回答"PatchGAN 到底值不值"。

---

> 📐 SDE 增益翻转问题的完整技术简报见 [`docs/sde_analysis_brief.md`](docs/sde_analysis_brief.md)。

## 7. 不要做的事（血泪清单）

| 不要 | 因为 |
|---|---|
| 在 VAE 里挂 DINO/CLIP 等语义特征 | 它们对低级纹理不变；实测只有像素域高频项能提升 HF |
| 盲设辅助损失权重 | `topband` 量级 0.0034、`focal` 6.9e-5 —— 盲用 0.5 等于没加 |
| 对**已训练**模型事后缩放 latent | 实测（×1.15/×1.30）SSIM 0.6129→0.6098/0.6042，破坏已学到的解码流形 |
| 沿用别人的 `scaling_factor` | 那是**拟合值**。我们的 raw latent 本就 ≈N(0,1)，套用 SD 的 0.18215 会缩小 5.5×（E5） |
| 只做 scale 不做 shift | 半修复：存盘 mean 仍会偏移（FLUX 专门用 `shift_factor` 处理它） |
| 换 VAE 后 warm-start DiT | 采样起点仍是 N(0,1)，路径形状变了 → 属**不同的学习问题** |
| 复用 `results_dir` 重跑评测 | `run_in_mem_eval` 会因"(step,set) 已完成"跳过采样，并把旧图当新图拷贝（E6） |
| 信"训练内指标"而不对齐离线判据 | 曾被自己的 `band_fraction` 口径差 41% 误导，误判"高频已修好"（E7） |
| 就地给**缓存共享**的 VAE 对象打补丁 | 会静默污染（实测新旧路径差 227 灰阶，E9） |

---

## 8. 相关文件

| 文件 | 作用 |
|---|---|
| `docs/vae_latent_contract.md` | **latent 契约**：定义、5 条不变量、冻结/部署 runbook、失败案例 E1–E9 |
| `tools/vae/train_dino_calli_vae.py` | 训练器（现已支持 `--w-lpips/--w-edge/--w-gan/--disc-start/--w-lecam/--w-hf` 等；`w_struct=w_style=0` 时不加载 DINO） |
| `tools/vae/gan_losses.py` | PatchGAN 判别器 + Hinge + **自适应 λ** + LeCam + Laplacian 边缘损失 |
| `tools/vae/hf_losses.py` | 高频损失（`topband` 与离线普查**同口径** / `focal` / `fft` / `dog`） |
| `tools/encode_aug_latents.py` | 编码器（`--scaling-factor auto` 自动拟合 sf/shift + 自写 PROVENANCE） |
| `tools/vae/judge_vae.py` / `tools/hf_decompose.py` / `hf_roundtrip_ab.py` | 验收：后验宽度、往返高频、同批字 HF 分解 |
