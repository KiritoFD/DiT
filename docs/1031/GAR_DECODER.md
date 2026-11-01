# GAR 解码器后训练（K 冻结）

## 目标

在**完全不改动生成器 K** 的前提下，用一个对纹理/感知敏感的目标函数（GAR）
后训练 VAE 解码器，看能否在**不恶化风格（E-G）与 OCR**的情况下提升图像质量
（SSIM/LPIPS）。这是 D 系列（D1/D1b/D1c/D1d）的直接延续。

## 约束（用户明确要求）

- **不动 CFG**：K 部署口径就是 `cfg=1.0`，训练里生成器也只在 `cfg=1.0` 调用。
  注入 guidance 会让解码器看到推理时永远不存在的 latent 分布。
- **保持 K 原貌**：整条生成器冻结，逐张量审计必须 bit-identical。
- 只训练解码器（49.5 M 参数），生成器 114.6 M 参数全程冻结。

## 实现（复用既有链路，不新造轮子）

关键决定：**不写独立训练脚本**，而是把 GAR 作为第 10 个 arm 接进仓库已有的
审计过的 joint 目标与训练器。前一个独立脚本 `tools/d_20261101/gar_train.py`
在三处违反了仓库既有口径，已废弃：

| 违规项 | 独立脚本 | 仓库正式口径 |
|---|---|---|
| 书家/字体 id | CSV 原始 `calligrapher_id 0..9` / `script_id ∈ {0,3,4}` | 必须经 `callig_remap_p125.json` → 0..46 + null，`font_remap.json` → {0,1,2} |
| STD 条件 g | 硬二值 `std_bbox64` | σ=0.65 软字形（`std_soft_eval_sigma`，14999 张） |
| latent 来源 | 训练时现算 VAE encode | `shards_all` 缓存 latent（53522/53522 覆盖） |

`script_id=4` 越界正是那次 `CUDA error: device-side assert triggered` 的根因
（`backbone.script.weight` 只有 4 行 = 3 字体 + null），并非 cuDNN 故障。

改动只落在两个文件：

- `src/unet/joint/objective.py`：新增 `GAR` arm。`image_loss` 改走 `GARLoss`；
  生成器前向加 `torch.no_grad()`（**保留 autocast**，否则冻结前向会掉到 fp32，
  实测 1 步从 0.24 it/s 掉到 0.09 it/s）。
- `tools/d_20261101/make_gar.py`：从 D1b spec 派生 GAR spec。

### 目标函数

`GARLoss = MSE + [LPIPS(Alex) + 0.1×DINOv2-base feature L1] + 10×Gram(VGG16 relu1_2/2_2/3_3)`

- **没有 edge 项**：edge 奖励平坦低频结构，正是 D1 变粗/并笔漂移的机制。
- Gram 项按 10 倍权重压纹理统计量。
- DINOv2 走本地 hub 缓存（`source="local"`），不联网。

## 设定

| 项 | 值 |
|---|---|
| 起点 / 生成器 | K = `assets/abc_20261031/execution/runs/A/checkpoints/0043429.pt` (step 43429) |
| 冻结 | `freeze_backbone=true`（368 张量），`skip_fm=true` |
| 训练参数 | 仅解码器 49.5 M |
| decoder_lr | **5e-5**（D1b/D1c 用的 5e-6 在 1000 步只动 ~1e-3，等于没动） |
| 每步采样 | `image_batch=64`，`image_micro=8` |
| 曝光 | 672 步 = **2 epoch**（21460 unique 作品，1 epoch = ceil(21460/64) = 336 步） |
| t 区间 | 沿用 joint 分支的 `.2..5` 单步 Euler |
| 吞吐 | 0.287 it/s（3.48 s/step）实测 |
| 时长 | ~39 min |

## 结果

### 冻结审计（每个 checkpoint 都跑）

`tools/d_20261101/verify_d1b.py` 在 step 100 上：

```
delta  vs K.ema: 368 tensors, max|diff|=0.000e+00  -> IDENTICAL
ema    vs K.ema: 368 tensors, max|diff|=0.000e+00  -> IDENTICAL
decoder: 138 tensors, total norm=2434.8243
OK: generator is bit-identical to K; only the decoder changed
```

**K 原貌成立**：368/368 张量最大差 0.0，解码器确实在动。

### 训练损失

| step | mse | lpips | DINO | gram |
|---|---|---|---|---|
| 1 | 0.00752 | 0.0318 | 0.494 | 0.001212 |
| 100 | 0.00827 | 0.0368 | 0.288 | 0.000872 |
| 270 | 0.00826 | 0.0313 | **0.244** | **0.000694** |

DINO −51%、Gram −43%，而 pixel MSE 基本持平——正是 GAR 的预期特征：
纹理/感知改善，而不是把像素拟合得更狠（后者才会带来 OCR 风险）。

### 面板指标（val100，`--use-ckpt-decoder`）

| tag | step | SSIM↑ | LPIPS↓ | Ink-IoU↑ | frag↓ | OCR top1 | writer dir | w50 |
|---|---|---|---|---|---|---|---|---|
| D1b_2500 | 2500 | 0.594575 | 0.346214 | 0.288571 | 2.658 | 77.78 | — | 7.936 |
| GAR_s000100 | 100 | 0.595157 | 0.345525 | 0.288650 | 2.655 | 77.78 | 0.5571 | 7.933 |
| GAR_s000336 | 336 | 0.596067 | 0.344474 | 0.288794 | 2.590 | 77.78 | 0.5571 | 7.943 |
| GAR_s000672 | 672 | **0.596863** | **0.343268** | **0.288968** | 2.666 | 77.78 | 0.5571 | 7.959 |

相对 D1b（生成器与 K 逐位相同的干净对照）：

| Δ | SSIM | LPIPS | Ink-IoU | OCR | frag | w50 |
|---|---|---|---|---|---|---|
| @100 | +0.000583 | −0.000689 | +0.000080 | 0.00 | −0.0033 | −0.0032 |
| @336 | +0.001492 | −0.001740 | +0.000223 | 0.00 | −0.0674 | +0.0064 |
| @672 | +0.002288 | −0.002946 | +0.000398 | 0.00 | +0.0082 | +0.0224 |

### 验收判定（对照预先设定的门）

| 门 | 要求 | 实测 | 判定 |
|---|---|---|---|
| LPIPS | 改善 | −0.002946（单调） | ✅ |
| SSIM | 不退化 | +0.002288（单调） | ✅ |
| Ink-IoU | 跌幅 ≤0.003 | **+0.000398**（反而升） | ✅ |
| OCR top-1 | 不显著下降 | 77.78 → 77.78（n=90） | ✅ |
| 笔画宽度膨胀 | 无系统性膨胀 | w50 +0.0224（**+0.28%**） | ✅ |
| writer 潜向 | 与 K 相同 | dir 0.5571 全程不变 | ✅ |

**六道门全过**，且 SSIM/LPIPS 在 100→336→672 上**单调**改善 —— 不是单点噪声。

三条关键一致性证据：

1. **writer direction 恒为 0.5571**，与 `K_plus_D1dec` 相同 → 生成器确实没动，
   解码器改动没有泄漏回 latent 空间。
2. **两轴共同改善**：SSIM 与 LPIPS 同向变好，不是"骗一个指标"（例如锐化会
   同时抬高 SSIM 与 LPIPS）。
3. **宽度几乎不动**（+0.28%），说明去掉了 edge 项之后，D1 那套"变粗/并笔"
   漂移没有复现。

### 效应量的诚实边界

改善幅度**很小**：SSIM 相对 +0.38%、LPIPS 相对 −0.85%。这是解码器侧后训练的
物理上限——latent 由冻结的 K 决定，解码器只能重新分配已有的高频信息。
要拿到量级更大的提升，必须动生成器（那是另一条线，且受"不动 CFG"约束）。

## 后续：GARU（解冻主模型）

### 为什么要做

D 计划其实**已经跑过联训**：D2 = `joint.arm=J3`（U-Net + decoder，FM + 弱图像监督），
D3 = 纯 FM。当时的结论是「联训伤了 Writer」（D2 direction 0.4291）。**该结论已作废**
（`docs/d_results.md`）：`build_specs.py` 把 `transfer_style` 当 F 臂遗留字段剥掉了，
于是 D1/D2/D3 全是「K 的主干 + 一个**随机初始化**的 style 分支」——
`check_init.py` 实测 **33/368 张量不同，max|diff|=36.06**，落在
`style_injections.2.gain`：

| site | K | D1/D2/D3（同一随机初值） |
|---|---|---|
| 0 | 14.8104 | 59.1093 |
| 1 | 24.2685 | 66.8432 |
| 2 | 37.0578 | 63.8560 |

三臂的 gain 差异**完全相同**（`construct()` 用 `config.seed` 确定性初始化），这是
「三臂共用同一个全新 style 分支」的指纹。所以 D2 的 0.4291 不是「联训伤风格」，
而是 **3000 步 FM 只把一个随机分支练回了一半**。**联训假设至今没有被干净测过。**

### GARU 的设计

`joint.arm=GARU`，与 GAR 同一个目标函数，只改三处：

| 项 | GAR（decoder-only）| GARU（解冻）|
|---|---|---|
| `freeze_backbone` | True | **False** |
| `skip_fm` | True | **False**（`common.step(fm=True)` 才建 velocity-MSE 图）|
| `joint.arm` | GAR | **GARU**（`detach_image=False`，保留生成器图）|

并**显式补回**三个 D 臂缺失的字段：

- `transfer_style: True` —— 这是 D1/D2/D3 出错的根因，不补就是重犯同一个错
- `calibrate_gains: False` —— `train.py` 的 `condition_report(calibrate=True)` 会把
  K 的 injection gain（14.81/24.27/37.06）重标成 6.01/19.87/22.52，等于偷偷扰动
  「冻结」的生成器
- `lr_backbone = 5e-6` —— 沿用 D2 的小主干 LR，保持可比性；style 组同 D2

`update_ema` 会运行（`freeze_backbone=False`），所以 EMA 才是部署权重。

### 验收口径变了（重要）

GAR 那条线的全部可信度建立在「**368/368 与 K 逐位相同**」上。**GARU 按定义不成立** ——
生成器就是该动。因此：

- **不做** frozen-K 审计（做了必然失败，且失败不是 bug）
- 验收改为 **与 K 的配对比较**（同图同种子 bootstrap），门槛是
  「真的超过 K」而不是「与 K 相同」
- `direction` 此时**是**有效读数：它在 latent 空间，生成器一动它就该动，
  正是判断「联训是否伤风格」的灵敏指标

### 已具备的两道门（`tools/d_20261101/run_garu.sh`）

1. **CPU 构造预检** `check_spec_init.py`：实测 GARU
   `368 tensors, max|diff|=0.000e+00`（起点确实是 K）、
   `trainable tensors: 362/365`（生成器确实解冻）。
   **这一条就是能提前抓住 D2 那个 bug 的检查。**
2. **梯度路由探针** `probe_grad_routing.py`：验证只有 GARU 产生生成器梯度，
   GAR/G2J4 必须为 0。若 GARU 的生成器梯度是 0，它就悄悄退化成 GAR，
   整个实验不可解释。CPU 侧已验证**图策略**正确（GAR/G2J4 阻断、GARU 保留）；
   GPU 侧的实际梯度数值需在 GARL 让出显卡后补跑。

### 尚未执行

GARU **还没有训练**。设计、代码、spec、两道门、启动脚本都已就位，
`run_garu.sh` 会先等 GARL 结束再自动接管。预算：2500 步 ≈ 2.5 h（`lr_decay_steps=2500`）。

## 边界

- 2 epoch 是这条线的首次曝光；更长的曝光需另行指令。
- OCR 在 n=90 上为 77.78 且**完全没变**，因此该面板对 OCR 的分辨率不足以排除
  小于一档的变化；不能声称"OCR 改善"。
- E-G 是 latent 空间的量（生成器决定），GAR 只动解码器，因此 E-G 的预期变化
  只有 OCR 那条路径；这与 D1c 的结论一致。
- 三个 checkpoint 的 frozen-K 审计都在**独立进程**里跑过，非训练期自证。

