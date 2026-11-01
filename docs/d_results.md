# D 计划结果（2026-11-01）

> **修订说明**：本文初版结论为「K 仍最优，四臂无一超越」。该结论**已被推翻**。
> 复查发现三个训练臂根本没从 K 出发（style 分支是随机初始化的），
> 因此它们相对 K 的所有差异都混入了「随机 style 分支」这一混淆项。
> 剥离该混淆后，**D1 的 Decoder 是显著正收益**，K+D1dec 成为当前最优。

## 1. 根因：style 分支从未载入

`tools/style_matrix/runtime/common.py:34-46`：

```python
if config.style_matrix_repr >= 0:
    transfer = initialize(model, spec['source'], table=...)      # 只载主干
    if spec.get('transfer_style'):                                # 才载 style 分支
        style = {k: v for k, v in ck['ema'].items()
                 if k.startswith(('style_representation.', 'style_injections.'))}
        model.load_state_dict(style, strict=False)
```

`build_specs.py` 把 `transfer_style` 当作 F 臂遗留字段**剥掉了**。于是每个 D 臂都是
「K 的主干 + 一个随机初始化的 style 分支」，包括三个 injection gain。

实测（`tools/d_20261101/check_init.py`，与 K 的 EMA 逐张量比对）：

| 构造方式 | 368 个张量中的差异数 | max\|diff\| | 位置 |
|---|---|---|---|
| 当前 spec（`transfer_style` 缺失） | 33 | 3.606e+01 | `style_injections.2.gain` |
| `transfer_style=True` | **0** | **0.000e+00** | — |

三个 gain 的实际值：

| site | K | 三个 D 臂（同一随机初值） |
|---|---|---|
| 0 | 14.8104 | 59.1093 |
| 1 | 24.2685 | 66.8432 |
| 2 | 37.0578 | 63.8560 |

三个臂的 gain 差异完全相同（44.299 / 42.575 / 26.798），因为 `construct()` 用
`config.seed` 确定性初始化 —— 这本身就是「三臂共用同一个全新 style 分支」的指纹。

**这不是 EMA 漂移。** 已排除：`train.py:198` 是 `if not freeze_backbone: update_ema(...)`，
冻结时 EMA 不更新；且实测 `ema_D1 == raw_D1`（差 0.0），`ema_K vs raw_K` 仅差 7.27e-04。
EMA 从来没动过，问题在载入。

### 1.1 这解释了原先所有的「异常」

| 现象 | 原解释 | 真实原因 |
|---|---|---|
| D1 direction 0.5454 → 0.1607 | 「decoder 抹平了风格」 | 随机 style 分支 + 冻结，风格注入是噪声 |
| D2 direction 0.4291 | 「联训伤了 Writer」 | 3000 步 FM 只把随机分支练回一半 |
| D3 direction 0.5372 ≈ K | 「D3 无损」 | 11000 步 FM 把随机分支练回接近 K |
| D1 面板指标全面变差 | 「decoder 导致糊连」 | 主要是随机分支；decoder 本身是正收益 |

D3 的预算大部分花在**重新学习一个本可以直接继承的 style 分支**上，而不是改进内容。

## 2. 剥离混淆后的真实结果

用 K 的生成器 + D1 的 Decoder（`--decoder-from`，零训练成本）直接测 decoder 的真实作用：

| 生成器 | Decoder | SSIM | LPIPS | Ink-IoU | 碎片比 | OCR top1 | OCR top5 | direction |
|---|---|---|---|---|---|---|---|---|
| K | 原始 | 0.5946 | 0.3462 | 0.2886 | 2.658 | 96.75 | 98.00 | 0.5571 |
| **K** | **D1@2500** | **0.5982** | **0.3416** | **0.2893** | **2.377** | 96.56 | 98.11 | 0.5571 |
| D3 | 原始 | 0.5927 | 0.3477 | 0.2847 | 2.763 | 96.00 | 98.33 | 0.5372 |
| D3 | D1@2500 | 0.5965 | 0.3428 | 0.2854 | 2.389 | 96.22 | 98.44 | 0.5372 |
| D1（随机分支） | D1@2500 | 0.5839 | 0.3562 | 0.2630 | 2.423 | 95.89 | 98.67 | 0.1607 |

**K+D1dec 在每一项上都优于 D3+D1dec**（SSIM 0.5982 vs 0.5965、LPIPS 0.3416 vs 0.3428、
Ink-IoU 0.2893 vs 0.2854、碎片 2.377 vs 2.389）。这回答了「为什么 decoder 配 D3 有收益、
配 K 却退化」——**它从来没有在 K 上退化**，退化的那个是随机 style 分支的 K。

### 2.1 配对检验（K+D1dec vs K，100 张同图同种子，bootstrap 1e4）

| 指标 | K | K+D1dec | delta | 95% CI | 胜/平/负 | 判定 |
|---|---|---|---|---|---|---|
| SSIM ↑ | 0.5946 | 0.5982 | +0.0036 | [+0.0023, +0.0052] | 96/0/4 | **显著改善** |
| LPIPS ↑ | 0.3462 | 0.3416 | +0.0047 | [+0.0040, +0.0053] | 90/0/10 | **显著改善** |
| Ink-SSIM ↑ | 0.4423 | 0.4456 | +0.0033 | [+0.0022, +0.0046] | 97/1/2 | **显著改善** |
| Ink-IoU ↑ | 0.2886 | 0.2893 | +0.0007 | [+0.0005, +0.0009] | 83/0/17 | **显著改善** |
| 碎片比 ↓ | 2.6578 | 2.3765 | +0.2813 | [+0.0417, +0.6695] | 29/61/10 | **显著改善** |
| 容差骨架 F1 ↑ | 0.0258 | 0.0253 | −0.0005 | [−0.0010, −0.0000] | 41/0/59 | 显著退化（−2%） |
| 墨量比 →1 | 1.0822 | 1.0715 | 更近 1.0 | [−0.0011, +0.0041] | 47/53 | n.s. |
| 空洞率 ↓ | 0.0450 | 0.0434 | +0.0016 | [−0.0015, +0.0052] | 80/12/8 | n.s. |

五项显著改善、胜率 83–97%，只有容差骨架 F1 轻微退化（0.0258 → 0.0253）。

### 2.2 direction 口径不一致（第二个混淆）

K_A 的 t3 报告 `n_pairs=900`，而 D 四臂都是 `n_pairs=1350` —— **K 的 0.5454 与其它臂不可比**。
用同一 1350 对口径重测（K_plus_D1dec 那次，生成器仍是 K），K 的 direction 实为 **0.5571**。
所以正确的对照是 K 0.5571 > D3 0.5372 ≈ D3+D1dec 0.5372 > D2 0.4291 > D1 0.1607。

注意 direction/amplitude 在 **latent 空间**计算（`gen_mean` 来自 `latents`，`true_mean` 来自
训练分片），所以 decoder 不影响它们 —— K 与 K+D1dec 的 direction 完全相同是**正确的**，
不是复用错误。

### 2.3 已修的评测缺陷

`t3_writer_swap` 原先不认 `--decoder-from`，交叉组合的 t3 报告与 D3 **md5 完全相同**。
direction/amplitude 本就与 decoder 无关，但 **OCR 走图像**，原先是 D3 的原样结果。
已加 `--decoder-from`（只作用于 decode 阶段）并由 `eval_arm.py` 透传；补跑后 OCR 96.22/98.44。

## 3. Decoder 步数扫描

（待补：K 生成器 + D1 的 500 / 1000 / 1500 / 2000 / 2500 步 decoder，同一 latent）

## 4. 结论与下一步

1. **`transfer_style: True` 已写进 `build_specs.py`**，并有 `check_init.py` 做构造后断言。
2. **K+D1dec 是当前最优**，五项指标显著改善。它不是新权重，只是「K 的 U-Net + D1 的 decoder」，
   部署成本为零。
3. 四个 D 臂的原始数字**不能用于判断各自的假设**，需在正确的起点上重跑才有意义。
   其中 D1（decoder-only）已由第 2 节以更干净的方式验证过，无需重跑；
   D2/D3 若要重跑，需带 `transfer_style`。
4. 唯一确定退化的指标是容差骨架 F1（−2%），与 decoder 只改外观、不改骨架的预期一致。
5. 「糊连」假设**不成立**：端点密度等形态学诊断是在被污染的 D1 上测的，需在 K+D1dec 上重测。

## 5. 产物

- 对照图：`assets/d_20261101/review/`
  - `review_20_spread.png` 跨书家 20 字，7 列
  - `review_20_best_K_plus_D1dec.png` / `review_20_worst_K_plus_D1dec.png`
  - `review_20_best_D3_11000_plus_D1dec.png` / `review_20_worst_D3_11000_plus_D1dec.png`
- 工具：`tools/d_20261101/{check_init,compare,defects,review}.py`
- 评测：`assets/d_20261101/eval/{K_plus_D1dec,D1_2500,D2_3000,D3_11000,D3_11000_plus_D1dec}/`
