# 57. 模型 vs ref 深度对照 + 数据脏检查（2026-09-14 二次复核）

> **动机**：前次复核把"eval 端 maybe_add_white 8ch 维度不匹配"列为真 bug（已修），
> 但误判了"aux_zero_white 未注册"与"gate×1.2 缺失"（前者已注册，后者是设计差异）。
> 本文是**二次、更彻底**的逐项对照：把 `ref/moyi/moyun/moyun_2.py` 与
> `src/model/dit.py` + `src/model/modules.py` 逐块比，并顺带核查数据侧是否还有脏。
>
> 相关：52（ref 主模型精读）、56（白底/极性/对齐复盘）。

---

## 0. 一句话结论

模型主干与 ref 高度一致（adaLN 维度、zero-init、FinalLayer、位置编码在 rope=0 时都已对齐）；
**真正的 gap 只有一处**：CFG 作用通道数 —— ref 只对图像前 3ch 做 CFG，
我们对全部 12ch（image+canny+skel）做 CFG，把**结构辅助通道也放大了**。
数据侧干净，唯一隐患是 canny PNG 仍是黑底（只重建了 latent）。

---

## 1. 模型设计逐项对照（比 52 号更细）

| # | 维度 | ref `moyun_2.py` | 我们 `modules.py`/`dit.py` | 定性 |
|---|---|---|---|---|
| 1 | Block adaLN 维度 | `Linear(h, 6h)` `moyun_2.py:375` | `Linear(h, 6h)` `modules.py:345` | ✅ 一致 |
| 2 | adaLN-zero 初始化 | 置零 `moyun_2.py:537-539` | 置零 `dit.py:811-812` | ✅ 一致 |
| 3 | FinalLayer 调制头 | `Linear(h, 2h)` `moyun_2.py:325` | `Linear(h, 2h)` `modules.py:367` | ✅ 一致 |
| 4 | FinalLayer 输出 | `Linear(h, p²C)` zero-init | `Linear(h, p²C)` zero-init | ✅ 一致 |
| 5 | 时间步嵌入 | sinusoid+2×Linear | sinusoid+2×Linear | ✅ 一致 |
| 6 | 位置编码（rope=0） | `x + pos_embed`(sincos) `moyun_2.py:585-586` | `x + pos_embed` `dit.py:918-919` | ✅ 一致 |
| 7 | **CFG 通道数** | `model_out[:, :3]` `moyun_2.py:648` | `model_out[:, :in_channels]`(=12) `dit.py:1088` | 🔴 **真 gap，见 §2** |
| 8 | gate ×1.2 缩放 | `shift/scale/gate *= 1.2` `moyun_2.py:404-409` | 无 | 🟡 设计差异（52 #5 可选消融） |
| 9 | 归一化 | LayerNorm | RMSNorm | 🟡 现代化（非 bug） |
| 10 | FFN | GELU(tanh) MLP | SwiGLU（等参数） | 🟡 现代化（非 bug） |
| 11 | LabelEmbedder | **3 表** callig/font/char→concat→Linear `moyun_2.py:222-225` | **2 表** callig（char 已按 no_char_cond 移除） | 🟡 缺 font/script 先验（52 #4 待做） |
| 12 | learn_sigma | True（输出 24ch） | False | 🟡 口径差异 |
| 13 | 扩散/预测 | DDPM eps | flow velocity | 🟡 口径差异 |

**要点回顾**：
- 之前记忆中"adaLN_modulation 应为 9*hidden"是**错误信息** —— ref 与我们都用 6*hidden；
  6 个调制量 = shift/scale/gate × (attn + mlp) 两组，正好是标准 DiT。
- 位置编码在 `rope=0`（本 run 配置）时**已正确退回 sincos 加到残差流**，与 ref 完全一致，
  不存在"关了 RoPE 就没有位置信息"的问题。

---

## 2. 唯一的真 gap：CFG 作用通道数 🔴

### 2.1 差异

```python
# ref  moyun_2.py:648 —— 只对前 3 通道 (RGB) 做 CFG，其余 passthrough
eps, rest = model_out[:, :3], model_out[:, 3:]

# 我们  dit.py:1088 —— 对前 in_channels(=12) 做 CFG，即全部 12ch 都被放大
eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
```

本 run 目标 `x = cat(image(4), canny(4), skel(4))`，`in_channels=12`，所以我们对
**canny、skel 两个结构辅助通道也做了 CFG 缩放**。

### 2.2 为什么这是错的

- CFG 的作用是「用无条件和有条件的差放大条件信号」，条件 = 书家风格（`y_callig`）。
- `image` 通道承载「风格×结构」，CFG 该作用。
- `canny`/`skel` 通道是「结构目标」，与书家风格**无关**；被 `cfg_scale` 放大后，
  结构通道的预测被无端推离真实结构。
- ref 用 `:3` 是「RGB 遗留写法」（52 §1.5 已注明）；对我们 4ch sd-vae latent，
  正确对应是 **`:4`（图像前 4 通道）**，而不是 `:in_channels`。

### 2.3 影响面（为什么当前还没暴露为「训练坏」）

- `forward_with_cfg` 只在**采样/推理**调用，训练用 `forward`（无 CFG），所以**训练权重不受影响**。
- 但本 run 开了 `eval_self_cond=true`：pass-1 预测的 skel 通道被回灌成 pass-2 的条件 g。
  若 skel 通道在 CFG 里被放大，会污染回灌骨架，**影响 self-cond 采样质量**。
- 结论：这是**推理端质量问题，不是训练 bug**，但值得修。

### 2.4 修复方案（一行）

```python
# dit.py forward_with_cfg 内，把 in_channels 改成图像 4 通道即可（或加开关）
eps, rest = model_out[:, :4], model_out[:, 4:]
```

> 建议做成配置开关（如 `cfg_image_channels`，默认 4），避免硬编码；同时同步
> eval 侧所有 `forward_with_cfg` 调用点的语义（gpu_ablate_eval.py / inference.py 采样器）。

---

## 3. 数据侧脏检查

| 项 | 结论 | 依据 |
|---|---|---|
| aux 12ch 拼接顺序 | ✅ `cat(image4, canny4, skel4)`，与 config `aux_latent_shards_dirs` 顺序一致 | `latent_dataset.py:304-305,331-338` |
| 反相（拓片类） | ✅ 1,681 张 UniCalli 反色图已原位反相归一 | `tools/fix_polarity.py`、55 号 §1.1 |
| 白底归零的一致性 | ✅ 三类 wz latent 都减了同一 4ch 白底；eval 加回（8ch bug 已修） | `rebuild_latents_wz.py:117`、`inference.py:248` |
| canny PNG 极性 | ⚠️ **canny PNG 仍是黑底**（只重建 latent 未改 PNG） | 56 §8 #3 |
| 数据规模 | fame 27,552 + tongji 2,092 + unicalli 25,248 = 54,892，45 书家 | `base_inventory.txt` |

### 3.1 唯一隐患：canny PNG 黑底

`tools/gen_base_images.py:59-60` 落盘的 `final_canny_base/*.png` 是 `cv2.Canny` 原生输出
（前景=255/背景=0，即黑底白线），与 skel 的白底黑线**极性相反**。`rebuild_latents_wz.py`
只重建了 latent（`invert=True` 先翻转再编码），**没改 PNG**。

**影响**：当前训练直接用 wz latent，不受影响。但任何「绕过 wz latent、直接从
`final_canny_base/*.png` 重新编码」的脚本都会重新踩 E1（黑底）。

**建议彻底修**：改 `gen_base_images.py` 的 canny 分支为 `255 - cv2.Canny(...)` 统一白底，
再整体重生成 PNG + 重编 latent。否则此坑会一直潜伏。

---

## 4. 待办清单（按优先级）

1. **修 CFG 通道数**（§2.4）：`forward_with_cfg` 只对前 4ch CFG，开关化 + 同步 eval 采样器。
2. **修 canny PNG 黑底**（§3.1）：改 `gen_base_images.py` 统一白底，重生成 PNG + latent。
3. **加 font/script label**（52 #4）：csv 有 `script_id`，加一张 Embedding 复刻 ref 字体先验。
4. gate ×1.2（52 #5）、DDPM-vs-flow、learn_sigma 三处口径差异：留着做单变量消融，不做基线改动。

---

## 5. 附：本会话已修正并同步远程

- `src/eval/inference.py:maybe_add_white` 支持多通道（8ch/12ch aux 白底按组重复），
  修复 eval 从 27500 步起的 `8 vs 4` FAILED。已 scp 远程，训练已 resume（40000 步起）。