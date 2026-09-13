# V3B+ HYBRID 混合初始点方案（标准字形 + 高斯噪声）

> 状态：**方案已定，代码已就位**。当前 V3B 训练(step≈19k, SSIM 0.46)先收敛，**收敛后再启用本方案重训/续训**。
> 初始化该方案改动：仅在 **采样起点(z 初值)** 做混合，不改变训练前向扩散、loss 或条件注入方式。

---

## 1. 动机

- 复现/经验给出两种初始点与 GT 的 latent 距离：
  - `D(std→GT) = 1.122`
  - `D(noise→GT) = 1.517`
  - 比值 ≈ **0.74** → 标准字形 latent 是一个比纯噪声**更接近 GT** 的起点。
- 但标准字形终归与 GT(带书家笔意)有结构差(IoU≈0.015), 纯用 std 会"锁死"在规范字形上、缺乏书家风格自由度;
  纯用噪声则起点偏离, 采样绕路。
- **结论：HYBRID** —— 采样初始点 `x_T` 取 `标准字形 latent` 与 `高斯噪声` 的线性混合,
  既用 std 提供结构先验、又保留噪声的随机探索给书家风格留余地。

## 2. 数学定义

设在最后一拍(完全噪声区)采样的初始 latent：

```
g ~  N(0, I)                     # 标准高斯噪声 (4,32,32)
s ~  VAE.encode(std_glyph)       # 标准字形 latent   (4,32,32)，已在 /0.18215 尺度同域
xT = alpha * g  +  (1 - alpha) * s
```

- **alpha ∈ [0,1]**：噪声占比。
  - `alpha=1` → 当前 V3B(纯噪声, 等价于现在行为)。
  - `alpha=0` → 纯标准字形初始。
  - `alpha≈0.5~0.7` → 推荐 HYBRID 起点：结构先验 + 风格随机性兼顾。
- `s` 与训练用同一个 `GlyphLatentLookup(script_id,char)` 缓存(`std_glyph_latent/{kai,li}/U+XXXXX.npy`, VAE 编码后 **(4,32,32), 与 xT 同一 0.18215 缩放域)**，保证**训练/推理一致**(用户硬性要求)。

## 3. 代码改动点

均只需给采样起点加一行混合, 不改扩散循环:

- `eval_auto.py` L242 (auto-eval / 历史取样主路径, **已实现**):
  - `z = torch.randn(b,4,32,32,device)`
  - → 当 `glyph_init_mix 在 (0,1)` 且 `gs` 存在时:
    ```
    z = alpha*noise + (1-alpha)*gs[i:j]
    ```
  - `eval_gen_in_memory(...)` 新增参数 `glyph_init_mix`(默认 0 = 全噪声, 保持与现有一致)。
- `train.py`（**已实现**）:
  - 新增 `--glyph-init-mix`(默认 0) 并透传给 `eval_gen_in_memory`。
- `eval_gen.py` (离线独立生成脚本, **暂不改**): 它是 legacy 3-cond 生成器, 未建 glyph-cond 模型
  (`forward_with_cfg` 不接受 `g`), 属次要离线路径。若将来需离线 HYBRID 生成, 需先给它加
  `use_glyph_cond` + 标准字形 latent 读取。V3B 训练期 auto-eval 走 `eval_auto.py` 已覆盖。
- 训练端 `train.py` 前向/loss：**无需改动**（初始点只影响采样, 训练一律从数据前向加噪）。

## 4. 训练/推理一致性保证

- 采样起点用与训练条件注入**同一个** `g` 缓存来源(同一 script×char→latent)。
- alpha 同一值贯穿一次采样的全部 batch/step；不同实验间以 `--glyph-init-mix` 区分。
- CFG 仍只 drop 书家条件, `g` 恒在 —— 与训练一致。

## 5. 待办(收敛后)

- [ ] 收敛后可再跑一个 `--glyph-init-mix 0.6` 的对照 ablate(alpha∈{0,0.4,0.6,0.8,1})。
- [ ] 观察 SSIM/结构清晰度是否随 alpha 降低而提前提升、书家风格是否保留(避免全 std 锁死)。

## 6. alpha 选择建议

| alpha | 含义 | 适用 |
|-------|------|------|
| 1.0   | 纯噪声 | 基准(现 V3B) |
| 0.7   | 弱 std 引导 | 保守延续, 几乎不变 |
| 0.6   | 推荐 HYBRID | 结构+风格平衡 |
| 0.4   | 强 std 引导 | 加速收敛、更贴规范字形 |
| 0.0   | 全 std | 探索上限, 可能损失风格 |
