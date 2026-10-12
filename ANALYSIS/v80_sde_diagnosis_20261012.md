# v80 / 新 VAE / SDE 诊断（4090，2026-10-12）

## 结论

截至 2026-10-12 00:06（Asia/Shanghai），4090 上 v80 训练约到 step 51,950，最新完整 checkpoint 和自动评测是 step 50,000。

**这份最新结果不是 SDE 结果，而是 ODE（deterministic Heun）结果。** 因而目前不能从 v80 最新曲线得出“SDE 在新 VAE 下仍然不 work”。更准确的说法是：v80 已经满足了 SDE 的训练前提（`use_c2ot=false`），但尚未对 v80 checkpoint 启用 SDE 推理。

证据：

- 远端运行时配置 `assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/resolved_config.json` 中 `sde_gamma: 0.0`。
- 源配置 `src/train/configs/v80_callivae_physics_noOT.json` 同样为 `sde_gamma: 0.0`。
- 启动日志只打印 `FlowMatching(... sampler=heun, ... use_ot=False)`；当 gamma 大于 0 时描述中会额外打印 `sde_gamma=...`。
- `src/eval/in_process_eval.py` 的自动评测通过 `flow_kwargs_from(args)` 直接复用训练参数，因此 step 5k–50k 的 `eval_stdskel_summary.csv` 全部走 gamma=0 分支。
- `src/loss/flow_matching.py` 只有在 `self.sde_gamma > 0` 时才进入带随机增量和 score 修正的 SDE 分支；gamma=0 会落入普通 Heun ODE 分支。
- 检查时没有独立的 v80 SDE 评测进程，也没有单独的 v80 SDE 结果目录/日志。

配置注释里的“本 run 使用修正后的 SDE 采样器”容易造成误解：它表示代码版本包含修正后的 SDE 实现，不表示本次自动评测实际打开了 SDE。

## 最新 v80 结果实际上说明了什么

50k、相同评测集、相同 ODE 口径下，v80（no OT）相比 v79（C2OT）整体更好：

| set | 指标 | v79 50k | v80 50k | v80 - v79 |
|---|---:|---:|---:|---:|
| eval200fix | SSIM | 0.5733 | 0.5835 | +0.0102 |
| eval200fix | LPIPS | 0.36379 | 0.35536 | -0.00843 |
| eval200fix | ink IoU | 0.26665 | 0.28397 | +0.01732 |
| eval200fix | frag ratio | 1.7958 | 1.7490 | -0.0468 |
| seen | SSIM | 0.5883 | 0.5920 | +0.0037 |
| seen | LPIPS | 0.34264 | 0.33958 | -0.00306 |
| seen | frag ratio | 3.6075 | 2.5768 | -1.0307 |

所以当前数据支持两点：

1. 去掉 C2OT 没有让新 VAE 的模型训练退化，反而在 50k 同步步数上明显改善了 eval200fix。
2. v80 已经建立了一个比 v79 更合适的 ODE baseline，可以在同一 checkpoint 上做真正的 corrected-SDE A/B；但该 A/B 还没有发生。

## 如果真正打开 SDE 后 SSIM 仍不涨，最可能的原因

以下是机制判断，不是当前 v80 数据已经证明的事实，按可能性排序：

### 1. SDE 的目标与单 GT 的 paired SSIM 不一致

ODE 更接近条件分布的确定性代表/条件均值；SDE 则从条件分布采样。对每个条件只拿一张固定 GT 做 SSIM 时，额外随机性即使提高分布覆盖度，也通常不会提高点对点相似度。SDE 是否“work”不能只用单 seed SSIM 定义，还应同时看 LPIPS、连通性/碎片率、多 seed 稳定性与分布指标。

### 2. 新 VAE 解决了全局 scale，但没有消除后验内随机性

物理 VAE 的 stored latent std 已校到约 1，排除了旧版 latent 全局尺度失配；但训练目标仍来自较宽的 posterior sample（既有诊断中 posterior 相对宽度约 0.277）。这使每张图对应的不是一个确定 latent 点，而是一团不可由字/书家条件完全预测的 latent 分布。SDE 再注入路径噪声时，更可能增加样本方差，而不是把样本推近那一张固定 GT。

### 3. SDE 的收益主要是修碎笔，v80 ODE 已没有足够大的修复空间

历史上 SDE 在 v68 上的正收益伴随 frag ratio 大幅下降；到了 v79/v80，ODE 本身的碎片率已经低很多。此时 SDE 继续平滑/连通笔画的收益变小，而高频损失仍存在，净 SSIM 很容易为零或略负。

### 4. SDE 无法修正条件路由错误

若主要误差是未见“书家 × 字”组合中的偏旁/字形路由错误，采样器只会沿着已经学到的错误条件速度场走。随机扩散能改变局部纹理和连通性，但不能补回模型没有学到的字义结构约束。

### 5. `gamma=0.5` 不一定适合新 VAE

全局方差为 1 并不等于每通道、每空间频段都被白化。新 VAE 的局部几何、后验协方差和频谱与 SD-VAE 不同；直接继承 gamma=0.5 的各向同性扩散强度可能过大。若做正式 A/B，应该把 gamma 当作新 VAE 下需要重新标定的推理超参，而不是把 0.5 当成固定真值。

## 最小的判定实验（本次未执行）

不需要重训，也不应改 v80 正在运行的训练任务。待需要验证时，对同一个 v80 checkpoint、同一批样本、同一初始噪声分别评测：

- gamma=0（ODE baseline）
- gamma=0.1 / 0.2 / 0.35 / 0.5（corrected SDE）
- 每个 gamma 至少 3 个 SDE 路径 seed

除 SSIM/LPIPS/ink/frag 外，同时记录 HF/GT、seen/unseen 分层和逐样本差值。这样才能区分“代码没启用”“gamma 过大”“SDE 只改善连通性”和“条件模型本身错误”四种情况。

## 当前判断置信度

- **高置信度**：4090 上 v80 的最新自动评测是 ODE，不是 SDE。
- **高置信度**：v80 no-OT 在 50k 的 ODE baseline 优于 v79 50k。
- **中等置信度**：若 corrected SDE 仍无 paired-SSIM 增益，主要原因会是后验随机性 + paired metric 偏好确定性输出 + 高频/连通性权衡，而不是 latent 全局 scale。
