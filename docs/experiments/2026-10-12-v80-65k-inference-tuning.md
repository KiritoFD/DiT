# v80 / 65k：多指标推理调参

结论：**当前有可复核的 LPIPS / 墨迹 IoU 收益，但不是 SSIM 全面上涨，也不是 SDE free lunch。**
推荐 `Heun 50 steps / shift 1 / CFG 1.5 / gamma 0`；同原基线 100 NFE，不训练、不加步数。
若偏重感知与墨迹重叠，可用 CFG 2.0，但它的 SSIM 代价更大，较 CFG 1.5 的额外 LPIPS 改善仍有不确定性。

## 范围与复现

- 固定 4090 上的完整 `0065000.pt`，仅本地 RTX 4070 Laptop 推理；不训练、不改远端运行中的任务。
- EMA strict 加载；SHA256 `a498386dfc7a3473678a85ee269b69056045a7ca2edeee9b31e0f4bc3f456492`，已与远端核对。
- VAE 权重 SHA256 `42a903282343cd80e14ea7135ffd2d9eeb1c476e1bbbaf871e5cca2109199c2f`；评测 CSV SHA256 `f2205eb2fbd34b03915d4dd26b379a107183606683cf3389b5264bcdd6d45a27`，均与远端一致。
- 187 张 `exp-std/csv/eval200_fixed.csv`，按书家×书体分组、固定 RNG 20261012 轮转取 80 张调参，剩余 107 张留作验证。分割只看元数据，不看分数。
- 同一 seed 在全 CSV 上生成相同初始 latent，再切子集；同时固定 SDE Brownian RNG、批大小 8。不同 seed 同时更换初始噪声与 Brownian 路径。
- SSIM：原主评测高斯窗 11 / sigma 1.5；LPIPS-alex 和 LPIPS-vgg 同时记录，不跨 backbone 比绝对值。
- 额外记录 ink-SSIM / ink-IoU、frag、`|frag−1|`、hole、墨迹面积比；避免仅凭白底 SSIM 或降低连通块数选择过度粘连的输出。
- 固定 Heun 50 步 / 100 NFE；CFG 前向包含条件/无条件两路。没有增加步数、best-of-N 或按 GT 选 seed。

## 先修解码口径，再谈推理收益

远端 shard `exp-std/data/shards_img_aug_calli_physics/PROVENANCE.json` 明确记录：

```
z_stored = (posterior.sample() - SHIFT) * SF
decode(z_stored / SF + SHIFT)
SF    = 0.8439769211405432
SHIFT = -0.03341800242750449
```

但原训练 CLI 没有注册 `vae_shift_factor`，JSON 里的字段未被解析进 checkpoint。
因此原训练内评测和原离线评测都没有加回它。修复本地 train/eval CLI 注册和 batch_eval 解码；
对已有权重必须显式按 provenance 补回 shift，而不能臆测 checkpoint 缺项就是 shard shift=0。
没有改远端训练、shards、权重或归一化参数。
v80 使用预编码 shard 训练，其 x0 已按上述合同存盘；这次缺字段首先是解码/元数据问题，
不能据此声称 velocity 训练的 x0 使用了错误 shift，也不需要为解码修复重新训练。

另修 `_ShiftedVAE.eval()` 的返回值：通过 `__getattr__` 返回 base.eval() 会在链式赋值中丢失包装。
最初临时扫描的 baseline 与 historical_decode 完全相同，因此触发检查并中止；
`assets/results/v80_inference_tuning_65k/coarse/` 是该无效临时扫描，**不用于选参或结论**。
正式扫描单独保存在 `coarse_corrected/`。

80 张调参集上的解码审计（同权重 / 初始噪声 / CFG=1）：

| 解码 | SSIM ↑ | LPIPS-alex ↓ | LPIPS-vgg ↓ | ink-IoU ↑ |
|---|---:|---:|---:|---:|
| 历史：shift=0 | 0.594642 | 0.405901 | 0.344343 | 0.313506 |
| 正确：按 shard 补 shift | 0.594619 | 0.405969 | 0.344377 | 0.313753 |

差异很小；这是合同修复，**不是 free lunch，也不能解释 SDE 主体收益不足**。

## 时间窗 SDE 的实现

对线性独立高斯插值，以 t:1→0 积分：

```
D(t) = gamma² * t * (1-t) * w(t)²
noise_std = sqrt(2 * D(t_i) * h)
b(x,t) = v + gamma² * (1-t) * w(t)² * [x + (1-t)*v]
```

`w(t)` 是仅依赖时间的确定性权重，漂移和噪声同步改变；不是只截断噪声。
默认 w=1 保留原 corrected bridge 行为。中段窗口 `[0.2,0.8]` 两端各 0.1 线性 ramp，
低噪声尾段退回 ODE，减少 gamma 对尾段速度误差的放大。
这仍是左端点扩散系数、共享噪声的随机 predictor/corrector；不声称时间变扩散下二阶强收敛。
同边缘论证只适用于精确场、独立 Gaussian pairing 的目标密度；CFG≠1 不能直接套用原条件密度证明。

实现与测试：

- `src/loss/flow_matching.py`：`sde_tmin / sde_tmax / sde_ramp`，默认不改变行为。
- `tests/test_sde_time_window.py`：窗口边界、工厂透传、gamma=0、Gaussian oracle 边缘、CLI shift 与包装器回归。
- `tools/analysis/tune_current_checkpoint.py`：单进程多配方扫描，逐样本 JSON、PNG、manifest、汇总 CSV。
- `tools/analysis/summarize_inference_tuning.py`：paired delta；先对每张图平均 seed，再 bootstrap 图像，不把多 seed 当独立 GT。

## 结果

### 调参集：80 张 / seed 0

| 配方 | SSIM ↑ | LPIPS-alex ↓ | LPIPS-vgg ↓ | ink-IoU ↑ | frag 偏差 ↓ |
|---|---:|---:|---:|---:|---:|
| ODE CFG 1.0 | 0.594619 | 0.405969 | 0.344377 | 0.313753 | 1.884537 |
| ODE CFG 1.3 | 0.596191 | 0.401212 | 0.342333 | 0.322508 | 1.775658 |
| ODE CFG 1.5 | 0.596272 | 0.398893 | 0.340499 | 0.326395 | 1.682292 |
| ODE CFG 2.0 | 0.594981 | 0.398666 | 0.339570 | 0.330479 | 1.595647 |
| ODE CFG 2.5 | 0.594313 | 0.398942 | 0.339496 | 0.335334 | 1.553509 |
| CFG 1.0 + bridge γ0.5 | 0.595104 | 0.406810 | 0.344640 | 0.313935 | 1.856518 |
| CFG 1.0 + mid γ0.7 | 0.594467 | 0.406362 | 0.344814 | 0.313034 | 1.915935 |
| CFG 1.5 + mid γ0.7 | 0.595617 | 0.399752 | 0.340836 | 0.325565 | 1.661293 |
| CFG 1.5 + bridge γ0.5 | 0.596371 | 0.398711 | 0.340056 | 0.327099 | 1.599789 |

`frag 偏差 = mean(|frag_ratio−1|)`，不是 `|mean(frag_ratio)−1|`。
降低 CFG 到 0.7/0.85 整体变差；time shift 0.7/1.4 未见实质收益。
提高 CFG 是主要候选方向；SDE 在 CFG=1.0 的 LPIPS 略变差，CFG=1.5 的全程 SDE 有微小同向增量，需验证。

据此在查看预留集结果前冻结四个配方：baseline / cfg150 / cfg200 / cfg150_bridge050，
对 107 张预留集使用 seed 0/1/2 复核，不再根据验证集追加调参。

### 预留集：107 张 / 三个 seed

以下为逐图、逐 seed 配对后的三个 seed 均值，**不是从三 seed 中选最好的一张**。

| 配方 | SSIM ↑ | LPIPS-alex ↓ | LPIPS-vgg ↓ | ink-SSIM ↑ | ink-IoU ↑ | frag 偏差 ↓ |
|---|---:|---:|---:|---:|---:|---:|
| ODE CFG 1.0 | 0.594792 | 0.409475 | 0.344714 | 0.441834 | 0.301151 | 1.070593 |
| ODE CFG 1.5 | 0.593964 | 0.405578 | 0.342259 | 0.444276 | 0.311771 | 1.008986 |
| ODE CFG 2.0 | 0.592808 | 0.404332 | 0.341609 | 0.444191 | 0.318199 | 1.002219 |
| CFG 1.5 + bridge γ0.5 | 0.593942 | 0.405573 | 0.342140 | 0.443521 | 0.311732 | 1.037259 |

**CFG 1.5 vs CFG 1.0（主要收益）**：

| 指标 | 配对均值变化 | 图像 bootstrap 95% CI |
|---|---:|---:|
| SSIM | −0.000828 | [−0.002703, +0.001070] |
| LPIPS-alex | −0.003896 | [−0.005912, −0.001844] |
| LPIPS-vgg | −0.002455 | [−0.003839, −0.001109] |
| ink-IoU | +0.010620 | [+0.007801, +0.013650] |
| frag 偏差 | −0.061607 | [−0.166049, +0.044626] |

三个 seed 的 LPIPS-alex 变化分别为 −0.003351 / −0.004882 / −0.003456，
LPIPS-vgg 变化分别为 −0.002595 / −0.002427 / −0.002342；感知收益方向一致。
SSIM 三个 seed 都下降，因此不能把“CI 跨零”偷换成“已证明 SSIM 不下降”。
断笔偏差三个 seed 均值都改善，但图像 CI 跨零，不把它当强统计证据。

**SDE vs 相同 CFG=1.5 的 ODE（额外增量）**：

| 指标 | 配对均值变化 | 图像 bootstrap 95% CI |
|---|---:|---:|
| SSIM | −0.000022 | [−0.000631, +0.000582] |
| LPIPS-alex | −0.000005 | [−0.000774, +0.000741] |
| LPIPS-vgg | −0.000119 | [−0.000668, +0.000416] |
| ink-IoU | −0.000039 | [−0.000947, +0.000936] |
| frag 偏差 | +0.028273 | [−0.029231, +0.089866] |

调参集上 SDE 的微小“全指标同向”没有在预留集稳定复现。SDE 的额外 LPIPS-alex 改善仅约 5e−6，
SSIM / 两种 LPIPS / ink-IoU 的 CI 均跨零；frag 偏差在三个 seed 下分别变差
+0.042651 / +0.025320 / +0.016846。**本轮不推荐开启 SDE 作为涨点配方**。

统计口径：5000 次固定 RNG bootstrap，先对每张图平均三个 seed 的配对变化，再重采样图像。
因此 CI 条件于这三个 seed，不是对所有 seed 的无条件保证，也不做多重检验下的广泛发现宣称。
这是本 checkpoint、本评测分布的推理调参结果；跨 step / checkpoint 的稳定性尚未验证。

## 实现与数学上的解释边界

1. **当前最可补的是条件引导，不是简单“多加一点噪声”。**提高 CFG 的 LPIPS、墨迹重叠改善在三个 seed 下同向；
   样图上能看到部分结构与长笔画连接更完整。这支持加强条件方向的工程选择，不能单凭它证明训练根因。
2. **正确 SDE 不等于更低重建误差。**精确场的同边缘性质不会让每张生成图自动更贴近配对 GT。
   同一点上的速度误差 e 在 corrected drift 中变为
   `[1 + gamma²*w(t)²*(1-t)²] * e`；没有新增语义信息，还可能放大低噪声端的系统误差。
3. **数学可行的窗口也不保证实际更好。**mid-window 在 oracle 上保边缘，但本模型调参集未赢过同 CFG 的 ODE；
   在它上面强行讲“尾段 ODE 必然 free lunch”不符合数据。
4. **CFG 的改善也有代价。**预留集墨迹面积比从 1.13623 升到 1.17985（CFG 1.5）/ 1.20585（CFG 2.0），
   GT=1；墨迹重叠更好同时也更容易偏粗，SSIM 随之小降。因此优先较温和的 CFG 1.5，而非盲目拉到 2.5。
5. 本轮只排除了已扫描配方在这份权重上的稳定额外收益，不证明所有 SDE 变体永远无效。
   不能把 v68 历史采样器的涨 SSIM 直接推广到这套新 VAE / corrected SDE / 多指标目标。

## 推荐配方与复现命令

`ANALYSIS/v80_65k_inference_recommended.json` 是**推理侧**配方，不覆盖运行中的训练 config。
已有 v80 checkpoint 缺失 shift 字段，使用 batch_eval 时须显式传 provenance 中的 shift。

```powershell
python tools/analysis/tune_current_checkpoint.py `
  --ckpt assets/results/v80_callivae_physics_noOT/20261011-195827-v80-callivae-physics-noOT/checkpoints/0065000.pt `
  --recipes ANALYSIS/v80_65k_tuning_confirm_recipes.json `
  --out assets/results/v80_inference_tuning_65k/confirm_reproduce `
  --subset holdout --seeds 0 1 2 --vae-shift -0.03341800242750449
```

完整本地运行数据：`assets/results/v80_inference_tuning_65k/{coarse_corrected,refine,confirm}/`，
包括逐样本 JSON、PNG、配方、seed、batch、checkpoint / VAE / CSV / 源码哈希。
精简可版本管理证据位于 `docs/experiments/v80_65k_inference_tuning/`：manifest、汇总、1284 行逐图配对 CSV、
两个参考基线的 CI。**无需 GPU / checkpoint 即可重算证据中的统计量**：

```powershell
python tools/analysis/summarize_inference_tuning.py docs/experiments/v80_65k_inference_tuning --subset holdout
python tools/analysis/summarize_inference_tuning.py docs/experiments/v80_65k_inference_tuning --subset holdout --reference cfg150
```

相关采样器与接口回归：`python -m pytest tests/test_sde_time_window.py tests/test_sit_sde_sampler.py -q`，15 项通过。
SSH 流程用于只读核对与权重拉取；没有远端评测、重启训练、替换远端文件或改动其它用户进程。
