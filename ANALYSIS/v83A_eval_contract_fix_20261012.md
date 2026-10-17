# v83A 白化诊断与评测修复

## Bug 分析报告

### 一、错误现象

4090 的 v83A 在 55k 评测中大量欠墨，但这些输出不能直接用于判定零输入响应消除失败。训练带有 std-soft，旧评测底层不支持这种条件，另外实际采样使用了配置注释声称关闭的 SDE。

检查对象是 `assets/results/v83A_stdsoft_zerofix/20261012-152705-v83A-stdsoft-zerofix/checkpoints/0055000.pt`。用户要求停止训练后，已向核对过命令行的训练进程 PID 1481562 发送 SIGTERM；完整 checkpoint 保留，训练没有重启。

复现步骤：在旧远端环境用 run 的配置调用 `src.eval.inference.make_eval_cache`，`skel_root=None`，两个骨架 shard 路径均为空。单样本 CPU 检查返回 `skels_latent is None=True`、`skels.sum()=0`，函数签名没有 `std_soft_root`。输入信息完整，采用调用链诊断，不进行 Git 二分。

### 二、根因分析

训练侧 `src/train/train.py` 将 `batch['std_soft']` 送入 `model_kwargs['g']`，并因 `std_soft_cond=True` 开启 glyph encoder。v83A 的 `glyph_drop_prob=0`，所以缺少这个条件不是正常训练分布。

远端旧 `src/eval/inference.py` 的缓存只支持旧 PNG/latent 骨架。路径为空就返回 `skels_latent=None`；`sample_latents` 遇到 None 不传 `g`；模型前向的 `g is not None` 条件不成立，整个空间条件分支被跳过。这里的问题先发生在缓存构造，不是收到正确 g 后把它路由成了 cond。

训练进程于 15:27 启动，模块在训练入口顶层导入。远端 `in_mem_eval.py` 于 18:57 更新，而底层 `inference.py` 仍是 10 月 10 日的旧版。更新后的上层开始传 `std_soft_root`，旧底层签名却不接受它；训练仍成功输出旧口径评测，说明不能把磁盘更新当作运行中评测已更新。没有对活进程内存做侵入式检查；缺 g 的执行链依据旧底层复现、启动时间与成功评测日志作高置信判断。

另一个确定偏差是注释写 ODE γ=0，resolved config 和初始化日志实际是 γ=0.5。`in_mem_eval` 直接传 `flow_kwargs_from(args)`，所以旧结果用的是 corrected SDE，而训练为 C2OT 配对。该 score 转换依赖独立高斯配对，不能宣称在 C2OT 下保持同一边缘分布。

数学上，`G(g)=F(g)-F(0)` 只保证 `G(0)=0`，并不保证模型在没有 g 的情况下也生成正确字形。缺 g 时甚至不会计算这项。55k EMA 的前 16 个固定字形 CPU 探针测得 `F(0)` 平均 token 范数 0.6367，`G(g)` 为 1.4133，glyph scale 为 0.4216。这不支持把当前白化直接归为此前 v81 那种巨大零输入响应膨胀；但也不证明该结构在所有训练阶段都稳定。

旧 55k 的 187 张 PNG 中，生成软墨量均值 0.09506、GT 为 0.20323；总墨量比 0.4678，逐图墨量比均值 0.5268；33/187 张低于 GT 墨量的 10%。旧 SSIM=0.57712、LPIPS VGG=0.40714、Ink IoU=0.09368。

### 三、最小化Diff格式修复补丁

部署范围仅为评测链路、v83A 的 ODE 配置、回归测试与独立评测入口。没有更改训练模型或已有 checkpoint 权重。远端被覆盖文件已备份到 `assets/results/_diag_v83A_eval_backup_20261012`。

```diff
--- a/src/eval/inference.py
+++ b/src/eval/inference.py
@@ sample_latents
+    if model_uses_std_soft(model):
+        validate_std_soft_condition(skel, n)
--- a/src/train/configs/v83A_stdsoft_zerofix.json
+++ b/src/train/configs/v83A_stdsoft_zerofix.json
@@
+  "sde_gamma": 0.0,
```

完整实现见 `src/eval/inference.py` 和 `src/eval/in_mem_eval.py`。std-soft 必须为有限浮点 `[N,1,64,64]`、范围 `[0,1]`、逐张非空；旧缓存、缺 g、模型与 args 的 std-soft 标志不一致均报错。实际条件张量 SHA256、来源、sigma、sampler、gamma、CFG、seed、VAE 路径和缩放合同按 step/set 写入 `eval_protocol/`。

独立入口 `tools/eval_std_soft_checkpoint.py` 从 checkpoint args 重建并 strict=True 加载 EMA，默认 Heun 50、CFG 1、γ=0；拒绝 OT checkpoint 的 corrected SDE，要求实际本地 VAE 路径，开启 LPIPS 和软墨量指标，并记录 checkpoint/source/CSV 哈希。真实 glyph encoder 前向钩子验证非空条件确实被消费。输出目录必须是新的空目录，避免复用旧评测去重记录。

远端调用：

```bash
cd /root/Workspace/xy/DiT
/opt/conda/envs/cu121/bin/python -u tools/eval_std_soft_checkpoint.py \
  --ckpt assets/results/v83A_stdsoft_zerofix/20261012-152705-v83A-stdsoft-zerofix/checkpoints/0055000.pt \
  --out assets/results/v83A_55k_correct_eval_20261012 \
  --vae-path data/pretrained/pretrained_models/sd-vae-ft-ema
```

### 四、防回归测试用例

`tests/test_std_soft_glyph.py` 覆盖缺条件、旧 latent 形状、全空白、NaN、越界、整数 mask、分批 g 路由、旧缓存拒绝与 checkpoint γ覆盖。与 SDE/C2OT 相关现有单测一起运行，36 项通过。远端环境没有 pytest，因此额外用实际 55k EMA 做严格重建和全量条件预检：eval200fix 为 `[187,1,64,64]`，seen 为 `[20,1,64,64]`，覆盖、极性、范围、非空检查全部通过。

旧 `tests/test_c2ot_drop_interaction.py` 是启动时直接解析位置参数的实验脚本，无法按 pytest 收集；没有修改这个无关脚本。Python 编译检查及 `git diff --check` 通过。

实际 GPU 全量重评已完成，结果隔离在 `assets/results/v83A_55k_correct_eval_20261012`。实际协议确认 g 为 `[187,1,64,64]`、sigma=0.65、Heun 50、CFG 1、seed 0、γ=0，SD VAE 的 sf=0.18215、shift=0。真实 encoder 前向输入审计和完整指标分别见 `encoder_audit.json`、`metrics.json`；训练在评测结束后仍保持停止。

| 55k eval200fix 指标 | 旧评测 | 正确评测 |
|---|---:|---:|
| SSIM ↑ | 0.577117 | 0.577116 |
| LPIPS VGG ↓ | 0.407139 | 0.355603 |
| Ink IoU ↑ | 0.093678 | 0.276847 |
| Skel IoU ↑ | 0.007858 | 0.018382 |
| Fragmentation ↓ | 6.923367 | 2.545349 |
| 逐图软墨量比均值 | 0.5268 | 1.1238 |
| 近白率 ↓ | 33/187，17.65% | 1/187，0.53% |

旧软墨量和近白率由已保存 PNG 重算，新值由评测浮点输出计算，因此量化细节并非完全相同；差异幅度不影响白化判断。seen 20 张的正确评测为 SSIM=0.587988、LPIPS=0.331234、Ink IoU=0.383256、软墨量比=0.975893、近白率=0。

同一 checkpoint 的首张输出由底部小墨块恢复为完整字形，整体缺墨不再成立。全图 SSIM 几乎不变，而 LPIPS、墨迹重合和连通性明显改善。这支持先前评测合同失配的判断，而不是“v83A 的模型权重已经整体白化”。仍有一张近白图，也不能据本次重评证明所有字形与泛化问题已解决。

### 五、预防建议

新评测与旧历史曲线隔离，旧结果标为“缺 std-soft 的无效评测”，不作为 v83A 方法失败或优劣证据。训练是否白化应依据正确条件下的墨量、近白率、LPIPS 和结构指标，不只看全图 SSIM。

如果正确评测仍欠墨，再诊断输入 token-add 的幅度、时间依赖、空间共模和条件增强；不能跳过数据合同验证直接改注入层。此次验证同时修复了 g 缺失和 γ配置，改善量不等于单独 g 的因果增益，需要单变量比较才能拆分。
