# P0：先诊断断笔，不训练网络

所有入口默认不启动A/B/C训练，不修改原checkpoint或PNG。

## A：阈值与连通块（纯CPU、可复用已有图）

```bash
python tools/analysis/unet_connectivity/run.py thresholds --set raw20=assets/audits/unet_vs_dit_20261018/unet_raw20000/eval_samples_ctrl/step0020000/eval200fix --out assets/audits/connectivity/thresholds.json
```

扫描阈值0.3/0.4/0.5/0.6/0.7、最小面积0/2/4/8，8连通。
两边对称过滤，只写统计，不改图；同步输出Frag、组件数、墨量、IoU、缺失及错误粘连。
过滤噪点后Frag下降不代表生成已改善；若有效大笔画仍断开，应在较大面积门槛下仍看到额外组件。
没有把过滤后的数字替换成原始生成质量指标。

## B/C：同一checkpoint、Raw/EMA、Heun50/100、单尺度OOD干预

```bash
# 用已经修复EMA的同一个checkpoint；不要拿raw20k和EMA20.1k冒充同点对照。
python tools/analysis/unet_connectivity/run.py sample --checkpoint assets/results/Unet_ema_fixed_100k/20261018-020505-Unet-flow/checkpoints/0020100.pt --out assets/audits/connectivity/fixed20100

# legacy20k没有有效EMA，只允许显式raw：
python tools/analysis/unet_connectivity/run.py sample --checkpoint assets/results/Unet/20261017-233321-Unet-flow/checkpoints/0020000.pt --states raw --out assets/audits/connectivity/legacy20_raw
```

默认对raw/ema各运行baseline(Heun50)、Heun100、32²Skip×0.8、32²STD×0.8；未指定的尺度均1.0。
统一187字、固定初始noise(seed0)、clean STD、CFG1和VAE，所有GT只用于评分。
Heun100是两倍网络评估量，不视为同算力free lunch。
Skip/STD推理干预是OOD，只定位敏感性，不作为训练后提升证据。

生成前核验checkpoint SHA/size，严格加载；raw/ema及干预会写入不同EvalRequest身份，避免错用DONE。
每种情况输出原生SSIM/LPIPS-VGG/IoU/Frag；完成后用同一holdout CE补Writer，再做阈值扫描。
输出cases.json、writer.json/逐图记录、thresholds.json及各case的protocol/DONE/provenance。
没有自动宣判胜负；比较干预时还需排查错字、缺笔、粘连，不能只排序Frag。

每种情况还记录STD三尺度贡献比`RMS(projected STD)/RMS(pre-add feature)`：
只在真实eval条件的前4例、t=1纯噪声端测量，不误称为全轨迹STD强度。
训练不注册这些hooks，不引入热路径同步或CPU/GPU往返。

执行环境：在包含新src/tools的完整工作目录运行，需VAE、数据缓存、holdout CE及相应感知模型缓存。
这组GPU推理尚未执行；当前100k训练未停止、未替换源码。不要未经资源协调直接并跑。

后续训练配置与严格迁移：[A/B/C说明](../../../src/unet/configs/connectivity/README.md)。

## 60k串行执行（2026-10-18）

`serial.py --run`：等待指定checkpoint及其评测完成，核对PID/cwd后只停止该训练进程组。
依次执行当前基线完整续训至60k、同checkpoint raw重置optimizer/EMA的Control、组合结构版。
组合版启用Skip32门控、Decoder[3,3,2]、三尺度STD门控；不含全网加深与Writer-only调制。
Control与组合版同起点/随机状态，完整基线续训不是结构对照；LR horizon统一60k。
全部B512，每5k保存/统一评测；任何作业失败或缺少最终checkpoint立即中止队列。
`plan.json`记录源checkpoint哈希与实际配置，`status.json`记录队列状态，独立源码快照运行。
