# S0 / S4 / S5：等曝光续训到85k

2026-10-29 18:17现场确认：S0已到50,400，S5已到50,200，均真实完成恢复后的更新。旧generation2/later队列继续暂停；新任务不恢复旧自动晋级。

## 固定方案

| 机器 | 串行队列 | 当前窗口速度 | 训练峰值allocated | 当前起ETA估算 |
|---|---|---:|---:|---|
| 4090 | S0 50k→85k → S4 50k→85k | S0约1.61step/s；S4按历史1.53估 | S0约20.02GiB | 两组合计约12.5h，加终点评测/写盘波动 |
| 48 | S5 50k→85k | 约1.68step/s | 约21.38GiB | 约5.8h，加终点评测/写盘波动 |

每模型新增35,000次optimizer更新、17,920,000样本曝光。有效batch512、microbatch512，compile开启，无梯度检查点。训练CSV SHA、seed及batch已跨机核对一致；STD、VAE、模型结构、FM及配对采样继承各自50k配置，REPA/Critic/额外风格Loss不加入。

完整恢复raw权重、Optimizer、EMA和RNG，不从EMA另起优化器。统一LR在50k重启：35k horizon、200步warmup、峰值3e-5、终值3e-6。首个恢复更新保留原LR6e-6，随后进入现有restart scheduler。中断恢复不重新warmup。

每5k保存：55/60/65/70/75/80/85k。终点保留原val100和统一187字×3 noise seeds；共享训练字面板单列诊断，不作为未见字泛化证据。当前没有承诺85k已经充分收敛。

## 路径与恢复

- 4090有效根：`assets/style_focus_85k_20261029/4090/`，其`runs/S0`和`runs/S4`继承各自本机50k权重。
- 48有效根：`assets/style_focus_85k_20261029/48_v3/`，其`runs/S5`继承本机S5@50k。
- `48/`、`48_v2/`是启动前interval校验失败的保留记录，没有optimizer更新，不是对照实验。修正为eval_every=ckpt_every=5000后，v3已推进。
- 旧50k checkpoint只通过本机软链接作为fork起点，不覆盖、不搬运。
- 后台入口：`python -m tools.style_matrix.runtime.serial --plan <root>/plan.json --execute`；4090另加`--resident-pid 2107710`。失败后检查日志再用相同命令恢复，不自动无限重试。
- 全局GPU锁、source/input封存、25GiB剩余空间门槛与24h安全上限均沿用现有serial。当前48剩余空间足够新增7个S5 checkpoint，不自动删旧权重。

## 评价与停止扩展边界

S0为内容基线，S4为局部融合对照，S5为风格候选。三组到85k后，比较内容、书家身份和分布；不将不同指标求总分。S5++需成熟风格优势和独立特征支持后再讨论。成熟生成器确定前不启动Decoder-only适配；Writer-conditioned Decoder、新Velocity分支和复杂辅助Loss均不加入本轮。

E-G的AUC/EER/mAP/Style-FID是同一特征空间的不同统计，不是四套独立评价器。E-L的分歧仍保留。19:22更新：官方权重HWD已完成30面板，但HWD特征GT跨字AUC仅51–54%，不足以独立确认书家身份；详见[HWD](../hwd/REPORT.md)。原CHAED/TabNet正式复现仍未完成，探索回归基线未通过校准，不作为正式审美分数。

## 并行补充：单seed的字符聚类配对区间

已对固定10位书家、98个共同有效query，按书家分层、按字符聚类bootstrap2000次。以下是**逐查询ROC统计的书家宏平均差值**；不是此前池化AUC/EER差值的区间。单seed探索、未校正多重比较。

| 对照 | Δ宏平均AUC pp [95%CI] | Δ宏平均EER pp [95%CI] | ΔmAP pp [95%CI] |
|---|---:|---:|---:|
| S4−S0 | +2.81 [−0.81,+6.26] | −2.18 [−5.23,+0.96] | +3.53 [−1.46,+8.61] |
| S5−S0 | +5.73 [+1.65,+9.84] | −4.50 [−7.94,−0.95] | +5.33 [+0.25,+10.36] |
| S5−S4 | +2.92 [−0.33,+6.22] | −2.32 [−5.20,+0.53] | +1.80 [−3.19,+6.73] |

这支持优先研究S5相对S0的风格信号，但S5相对S4仍不能确认稳定领先，且没有解决E-G以外的独立验证问题。完整结果见`paired_verification.json`，原始query和输入身份见`evidence/`。
