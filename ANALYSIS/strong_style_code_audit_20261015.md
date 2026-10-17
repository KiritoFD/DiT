# 强风格后训练代码审查（2026-10-15）

实机验收（03:51）：朴素顺序反传的batch224已连续运行超过40步，辅助active=224，探针ratio=1.0000，首轮系数约0.0376；PyTorch显存高水位21.98GiB。A100从正确v84A@200k EMA重跑20k，单独运行，旧串行保持停止。显存为峰值而非每一时刻常驻量。

最终朴素infra：FM/REPA先反传释放saved activations，再构建完整batch辅助图并反传，最后统一clip和一次AdamW更新。梯度与合并loss一致，新增该等价性回归测试；无activation checkpoint、无重计算、无辅助microbatch。batch192实机已完成首步及后续步骤，日志active=192、ratio=1.0000、峰值reserved18.50GiB。随后按用户22GB目标将batch升至224重新启动20k，日志`logs/v86_A100_full_plain_b224_20261015.log`；旧串行不恢复。

后续重跑：用户要求朴素infra，已删除辅助分块与activation checkpoint。A100单独重跑、完整batch参与辅助项；辅助时间步独立均匀采样于0.05–0.30，正常FM时间采样保持不变。batch192朴素双前向首步OOM（约23GB以上，无有效步骤），改batch144实测；配置`src/train/configs/v86_A100_full_fixed.json`，日志`logs/v86_A100_full_plain_b144_20261015.log`。因此新的batch/辅助时间采样口径不同于旧8张实验，不能称为仅修正映射的严格消融。旧自动串行保持停止。

修正状态：已按用户要求停止串行与A60训练，保留全部checkpoint/log，不自动重启。Critic映射改为显式使用训练callig/font remap，legacy critic入口同时修正。4090真实训练CSV逐行检查，23槽匹配、零错误。辅助样本上限改为0（全部时间窗内样本），A/B/D每步使用完整有效子批，C每四步使用完整有效子批。真实ID乱序与超过8张batch覆盖测试通过；全批版本的4090显存/吞吐尚需启动前验收，不能沿用旧8张版本ETA。

独立Writer补评（同次原型库，N187）：v84A Top1=0.5294、margin=0.0494；Control20k=0.5294/0.0508；错误映射A30@20k=0.5561/0.0659。当前统一口径下A30提高5/187张，并非所有指标不变；但不构成无污染实验。当前基线与之前0.5241有一张差异，不能将不同评测输出混拼。数据远端`assets/strong_style_independent_audit_20261015.json`。

## 已确认的错误

`CriticLoss.build_id2slot`使用CSV书家首次出现顺序，`LatentDataset`的`y_callig_raw`则使用`callig_remap_json`固定映射。4090实际CSV与remap核对：颜真卿固定ID8、CSV顺序ID9；赵孟頫固定ID9、CSV顺序ID8。因此当前强风格辅助前向给这两位施加了相反书家的监督。A/B/C/D共用此映射，后续臂同样受影响。不能仅提高系数掩盖该错误。尚未改实现或停止远端训练。

## 强度与回传

A30的2000次探针全部有8个有效样本，比例30%；A60约1597次探针均达到60%，无欠强度、无NaN跳步记录。A60前100次/最近100次：Writer loss4.708→3.108，margin−3.354→−1.257，系数均值0.00691→0.00607。因此不支持“辅助项完全没有学到”的说法。

辅助项先执行可导前向，加入总loss，再统一backward、clip、AdamW step；冻结Critic没有包no_grad。model.eval关闭的是随机条件dropout，并非梯度。全局clip统一缩放合成梯度，不能单凭它推断Writer相对FM被单独削弱。现有日志缺少两项梯度夹角、裁剪前总范数和AdamW实际更新归因，60%范数不能等价于60%有效风格更新。

## 与原方案的实现差异和风险

- A采用同书体真实书家限定CE，并非v85E的完整26槽CE，不能称为原Writer CE的严格单项消融。
- A/B/D每步只抽8张辅助样本，并单独采噪、重新前向；不是原FM前向的全部窗口样本。范数虽足够，梯度估计方差可能较大。
- 辅助时间窗0.05–0.30，输入大部分来自真迹latent，监督的是一步去噪估计，不是完整采样结果。C两路共用真迹混合的z_t，低t时该输入已经带有原书家的笔法；这不是纯噪声采样的严格反事实，需要独立验证其迁移效果。
- B/C/D分位数来自Critic开发真迹库，不能当作独立验证集指标。
- 强度比是探针步的参数梯度范数比，不是每一步、每层或最终优化器更新的比例。

## 评估边界

此前实时报告仅有SSIM/LPIPS/InkSSIM等内容、形态指标。独立Writer、DINO/CLIP、BSD尚未完成，因此不能声称“其他指标也没有变化”。本次启动CPU独立Writer补评，不占训练GPU。

审计工具：`tools/analysis/audit_strong_style_status.py`。结论：先修复并验证ID映射，再判断真实风格收益；当前A结果不能作为干净的强风格实验结论。
