# v92A：32×256 纯深度基线

状态（2026-10-18 01:26 CST）：用户授权停止4090的v90B，已启动v92A；48G U-Net不动。
v90B最后日志159800，最近完整checkpoint150000；原始产物保留。
01:31后按用户新指令延长至200k；配置已更新，当前100k进程完成后自动完整状态续训，无需人工再次启动。

## 配置来源与单变量边界

`baseline_20x256.json` 是本轮 SSH 读取的原始配置快照，来源：
`4090:/root/Workspace/xy/DiT/scratch/v90_20l_gpu_20261017_01/src/train/configs/v90_20l/w256_strong_200k.json`。
原始 `_comment` 误写384/8heads，保留以便对账；实际远端注册函数为20×256、4heads、patch2。

`stage_a.json` 相比快照仅改变：

- 模型：20×256 → 32×256，仍为4heads、patch2。
- 条件路由（零基索引）：3,6,9,12,15 → 5,10,15,20,25。
- 初始阶段stop100000，后按用户指令延长200000；`max_steps/epochs`始终200000。
- 实验名、输出目录及说明文本。
- GPU预检后batch由288降为192；比较时必须同时按样本曝光量对齐。

继承 LR 5e-5、warmup1500、cosine、min_lr_ratio0.1、EMA0.999/interval4；
Writer32/Script16、冻结表、原FM/VAE及缩放、STD-soft/strong磁盘bank不变。
REPA/Critic/历史辅助监督关闭；不启用 Writer-only AdaLN、局部卷积或跳连。
使用既有训练器、统一构造及评测入口，不新增训练循环。

## 已验证与启动前门槛

实测参数 **38,810,806**。CPU B2：32块/4heads、准确路由、style_ada_rank=0，
4×32×32输出、有限值、反向、严格state_dict加载通过。
这不是 GPU 吞吐或收敛验证，也不是序列化完整训练恢复测试。

B192已完成80步真实训练入口预检（含优化器/EMA/STD bank/compile），后段6.03–6.05step/s，
日志allocated/reserved21.43GiB。首次编译计入的首段0.22step/s不作为吞吐。
未额外扫描更大batch；未单独预跑全套评测峰值，正式任务继承eval batch16/VAE batch8，评测与训练串行。
预检与正式训练完全独立，正式从零初始化、不接预检权重。

```bash
/opt/conda/envs/cu121/bin/python -u -m src.train.train --config src/train/configs/v92_32l/stage_a.json
```

使用隔离源码快照，记录源码/配置身份；不热改正在训练的目录。
本次实际部署复制远端v90B源码快照，只增加32层模型注册；已逐文件核对，Python差异仅`src/model/dit.py`。
没有部署本地新版训练器。源码逐文件SHA记录在远端`source_sha256.json`，配置SHA及启动信息在`launch.json`。

远端工作目录：`/root/Workspace/xy/DiT/scratch/v92_32l_20261018_01`。
正式日志：`logs/train_stage_a.log`；启动PID1731049；长任务记录`/root/.ssh-jobs/v92A-1731049/checkpoint.md`。
01:28 CST正式训练已过100步：Diff1.6416、REPA0、稳态6.05step/s、日志显存21.42GiB；GPU利用率100%。
短测推算100k纯训练约4.6小时，不含评测、初始化和长训速度变化；不是完整墙钟保证。

## 评测与后续

100k内存停止点无法仅靠修改JSON热更新；已部署`continue_200k.py`（PID1731714），
等待当前PID1731049正常退出并核验100k checkpoint DONE/size/SHA后自动`--resume-full`续至200k。
日志`logs/continue_200k.log`、续训日志`logs/train_to200k.log`；前段异常退出则报错，不回退旧权重。
200k LR horizon不变；完整状态续训不承诺DataLoader逐位重放。
Heun50/CFG1/EMA、固定噪声，保存同一187字完整产物；在相同样本上报告
SSIM、LPIPS、Ink-IoU、拓扑、独立Writer Top1/margin，并单列62个unseen，按字符做配对区间。
Writer-swap仅为诊断，不代替生成图评价。

Stage B 暂不实现训练配置、不启用：只有A成熟后，才从同一checkpoint分出等预算Control与
rank16/gain Writer-only分支（8,12,16,20,24,28），届时先检查零门控等价、有效梯度和加载。
不能用A的结果同时归因于新增风格结构；没有独立收益时不继续盲目加深。
