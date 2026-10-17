# U-Net EMA 修复与20k→100k续训

## 一、错误现象

旧run `Unet/20261017-233321-Unet-flow` 的EMA始终为初始化值；
raw在学习，但5k/10k/15k/20k历史EMA评测完全相同。
另有checkpoint顶层架构元数据缺失、VAE路径未映射两个已定位问题。
复现与输入充分；采用真实checkpoint及小型模型直接定位，未做Git二分。

## 二、根因分析

`update_ema`把compile源模型的`_orig_mod.*`和裸EMA参数名直接匹配，未匹配项静默跳过。
现修复同时解DDP/compile的source与target，首次建立缓存校验完整key集合及参数形状；
不允许零配对蒙混过关，也不破坏“source和EMA都compile”的DiT路径。
更新仍采用缓存张量对与foreach，热路径不新增逐参数建表。

## 三、最小修复及部署

- `src/train/ema.py`：规范化双方包装、完整配对；支持`ema_start_step`作为EMA warmup相对起点。
- `src/unet/io.py`：显式映射eval_vae_path；严格读取legacy args架构，仅raw可显式加载未验证legacy文件。
- `src/unet/train.py`：`--reset-ema`从恢复raw建立新EMA；恢复optimizer/scheduler/RNG，记录reset及LR horizon变化。
- `src/train/ckpt.py`已有本地架构元数据写入逻辑，本轮部署到远端新快照，远端原文件不覆盖。

source快照`/home/ds/Workspace/DiT/scratch/unet_ema_20261018_01`，
复制原远端src后仅替换上述四个文件及新配置；不混入本地其它训练器重构。
源码SHA与原20k checkpoint SHA见该目录`launch.json`。

## 四、防回归验证

`python -m pytest tests/training/test_unet_ema_resume.py -q --disable-warnings`：**5 passed**。
包含裸/compile混合、双方compile、浮点与整数buffer、缓存读取最新参数、错误架构失败及新warmup起点。
不跑全仓测试；正式训练20.1k额外保存一次checkpoint，验证真实网络的新EMA与身份。

## 五、恢复与运行身份

原checkpoint：`assets/results/Unet/20261017-233321-Unet-flow/checkpoints/0020000.pt`，
完整性size/SHA已核验，旧文件与旧评测不修改。

运行PID **3050580**；run `assets/results/Unet_ema_fixed_100k/20261018-020505-Unet-flow`。
配置`src/unet/configs/fm_48_resume_100k.json`，启动参数`--resume <20k> --reset-ema`。
B512、channels192/384/576、72,088,676参数及其它训练条件不变。
**max_steps=100000（累计总步数，不是再加100k）；LR horizon=100000**。
原horizon200k被显式缩短，恢复全局step20000后按新cosine重算当前LR为5.54347e-5；
不重跑LR warmup、不重置优化器动量。EMA decay仍.999/interval2，EMA warmup起点改20000。

日志已确认raw/optimizer/scheduler/RNG恢复、EMA配对**214/214**，新EMA以20k raw初始化。
这不是恢复历史真实EMA；历史EMA不可重建。也不承诺DataLoader逐位重放。
首次额外保存/评测20.1k，随后每5k正常保存评测，100k保存后退出。
纯训练剩余约8小时，另加评测、初始化及长训速度变化。

启动日志`logs/train.log`；长任务记录`/home/ds/.ssh-jobs/unet-ema-3050580/checkpoint.md`。
4090训练及其200k续训安排未修改。

## 20.1k真实验证

新checkpoint严格EMA加载通过，顶层架构/config正确、max/horizon100k、EMA start20000；
optimizer状态214项、scheduler.last_epoch20100。
214个EMA参数张量相对20k raw全部改变，并与20100 raw不同，输出层EMA norm1.19949（不再为零）。
首次有效EMA评测187字：SSIM **.57444**、LPIPS-VGG **.35783**、Ink-IoU .28588、Frag2.3996。
这些是原生浮点评测值，不与前文PNG统一重评SSIM混用。
与raw20k相比还多100步训练且LR horizon改变，不能把差值全部当作EMA单变量收益。
