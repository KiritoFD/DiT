# 第二代双机完整队列：24 / 48 / 64 小时

2026-10-28 23:10更新。两端控制器已挂起，计划根为`assets/style_matrix_20261028/generation2full/{4090,48}`。`status.json`登记阶段，`tasks/*/plan.json`是不可变执行清单，`ready.json`封存代码与输入，`clock.json`固定64h截止时间。预算是累计墙钟上限，阶段可以提前完成，不等待整点。

## 已挂入的任务

| 阶段 | 4090 | 48GB |
|---|---|---|
| 24h交付 | S9冷启动50k → 同一个50k权重分叉：S9-FM到60k、S9-Pair到60k | J4 → J5 → J6 → J7，各独立Deep85 EMA起点到5k |
| 48h交付 | S10冷启动50k → S9-FM到75k；Pair通过内容/方向检查才同样到75k | Decoder条件线通过门槛则选一对到12k；随后S11冷启动50k |
| 64h交付 | S10成熟75k、等曝光S0或S4到75k；若新冷启动模型明显退化，改做最优新模型独立seed | S11成熟75k、独立seed的S11到50k；最终187字×3 seeds与机制检查 |

4090的S8已完成50k及评测，旧矩阵的75k shell入口已停用。48的J0延长停于11,654；新J4为部署完整控制器暂存于687步后原样恢复，计步、数据offset、Optimizer、EMA和RNG均延续，不重置到零。

24h两条S9分支都固定+10k；不会因Pair较慢而只给它更多更新。每个任务包含原过程评测、统一187字seed0及共享字诊断。最终候选增加seed1/seed2，保存独立结果和逐图输出。

## 训练契约与实现

- S9：114.00M内容主干+13.56M三尺度Writer速度分支，总127.56M。主干不接Writer；null残差最终硬置零。Writer分支8/16/32三个尺度各5个ResBlock。
- S10：同一速度分支，输入增加局部法向、宽度、signed EDT和曲率。EDT在32²条件网格用可编译的可分离欧氏距离计算；与SciPy数值核对一致。无硬分叉计数。
- S11：学习式局部Attention分支，总127.73M；Writer查询内容局部tokens并将上下文送回多尺度残差。它测试学习式局部聚合，不引入额外风格Loss。
- J4/J5/J6图像支路截断`z_hat`，U-Net仍更新FM；J7保留图像到U-Net的梯度。J5在原Decoder的128/256层加入Writer+Script仿射，J6/J7使用带SiLU的Writer×STD空间条件头。Decoder深度和VAE编码坐标不变。
- 有效batch512、FM micro256；J图像每步16张、micro8；BF16+compile，无梯度检查点。

### Pair不是固定小权重

物理`t=1`对应模型时间1000；同字同书体A/B共享噪声和STD，差分真值为`z_b-z_a`。使用归一化SmoothL1。

第一步和每500步读取实际参数梯度范数，将Pair梯度标定到FM的25%；校准权重、实测比例和步数随checkpoint保存。校准直接利用本步FM和Pair反传，暂存FM梯度，不重算前向。未达到20%–30%或梯度非有限时直接停止。普通步骤只追加Pair反传；辅助RNG隔离，不改变下一步FM采样。

分叉仅建立本机只读checkpoint软链接，检查`.done`及身份。两分支从完全相同的模型、Optimizer、EMA和RNG恢复。续训LR重启锚点固定；中断恢复不重新开始warmup。

## 自动晋级门槛

Pair到60k后，如果SSIM比Control低超过0.01或LPIPS高超过0.01，且独立Writer margin与GT差分方向都无改善，则不再延长Pair。

Decoder晋级要求同时满足：SSIM退化不超过0.005、LPIPS退化不超过0.005、Ink-IoU退化不超过0.01；Writer margin至少+0.005且Macro Top-1不下降；共享字诊断至少6个字组、方向cosine的按字bootstrap下界大于0，且均值至少+0.03。按J7−J6、J6−J5/J4、J5−J4选择对照；均未通过则直接进入S11。该门槛用于预算筛选，不是正式风格改善的统计证明。

S9和S10都在50k明显弱于同协议S0时，4090停止追加旧基线预算，给较好的新结构补独立seed。否则根据内容/Writer表现选择S0或S4作75k等曝光对照。

## 评测口径

S侧书家编号与J侧不同：统一187字CSV按训练清单的书家姓名、书体和Unicode构造，已核验两侧全部187行映射与STD覆盖。每个seed中，同字同书体的不同Writer共享初始噪声。

除SSIM/LPIPS/Ink-IoU/Frag外，自动计算source-balanced独立Writer Top-1/margin及同批GT ceiling。GT–GEN差分记录方向cosine、投影和残差；特征包含去全局墨量偏移的低分辨率形状和几何统计。

共享字面板来自明确标注的训练GT，用于机制诊断；不能将其结果作为未见字泛化证据。187字沿用历史固定集，不能将其中所有字符都称为未见字；原S侧val100另行保留。三seed结果独立保存。

S候选额外测5个时间点的`R_style=RMS(v_writer)/RMS(v_content)`、null严格归零和Generator Writer-swap。J候选固定采样噪声生成latent，再以原Decoder和配套Decoder解码；条件Decoder增加只交换Decoder Writer的对照。每项输出独立目录，条件模块随raw/EMA正确加载。

## 已完成的预检与ETA

| 检查 | 结果 |
|---|---|
| 4090 S9+Pair完整更新，含Adam/EMA | B512/micro256，峰值13.27GiB，稳态短测0.973 step/s，Pair实测比例0.25 |
| 48 S11完整更新 | B512/micro256，峰值13.50GiB，短测1.332 step/s |
| 48 J7真实数据+图像反传 | micro8累计16张，峰值40.46GiB，已完整保存2步smoke |
| 新J4正式训练 | 恢复后已超过1.3k，累计约0.75 step/s，编译阶段峰值41.13GiB |
| 数学/路由检查 | Pair真值符号正确；三种分支null严格为零、content独立于Writer；梯度预算均为0.25；EDT与SciPy一致 |

小型原始证据见[evidence](evidence/)。J7 smoke只是链路/容量验证，不能作为模型质量结论。

4090首24h任务估计约16–20h（50k普通FM、10k Control、10k Pair，加编译评测），后续按正式200–500步吞吐更新；0.973是Pair短测，不能直接当普通FM速度。48的J4–J7估计约8–10h，后续12k晋级和S11合计通常仍能落在48h内。64h为硬上限，达到时安全保存并停止。

## 鲁棒性与登记

- 每任务启动前检查封存、GPU所有者、全局GPU锁和25GiB磁盘门槛；未知失败保留现场并停止，不无限重试。
- 4090原有约1.1GiB外部常驻上下文PID从1967368变为2107710；容器内无法读取其进程信息。当前按已观测的固定PID与原1.5GiB上限放行，保留该上下文；若所有者再次变化仍会拦截。
- 24/48/64h自动写预算快照，保留已完成的score路径与当前命令。控制器可重启，已完成训练/评测不重复执行。
- 本轮仅同步源码和小型清单；训练数据、bank、VAE、分叉checkpoint均复用各机资产，未跨机搬权重。
