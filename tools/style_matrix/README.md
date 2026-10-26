# Writer 表征×注入矩阵：实现与 batch 测试

本轮输入始终为 `STD + Writer ID + Script + Noise`。基础结构为114M U-Net。当前计划为 **S0–S8 + S2-large，以及独立的J0–J3**；旧20格+4跨机复测已退为历史容量探针，R4预训练暂缓。

[逐步进度、原始测量与待办](../../ANALYSIS/style_repr_injection_matrix_20261026/README.md)是当前执行状态的入口。以下R/I说明是复用实现，不代表每个组合都要正式训练。

## 当前执行边界

两端有效batch统一512，禁止梯度检查点。最新策略：**安全范围内吞吐最快为准，不必填满20/40GiB，不重复大规模搜索**。
当前正式训练均未启动。普通FM四类batch搜索已在两端完成，真实P1数据短测通过；J3反向链通过但图像micro16占46.14GiB，较小micro及其余臂待验收。串行脚本和正式步数预算尚未完成。

## 模型与唯一 Writer 路由

模型实现：[src/unet/style](../../src/unet/style/)。每个文件承担一项职责：

- `representations.py`：R0向量；R1多Token；R2 Writer×STD CNN场；R3 STD连续几何软路由×16个Writer基底；R4与R1同形状的预训练Token查表。
- `injections.py`：I0通道AdaGN、I1空间AdaGN、I2四头Cross-Attention、I3八个rank16局部卷积基底。
- `interface.py`：新Style分支是唯一可变Writer来源；旧主干始终查同一null行。
- `transfer.py`：严格迁移公共内容权重；R4只能从导出的固定表初始化。
- `diagnostics.py`：关闭新Style后A/B输出逐位相等、匹配原null路径、条件响应及梯度/Token/空间诊断。

注入点为`middle2`、Decoder16/32最后一个ResBlock的共享AdaGN之后、输出卷积之前。不是只在最终图像上加一个风格偏移。
I3先计算共享低秩卷积，再对低秩通道混合，最后统一做1×1输出投影；它与先分别输出八个全通道基底再相加等价，但不保存昂贵的`[B,8,C,H,W]`中间张量。

R3全部几何来自当前输入STD，包含方向/置信度、宽度代理及变化、曲率、软骨架端点/连接代理、距离与相对位置。这些是条件特征，不能称为真实笔顺或拓扑监督目标。STD增强后重新计算，因此不会把未增强STD的几何错误配给增强后的字形。

## 梯度累积

新矩阵使用`runtime/train.py`，复用原仓库的缓存数据集、P2配对采样、STD bank、FM、EMA、LR调度与原子异步checkpoint组件。普通U-Net旧入口没有通用累积，已对不支持的microbatch显式报错。

一次有效batch的顺序：

1. 在有效batch512层面采样；配对部分约152个样本。
2. 为完整batch一次生成FM时间、噪声及Writer/Script dropout。
3. 按microbatch拆分；每份损失为`per_sample_loss.sum()/512`，允许末尾不足整份。
4. 全部反向结束后，只执行一次clip、AdamW、Scheduler和EMA。

例如microbatch224对应`224+224+64`，不是把最后64张当成224张加权。microbatch448对应`448+64`。
若未来增加跨样本对比损失，本实现不能直接把全batch对比项拆成互不相见的小份；本矩阵仅使用逐样本FM。

训练保留固定LR总horizon，阶段仅改变stop；续训恢复raw、EMA、Adam、Scheduler、所有RNG与配对采样的epoch/样本偏移。DataLoader使用独立generator，避免重建迭代器消耗训练的STD增强RNG。BF16与不同CUDA内核不保证完整轨迹逐位一致。

## 测试协议与产物

资产根：`assets/style_matrix_20261026/`，两端相同相对路径。

| 目录 | 内容 |
|---|---|
| `specs/{4090,48}/` | 各10个主组合+2个复测的配置 |
| `batch_tests/` | 原B256的24项短容量结果，保留为历史记录 |
| `micro_search_b512/` | R2配四种I，每候选5步预热+100个optimizer steps |
| `batch_tests_b512/` | 将选定默认值应用到各端12项组合，逐项20步复核 |
| `accumulation_check*.json` | 同一固定batch的梯度、loss和更新结果 |

既有搜索：4090候选microbatch192/224/256，48候选384/448/512，21/42GiB安全上限。历史代码按目标90%筛选；下一步直接从已有结果按“安全且最快”重选，不能继续把填满目标作为要求。
记录optimizer steps/s、samples/s、峰值allocated/reserved、GPU利用率、CUDA event测得的forward/backward/clip+optimizer/EMA时间。所有吞吐均按**有效batch完整更新**计数。

**这些是合成张量的容量与计算吞吐测试。** 使用实际114M计算图、反向、Adam与EMA，但不读取GT、不使用P1科学起点，也不保存可用于结论的训练checkpoint。为检查各新增模块梯度，探针临时把原零初始化卷积设成小幅非零。
R4的运行时结构与R1相同，容量探针使用同形状Token；这不代表真迹Token已经预训练完成。
测试默认eager/BF16，未将未经测量的compile收益算入ETA。数据读取、STD bank取样、评测和长期稳定性的开销要另外预留。

短测试入口：

```bash
python -m tools.style_matrix.probes.micro_search --machine 4090 --resident-pid <已核验的空闲常驻PID>
python -m tools.style_matrix.probes.micro_search --machine 48
python -m tools.style_matrix.probes.batch_sweep --machine 4090 --selected-defaults --resident-pid <PID>
python -m tools.style_matrix.probes.batch_sweep --machine 48 --selected-defaults
```

每个探针结束释放自己的CUDA上下文。全局`/tmp/dit_gpu0.lock`加锁；未知GPU进程、超额常驻占用、实现错误会阻止启动或停止本轮，不会杀掉其他项目。

## R4与公共资产准备代码

`prepare/r4_tokens.py`实现DINOv2局部Token提取、训练集内字符留出、同书体2×2书家×字符采样、跨字SupCon及同字硬负例、选定checkpoint后导出11×8×64固定Writer表。
只读取指定生成训练CSV；最终评测CSV的路径/像素hash命中会报错。正式FM时不读取参考GT，R4与R1一样允许Token表更新。
该脚本尚未在本轮执行，不把R4预训练成本计为零。

`prepare/package.py`可导出仅含原P1 EMA的紧凑checkpoint、Top10需要的原值latent子集、原值STD增强子集和映射表。导出逐项检查不改dtype、不改latent数值、不修改原始P1/P2/VAE。
`--seal`检查manifest hash；当前R4已延后，不要求其表文件。打包尚未实际执行，不能据代码存在判定48的数据已就绪。

## 64小时口径

64h指两机并行的自然时间，不是两机合计64 GPU·小时。最终以B512实测数据重算所有模型共同的训练阶梯与统一LR总horizon。
有效batch、optimizer步数、样本曝光在模型间相同；microbatch与累积次数可以不同。B256历史结果不能直接当作B512速度或曝光预算。
预算必须预留资产准备、J组共同梯度标定、加载/保存、统一评测、归档及长期吞吐余量。短探针估计不包装成已验证的完整长训ETA。

## J组实现边界

`src/unet/joint/objective.py`提供冻结编码侧的可选图像反向和Decoder重建；`losses.py`是独立连续Sobel损失，不依赖历史VAE工具目录。`joint/profile.py`只测容量与梯度连通性。
J0冻结Decoder/FM；J1冻结Decoder/FM+图像；J2可训练Decoder/FM+独立GT重建；J3可训练Decoder/FM+图像+GT重建。统一P2@20k EMA，与S组的P1起点区分。
图像分支有效样本数独立记录，FM仍为全有效512。正式loss系数校准、配对Decoder评测和端到端恢复仍待实现/验收；此版本不可直接认定为可长训版本。
