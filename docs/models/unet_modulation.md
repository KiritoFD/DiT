# U-Net 调制与门控：实现及整理记录

## 三种机制分别控制什么

| 机制 | 配置 | 条件与作用位置 | 初始状态 |
| --- | --- | --- | --- |
| Skip / STD门控 | `skip_gate_scales` / `std_gate_scales` | 每尺度一个标量，缩放Skip或STD特征 | 增益1 |
| Writer-only AdaGN | `writer_film_rank=16`，默认0 | 每个ResBlock读取32D书家embedding，额外生成scale/shift | 额外scale/shift为0 |
| 条件残差门控 | `residual_cond_gate=true`，默认false | 每个ResBlock读取融合的time/writer/script条件，生成逐输出通道增益 | 增益1 |

所有开关可独立或组合使用，不更换GroupNorm，不改变FM目标、VAE或推理条件。
Writer分支读取dropout后的同一embedding；CFG无条件分支使用Writer-null，保留Script和STD。

## 计算与初始化

共享条件为`c=MLP(time(t)+writer(w)+script(s))`，当前不是完全解耦：

```
scale, shift = shared_affine(SiLU(c)) + writer_affine(writer_embedding)
h = GN(h) * (1 + scale) + shift
residual = Conv2(SiLU(h))
gain = 1 + 0.5 * tanh(gate_projection(SiLU(c)))
output = shortcut(x) + gain * residual
```

关闭Writer或gate时跳过对应项。Writer affine为32→rank→2C的MLP，末层零初始化；
gate_projection为512→C线性层，weight/bias均零初始化，gain有界于(0.5,1.5)。
这是AdaGN加残差门控，不称作AdaLN-Zero，也不额外门控Attention残差。

Conv2已零初始化，因此不能再让残差gain初值为0。当前gain=1：
从零训练时输出头先启动，随后Conv2收到梯度，再向gate及上游传播；从训练好的checkpoint暖启动时gate直接可学习。
Writer分支末层先学习，随后梯度进入低秩输入层，没有额外的零标量gate。

## 参数与使用

192/384/576、每尺度2块的实际构建计数：

| 配置 | 总参数 |
| --- | ---: |
| 原基线 | 72,088,676 |
| +Writer rank16 | 72,291,908 |
| +条件残差门控（单独） | 75,043,556 |
| Skip32 + Decoder[3,3,2] + STD三尺度门控 + Writer rank16 | 76,224,746 |
| 上一行再加条件残差门控 | 79,475,114 |

```bash
# 旧checkpoint增加结构：warm-start，统一重建optimizer/EMA；累计步数仍由调用方指定。
python -m src.unet.train --config src/unet/configs/connectivity/A_control.json --warm-start /path/to/0025000.pt --writer-film-rank 16 --residual-cond-gate --max-steps 60000 --lr-decay-steps 60000
```

JSON使用`writer_film_rank`和`residual_cond_gate`；CLI也支持`--no-residual-cond-gate`覆盖。
首次增加结构不能用同构`--resume`。`--warm-start`重建optimizer/EMA；
`--resume-expanded`严格迁移新增结构，同时按参数名保留旧Adam slots及旧EMA，新增参数无Adam历史。
该模式仅支持当前单optimizer group，不以参数位置强行对齐；新checkpoint保存完整Config，之后严格同构resume/eval。
迁移只允许明确新增的decoder块、gate、Writer分支参数初始化；旧权重缺失、移除已训练gate或改变Writer rank直接失败。
无新增开关时，不新增state_dict键，不改变原模型的随机数初始化顺序。

## 本轮整理与验证

范围为U-Net模型、配置/CLI、迁移和对应测试，不重写共享训练器。
原问题是零初始化与gain公式散落、扩展初始化混在模型构建中、文档仍保留旧100k状态。
采用小范围职责归位：

- `variants/modulation.py`统一零输出线性层和有界gain；不引入新的策略框架。
- `model._configure_modulation`集中安装逐ResBlock的可选支路，保持原参数路径及接口。
- `blocks.ResBlock`只负责AdaGN与残差计算，保留历史`film`/`writer_film`键名兼容旧文件。
- 本页汇总调制说明，模型总览只保留入口；历史实验记录不删除。

19项小型CPU回归通过，覆盖非零输出checkpoint的逐位前向等价、独立/组合开关梯度、
零Conv2启动无死锁、严格迁移与新checkpoint重载、Writer dropout/CFG-null、EMA及原诊断行为。
另核验CLI开关和Git空白检查。未做完整类型检查或GPU性能实验，不据此承诺提速或质量收益。

最新执行口径：旧60k组合任务取消，只跑原基线与原基线+Writer-only至100k。
同30k checkpoint保留旧optimizer/EMA分叉，两组各新增70k×512=35,840,000次样本曝光，
累计100k×512=51,200,000；这是重复采样次数，不是去重数据量。LR horizon统一100k。
不启用Skip门控、Decoder增深、STD门控或条件残差gate；4090不动。
预先实现的gate仍留作独立候选，需要另登记后才训练。
