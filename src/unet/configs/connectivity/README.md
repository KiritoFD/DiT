# U-Net 连贯性消融：实现完成，未启动新训练

先排除阈值/采样/EMA问题，再比较Skip与Decoder；不把“低Frag”单独当质量改善。
旧Frag5.52来自raw15k；修复后EMA20.1k的原生Frag约2.40，不混为同一个起点。
也不把EMA20.1k与raw20k的差值全归因于EMA：多了100步训练且LR horizon改变。

## 结构开关与参数量

尺度索引始终是 **0=32²、1=16²、2=8²**，decoder深度数组同序。

| 配置 | 相对72M的唯一结构改动 | 总参数 | 起点 |
| --- | --- | ---: | --- |
| A_control | 无 | 72,088,676 | 20k raw |
| B_skip32 | 3个scalar，只允许32²门控训练 | 72,088,679（仅新增1个可训练参数） | 同A |
| C_decoder | Decoder深度[3,3,2]，32/16各新增1块 | 76,000,868 | 同A |
| D_depth3_fresh | Encoder/Decoder每尺度均3块 | 93,045,860 | 从零，后置候选 |

按代码计数，基线14个ResBlock、5个Attention；C为16/5，D为20/5，纠正早期资料的16/4误计。

Skip门控`1+0.5*tanh(a)`，a=0初始化，范围(0.5,1.5)，允许增也允许减；
B的16²/8²参数被冻结，不靠梯度mask或weight decay维持。
Decoder新增块位于原Stage的blocks尾部、Attention之前；同通道shortcut为Identity，conv2为零。
默认不开gate、不加层，旧baseline的state_dict和构建随机数顺序保留。

P2提供独立`std_gate_scales`同型门控，默认关闭；诊断支持单尺度STD=0.8。
没有同时改STD增强分布，避免与Skip/Decoder收益混杂。
P3 Writer独立FiLM暂缓：当前没有后期风格退化证据，不增加这一变量。

## 受控加载、优化器和阶段预算

A/B/C的max_steps=50000（累计），从20k再训练30k；LR horizon统一100k，B512、seed90、FM/VAE/STD条件不变。
使用`--warm-start`，**不是完整resume**：三组都重建优化器和EMA、恢复同一个训练RNG状态及全局曝光时钟。
因此当前在跑的100k保留优化器主线**不等于这里的A**，不可冒充受控Control。

`src/unet/variants/transfer.py`先检查全部旧key、shape；只允许明确的新gate/新decoder块缺失，
用初始值填这些key后再strict=True加载。缺旧input.weight、意外key或不支持的Encoder变化直接失败。
从legacy20k只能选raw，不能加载旧失效EMA。结构升级的EMA也由升级后raw重建。
新checkpoint仍保存完整架构Config，后续`--resume`按同架构严格恢复。

```bash
# 只打印A/B/C命令，不训练：
python tools/analysis/unet_connectivity/ablate.py --checkpoint assets/results/Unet/20261017-233321-Unet-flow/checkpoints/0020000.pt

# 以后确认资源和诊断结果后才执行；--run会拒绝已有GPU计算进程，不会停现有任务：
python tools/analysis/unet_connectivity/ablate.py --checkpoint assets/results/Unet/20261017-233321-Unet-flow/checkpoints/0020000.pt --run
```

串行A→B→C，各自同一源checkpoint；子任务出错立即停止，不自动换权重或重跑。
请在包含本次提交完整src/tools的独立部署目录运行，不热改在训快照，不使用指向旧tools源码的软链。
D不在串行列表中，只有A–C结果明确且另行决定后才从零训练。

## 验证与验收

13个小型CPU回归检查覆盖EMA、非零输出下扩展前向逐位等价、新gate/conv2梯度、
严格旧key校验、推理干预保护、噪点vs真断笔、raw/OOD协议身份、STD探针hook收尾；没有启动GPU实验。
参数量通过实际构建计数，不承诺76M吞吐或显存已测。

P0入口：[诊断说明](../../../../tools/analysis/unet_connectivity/README.md)。
A/B/C最终需同时看187字同口径LPIPS/IoU、Writer及paired CI，及薄笔画召回、缺失/错误粘连。
若Frag下降但IoU、缺失、粘连或Writer变差，不能判成功。
