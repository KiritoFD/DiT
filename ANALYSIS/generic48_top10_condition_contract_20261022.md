# 48M两阶段条件链路核查

结论：一阶段已有STD字形条件和Script书体条件，二阶段保留这两条路径，再新增独立的Writer书家调制。实现具备正确接入条件；这不等于已经证明微调后的书家风格更好，二阶段尚未运行。

## 条件分别代表什么

| 条件 | 一阶段通用预训练 | 二阶段Top10微调 |
| --- | --- | --- |
| STD字形图 | 启用，三尺度空间注入 | 保留相同缓存、几何和路径 |
| Script书体ID | 楷/行/隶，进入共享AdaGN | 保留预训练权重与映射 |
| Writer书家ID | 不查表、不投影，不参与输出 | 冻结32D书家表，经独立Writer-only AdaGN分支 |

“字体条件”在这里指STD图和书体ID，并没有另加一个TTF名称Embedding。具体TTF由书体与字符覆盖率选择，逐字记录在STD index中；不是要求生成图严格照抄标准字体。

## 一阶段的实际数据流

`tools/generic_pretrain/full.py`按书体名称筛选并重新映射，不能直接套用统一数据集的数值ID：

| 书体 | 统一源ID | CSV兼容ID | 模型连续ID |
| --- | --- | --- | --- |
| 楷 | 3 | 0 | 0 |
| 行 | 9 | 3 | 1 |
| 隶 | 11 | 4 | 2 |

`src/unet/data.py`把CSV的Script ID映射到连续ID。共享训练器在`use_script_cond=True`时取映射后的`y_callig_raw`和`y_script`，通过`build_train_model_kwargs`同时传入STD与Script，不会丢掉书体条件。

`src/unet/model.py`执行：

```text
STD64 → std_stem/std_down → 32²、16²、8²投影 → Encoder
Script16D + Time128D → 各自投影 → condition MLP → 各ResBlock共享AdaGN
```

配置`writer_mode=none`、`decouple_writer_condition=true`、`writer_film_rank=0`。Writer查表被跳过，`writer_proj`也不参与共享条件。预训练CSV全部Writer=-1，但这是未知标记，不会被当成一个可学习书家类别。

Script正常参与训练并有CFG dropout：`drop_all=.05`、`drop_script=.08`，约12.6%的样本替换为Script-null；STD不随此dropout清空。这不是无书体训练。`y_char`只兼容训练接口，没有字符Embedding进入网络。

STD使用实际墨迹bbox等比最长边fit256，再BOX到64；模糊sigma=0、形变增强off。VAE保持SD EMA，posterior mode，scale0.18215、shift0。

## 二阶段如何加入Writer

二阶段是同一U-Net，不是重新初始化主干。`curriculum/transfer.py`加载50k EMA，检查结构、VAE、STD及映射契约；旧张量必须全部加载，只有新加的`.writer_film.*`键允许缺失。

Middle两个ResBlock及Decoder三个尺度共26个ResBlock新增：

```text
Writer ID → 冻结10×32独立对比学习表 → Linear32→16 → SiLU → zero Linear→2C
                                                               ↓
共享AdaGN的scale/shift + Writer专用scale/shift
```

新增末层权重、偏置均为零。因此刚加载后输出与一阶段相同，不会突然破坏预训练；但预训练残差/输出头已非零，Writer末层能立即获得梯度，不存在双重零初始化导致的死分支。第一步末层学习后，梯度继续进入前层。

书家表按raw-ID从独立对比学习Top20表选取对应10行，不重新随机生成；主训练、微调和EMA中均保持冻结。旧共享Writer投影一直关闭，Writer只走专用分支。

Writer-null有显式mask：先计算分支输出，再乘`writer != num_writers`。即使分支bias训练成非零，通用回放和CFG-null仍严格为零；只是把Embedding置零并不能保证这一点。Writer CFG保持STD和Script不变。

## 微调优化与串行预算

- 一阶段50k，LR7e-5，原50k余弦调度，B512；每5k保存、评估。
- 二阶段10k，前1k仅训练Writer分支；之后联合训练Decoder、Middle、Up、Output。Encoder、STD分支、Script条件路径和书家表继续冻结。
- 分支峰值LR5e-5，联合主干组峰值LR1e-5；warmup200，10k余弦调度。二阶段新建优化器、调度器和EMA，不继承预训练的优化器时钟。
- Top10训练23,485、验证100、测试200；另以明确10%采样概率混入通用训练图，不是按CSV大小混合。回放样本Writer=-1映射到null10。
- 1k切换联合训练时清除EMA冻结参数缓存，防止新解冻参数一直被当作冻结参数逐位复制。
- 微调在5k、10k保存和评估；不会保存1k检查点，符合最新每5k保存要求。

## 当前实验身份与验收

4090实验目录：`assets/generic48_kxl_top10_20261022`。通用train100,263/validation1,046，排除369张受保护留出近重复；这不是作品级来源留出的证明。124,894条VAE缓存包含两个阶段所需的训练/验证样本，预训练只读取自己的train.csv，不会因为缓存合并就读入Top10 CSV。

活动进程最初读入30k停止点。接管runner保留该训练进程；30k正常完成后完整状态续到50k，50k LR调度不变。然后从50k EMA初始化Top10。DataLoader恢复不宣称逐位重放一致。

实际后续配置是`pretrain_50k_config.json`和`top10_50k_config.json`，每个新run保留自己的resolved_config。原train_config.json的30k是历史身份，不能当作当前最终预算。通用模型48,325,668参数，Writer版48,504,772参数。

本轮只做15项轻量CPU检查，全部通过：STD/Script输出敏感性和梯度；跳过Writer查表；严格权重转移及零初始化前向等价；零初始化Writer梯度和更新后Writer-swap响应；非零bias下null隔离；冻结表与EMA；解冻EMA刷新；回放概率；bbox几何。

```powershell
python -m pytest tests/test_unet_curriculum.py tests/test_unet_writer_table.py tests/test_unet_regularization.py tests/test_std_bbox.py -q
```

这些检查证明条件接入及更新链路正确，不证明10k足够或真实风格一定提升。二阶段完成后仍须以固定seen/unseen生成图、独立Writer及内容指标验收，而不是只看训练loss。
