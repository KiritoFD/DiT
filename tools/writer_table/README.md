# Top20独立书家表修复试验

2026-10-22用户明确契约：书家表独立对比学习预训练，FM主训练期间冻结；不是冻结主干只训表。
旧U-Net错误地让表随FM优化；旧Top20初始库后10行还是随机初始化。旧实验保留原身份，不改写历史权重。

## 执行链

**最新状态：独立预训练已完成；下述30k→33k旧生成器换表续训未启动。**
用户最新指令优先准备48.33M下一代架构，未自动把旧修复续训插入队列。

正式原表导出至 `assets/data_cohorts/top20_20261021_v1/writer_tables_supcon_v1/`，
包含callig_table.npy、callig_remap.json、projection.pt、provenance.json、audit.json。
训练30386、validation200、test400；test未参与预训练或选点。5000步完成，validation选中2250步，
Top-1训练50.19%、validation52.00%；20类随机基准5%。行范数约1，最大异类余弦0.19639，无表塌缩。
这只是图像留出验证，不是严格跨字/跨作品泛化证明，也不是生成器监督质量保证。
下一代从零训练使用原单位范数表；下述对齐表仅供旧生成器兼容，不混用。

1. 只用冻结Top20 train的30386张GT提取本地DINOv2-small CLS，灰度3通道、256输入、ImageNet归一化。
2. 独立训练20×32表与小投影器：双向table-feature SupCon，跨字符同书家正样本，20位书家均衡采样，margin repulsion；5000步。
3. Validation200只用于独立表选点，不参与梯度；test400不进入表训练/选点。不会直接拿包含未知评测图的39万旧特征训练表。
4. 导出归一化原表；为接入已训练条件MLP，做全局正交坐标对齐，再保留原EMA每行范数。
   正交变换不改变归一化后的类间余弦几何；同时保存原表、变换和投影器，不伪称坐标天然一致。
5. 从Top20@30k完整resume raw/optimizer/EMA；先恢复checkpoint，再替换raw和EMA表、核对raw ID映射、清除旧表Adam动量、冻结整张writer embedding。
   保留来源raw checkpoint的CFG-null行并冻结；主干继续原FM训练，Script及其他配置不额外改动。
6. 先在30k换表后做一次validation，记录即刻条件偏移；继承50k调度和LR7e-5，停止于33k，31/32/33k保存并验证。
7. 检查31/33k raw和EMA表与外部表逐位相同；对这两个节点做Top20完整test/half-seen，再汇总Top2/5/10/20，共16项，与原30k对应档位比较。

未加任何生成器对比/风格loss，也不把GT特征用作生成条件。推理仍为STD、Writer ID、Script ID和噪声。
**这是一组“换表+冻结+继续3000步”的组合修复，不是从头训练的严格冻结对照，也没有等预算不换表Control。**
即使改善，也不能直接把收益全部归因于冻结；不改善也不能否定独立表路线，因为主干此前已适应旧的可训练表。

## 路径

根目录：4090 `assets/writer_table_repair_top20_20261022`。
独立表：`table_v2/`；日志`pretrain_v2.log`；生成器配置`train_config.json`，训练日志`train.log`。
状态`status.json`，固定身份`experiment.json`；最终`comparison.csv/json`、逐图CSV及`frozen_table_audit.json`。
续训输出：`assets/results/writer_table_repair_top20_20261022/时间戳-Unet-flow`。
首次DINO读取遇到dict-like Transformers输出，被误当成含x_norm_clstoken的hub字典，已修复并保留旧pretrain.log，不删除失败记录。

## Bug 分析报告

### 一、错误现象
旧FM主训练中writer表更新30000步，与“独立预训练、主训练冻结”契约不符；checkpoint可复现，初始化后L2漂移2.18253。

### 二、根因分析
nn.Embedding默认requires_grad=True，build未明确冻结，AdamW接收所有参数；Top20扩展后10行只有随机初始化，未独立对比预训练。

### 三、最小化Diff格式修复补丁
```diff
+ freeze_writer_table: bool = True
+ writer_table_file: str = ''
+ model.writer.weight.requires_grad_(not config.freeze_writer_table)
+ # after resume: install external table in raw+EMA; remove optimizer slot
```
历史checkpoint缺字段时显式还原freeze=False身份；EMA对source中requires_grad=False的参数使用精确拷贝，避免浮点滑动漂移。
修复顺序是恢复→换表→冻结→优化，不能在恢复前换表后被旧checkpoint覆盖。

### 四、防回归测试用例
`tests/test_unet_writer_table.py`：默认冻结/旧checkpoint身份、带旧Adam动量的恢复换表后raw+EMA精确不变、拒绝可训练外部表、DINO字典型输出识别。
四项轻量测试通过，无完整重训测试。

### 五、预防建议
配置和checkpoint记录冻结策略、外部表SHA与ID映射；主训练结束核验raw/EMA表不变；表预训练来源不再用“初始化”代替。
