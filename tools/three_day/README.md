# 双机三天实验

用户原方案归档于 `ANALYSIS/three_day_20261026/plan.md`。不新增DiT/CalliFlow/VAE大训练，不重复strong/mild，不把未通过的教师接入主线。

## 已启动的有限串行

- 4090：`assets/three_day_20261026/`，主进程149231；A0→A3→A1→A2，每组P0@75k EMA→10k→20k。B512，总曝光各15.36M；主干LR1e-5、Writer/Script条件组5e-5，drop_writer两阶段均.10，seed90；每阶段重置优化器、EMA和LR，STD形变时钟连续。每5k保存完整checkpoint/优化器并评val100，保留PNG。
- 4090后续监督进程188021：等待**确切的Day1进程及完成产物**，再启动B0→B1→B2→B3各5k。进程退出但没有完成标记会停链，观测超时不会重启训练。
- 48：同名根目录，已完成一版F1/F2/F3/F4；完整补充版审计进程451697，增加局部Patch监督、逐图LPIPS/笔宽和合格独立评价器门槛。采用本机Deep114@85k，不传主模型。

预计A组稳定速度约1.62step/s，纯训练20.6h、含加载/评测约22–24h。具体以各阶段日志和耗时为准，不将步骤估算写成完成量。

## 实现边界

| 项目 | 状态 |
|---|---|
| A0–A3配置、严格EMA迁移、共同LR/dropout/阶段规则 | 已实现并启动 |
| 原始语义Writer ID→embedding行 | 新checkpoint保存`semantic_writer_rows`；旧raw映射可解析，含糊identity拒绝猜测 |
| 数据/STD索引/表哈希、源权重哈希、实际Writer×Script曝光计数 | 已实现；每100步及阶段终点落盘 |
| B0/B1/B2/B3 | 已实现；后续串行进程已挂起等待Day1 |
| B4/B5/B6 | 仍待完整教师门槛、可微训练适配器及队列接入；不会凭空视为完成 |
| R0–R3、C0–C3、T1–T4 | 尚未全部实现/挂起，继续按原方案推进；门槛未过的分支应明确跳过并转预算 |
| test200最终确认、四层级归档 | 候选冻结后执行，Day1不读test200选择模型 |

第一天继承现有P2的表可训练标志（false freeze），四臂一致；不是新做“对比表冻结/解冻”变量。P1混合复用旧冻结CSV（约70/30），实际Top10曝光单独计数，不能说四臂Top10曝光相同。

配对端点沿用现有采样器，改变的不只是梯度协方差，也可能改变边际分布；没有附加对比或反事实loss。B1在存在的Writer×Script组中分层，配对只从真实存在的同字同书体跨书家组合抽样；端点覆盖不完整处不会伪造组合。B2只以训练覆盖、STD复杂度和训练GT质量生成1–1.5倍权重，保留原配对端点分布，只重采普通部分；不读取测试失败名单。

B0–B3都显式生成条件dropout mask并禁止模型重复dropout，使B3只给**实际未被drop的Writer且t>.7**的样本1.5倍权重，再按权重总和归一化。为保持这条新接口的匹配对照，B0重新训练，不复用旧strong结果。

## 审计契约

- F1：精确加载旧Pixel D1源实现；DINO使用本机DINOv2-S/14权重，当前Transformers注意力参数名转换后严格加载，无新下载。不是DINOv3，也不是DINOv2-B。
- F2：48不同字符×3个实际噪声seed；1%相对latent L2半径，零/随机/反方向对照；训练书家原型排除查询字符，**不是同字硬负样本协议**。固定留字数据中硬负覆盖不足，已显式记录改变，而非静默补位。
- 全局和局部Loss分别验收。局部使用旧Pixel D1中间16×16特征、前景加权mean/std，未引入码本或新教师训练。
- 独立评价器自身必须先通过F1基本跨字/来源检查；弱DINO探针不充当否决或放行真书风的权威。通过也只允许小规模训练，不等于风格已被证实。
- F4中DINO到48个STD字符的检索仅是诊断，不是真实OCR错误率。当前GT检索很弱，不能直接接它作为内容loss。
- 所有模型评估均固定原checkpoint VAE/scale/shift、STD、EMA和采样。48逐图结果/小海报返回4090；不搬大生成器checkpoint。

## 操作

```bash
python -u -m tools.three_day.run --host 4090 --root assets/three_day_20261026 --detach
python -u -m tools.three_day.run --host 48 --root assets/three_day_20261026 --detach
python -u -m tools.three_day.later.run --root assets/three_day_20261026 --day1-pid PID --detach
```

同一根目录加锁，禁止重复队列。失败保留日志、child PID和returncode；修复后先核实旧child已终止，再显式`--retry-failed`。运行中断的训练阶段拒绝从零重跑，必须审查后全状态resume。

本轮额外归档了4090根目录一个旧的未跟踪`io.py`副本，原文件会遮蔽Python标准库io、使编译子进程启动失败；保存于`assets/three_day_20261026/io_shadow_20261025.py.disabled`。训练进程已正常推进，不为该警告重启。
