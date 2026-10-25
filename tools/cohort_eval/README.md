# 单 checkpoint / 单集合评测

适用于当前独立 FM U-Net checkpoint；严格重建架构并加载指定 EMA（默认）或 raw。
与正在运行的10组矩阵队列互相独立，不重启、不改写旧队列。

## 1. 一次生成训练拟合探针（CPU）

```powershell
python -m tools.cohort_eval.datasets
```

默认固定源：`assets/data_cohorts/top20_20261021_v1/datasets_final`。
生成位置：同根 `train_probes_v1`。已有目录拒绝覆盖；重采样必须另指定`--output`。

| 层级 | test | train_equal | train_half |
| --- | ---: | ---: | ---: |
| Top2 | 40 | 40 | 20 |
| Top5 | 100 | 100 | 50 |
| Top10 | 200 | 200 | 100 |
| Top20 | 400 | 400 | 200 |

每位书家分别20/10张，尽可能匹配对应测试集的书体比例；不足时按可用训练样本分配，
实际 writer/script 数量写入manifest。half嵌套于equal，跨层级也严格嵌套。
来源严格为train.csv，不改变train/validation/test划分，不经过测试结果挑样本。
路径、源CSV与输出SHA、种子和各组计数均留档。

## 2. 指定权重 + 集合

```powershell
python -m tools.cohort_eval.run --checkpoint PATH/0030000.pt --collection top10/test --out assets/audits/my_test
python -m tools.cohort_eval.run --checkpoint PATH/0030000.pt --collection top10/train_equal --out assets/audits/my_train_equal
python -m tools.cohort_eval.run --checkpoint PATH/0030000.pt --collection top10/train_half --out assets/audits/my_train_half
```

`--collection`也支持任意CSV路径、`topN/validation`；不按CSV隐式截断。
支持`--state ema|raw`、`--steps 50`、`--cfg 1`、`--batch 4`、`--poster`。
`--root`选择冻结数据根，`--probes`选择探针目录。默认关闭巨大自动海报，但保留全部PNG。
输出目录必须不存在，避免不同权重/数据/协议被混在一起。

固定CSV自带逐图噪声种子原样保留；无种子的自定义CSV按图片路径和`--seed`补齐，
不依赖行号，子集与重排不改变同图噪声。`--seed`不覆盖已有逐图种子。
执行时校验书家/字符/书体在checkpoint映射表中合法；训练探针必须全部属于该权重的训练CSV，
test/validation别名必须与之无图片路径重叠。自定义CSV也报告训练成员数，不伪称留出数据。
原始数据已做的重复/来源分组隔离由冻结manifest承担，这里的成员核对不替代像素近重复审计。

## 3. 产物与解读

`identity.json`：权重文件SHA、state、实际步数、源/有效CSV SHA、训练成员数及完整配置。
`dataset.csv`：本次完整样本和实际逐图噪声；UTF-8无BOM以保留原生逐图日志的路径列。
`RESULT.json`、原生summary/batch CSV、PNG和eval_protocol/DONE：与统一评测器同口径。
包含SSIM、VGG-LPIPS、Ink-IoU、Frag、软墨量比和近空白率；缺核心指标或样本数不符直接报错。

训练探针仅诊断拟合，不参与checkpoint选择。比较相同checkpoint、同样协议下train_equal与test；
train_half用于观察估计随样本量变化的稳定性。数量匹配不等于字符/作品难度完全匹配，
不能把均值差直接全归因过拟合；还要结合逐书家/书体/seen字符分层。
若做三角矩阵，每个有效模型×层级再测两份训练探针，即10条test +20条train结果。

## 4. 所有已保存checkpoint × 所有支持档位

```powershell
python -m tools.cohort_eval.batch --output assets/audits/cohort_all_ckpt_20261022 --batch 32 --vae-batch 8
```

2026-10-22 用户要求停止Top2训练，改为全部权重评估：Top20/10/5各6个（5k至30k），
Top2仅有5k、10k完整权重，共20个。test + train_half（half-seen），三角展开112项。
half-seen明确指训练半量探针，并非“50%训练样本+50%未见样本”的混合集。
固定EMA/CFG1/Heun50；不等待Top2 30k，不补不存在的中途checkpoint。

每个权重只生成本层最大test和train_half一次，再按固定路径、逐图噪声和样本索引汇总
嵌套档位。复用旧30k完整test必须通过权重文件SHA、CSV SHA、协议及PNG/指标DONE完整性校验。
全部较小档位仍保留逐图CSV和原始PNG目录索引，不重复采样，不降低评测步数。
训练已停，batch32 / VAE8使用空闲GPU加速；最新权重优先，随后向较早步数展开。

`plan.json`：固定权重清单与预期结果量；`results.csv/json`：增量更新完整矩阵；
`fit_gaps.csv`：同checkpoint同档位train_half与test之差；`status.json`：当前任务/错误/完成数。
聚合来自原生逐图CSV的4–5位小数，SSIM舍入误差最多5e-5；所有档位使用同一聚合方式。
原生汇总及PNG完整保留，不将舍入后的聚合声称为未舍入浮点精确值。
