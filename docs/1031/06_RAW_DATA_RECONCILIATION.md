# 直接核查仓库 CSV/JSON：历史曲线、P75口径与数据表结构错误

> 2026-10-31 本次对 **GitHub中已实际提交的结构化数据**执行读取/逐行解析（未访问服务器运行盘）。此页用于纠正仅转述文字报告的潜在歧义。实际来源：[`assets/ledger/points_all.csv`](../../assets/ledger/points_all.csv)、[`assets/ledger/runs_all.csv`](../../assets/ledger/runs_all.csv)、[`p125_metrics_full_4090.csv`](../../p125_metrics_full_4090.csv)、[`tools/all_experiments_eval.json`](../../tools/all_experiments_eval.json) 与 [107统一复评](../107/02_model_benchmarks.md)。

## 1. 可直接核对的数据范围

| 源 | 仓库实际内容 | 允许的用途 |
|---|---|---|
| `assets/ledger/points_all.csv` | **616条**评测点、34列；有 `family/run/run_key/instance/set/step` 身份字段 | 重建旧v68/v70/v72等**原生同口径轨迹** |
| `assets/ledger/runs_all.csv` | **36条**历史run快照 | 查对应run和ckpt归属，不代表2026-10-31全部实验 |
| `tools/all_experiments_eval.json` | **278条**更早的异构实验评价 | 旧路线档案，需按其自己的协议、指标实现解释 |
| `p125_metrics_full_4090.csv` | **7条数据记录**、表头15列，但有行实际18列 | 先修表头/重复行才能入长期统一指标库 |
| `assets/master_results.csv` | 跨多run、配置字符串与多种protocol的宽表 | 只能以 run/protocol/metric_impl 联合身份join，不能用model短名合并 |

这些是**数据文件物理行数**，不是“616个独立模型”或“278个独立试验”。不同 instance/seed可能拥有同样的run短名。字段里的 `eval200fix` 也不能单靠名字证明真实字符名单等于今天的h187。

## 2. 直接从616条历史点抽出的等step曲线

下面仅取 `set=eval200fix`、`n=187`，每行展示源文件实际数字。v68在5k有**两个不同instance**（`20261006-180035` 与 `20261006-192021`）；为匹配后续完整轨迹，表中统一采用后者：

| Step | v68_aug_sp_c2ot | v70_aug_sp_stdskel_c2ot | v72_callivae_kl1e6_s12500_frozen_sp_c2ot |
|---:|---:|---:|---:|
| 5k | **.522465** | **.534108** | .524689 |
| 30k | **.549509** | .545722 | **.584405** |
| 50k | **.570074** | .549561 | **.599918** |
| 65k | **.583309** | .553535 | **.603436** |
| 200k | **.614868** | — | — |

SSIM均来自 `points_all.csv` 原生评测。源文件中 v68@200k另有 LPIPS **.324850**、InkIoU **.304417**；v70@65k LPIPS **.384606**、InkIoU **.240574**；v72@65k LPIPS **.397357**、InkIoU **.344583**。

### 与旧叙事相比需要修正什么

1. **v70 不是“从一开始就不能学”**：5k v70 比选定instance v68高 **+.01164 SSIM**；到30k v70低 **−.00379**，65k低 **−.02977**。这更像**早期优势消失、训练后期内容拟合/条件路径出现相对差距**，但不能在没有单变量模型对照时把它确定归因为“STD天花板低”或“空间注入错”。
2. **v72 的优势确实出现在早期训练**：对 v68，同50k SSIM **+.02984**、65k **+.02013**。不过 v72同时改变VAE、增强、数据/latent合同等，**数学上不能从这组曲线单独估计VAE贡献**。
3. **v68 后期不是绝对完全不再变化**：130k **.609855**、190k **.614752**、200k **.614868**；130k之后还有约 **+.00501**，但是200k附近增量很小。应说“后期边际收益减弱”，不是“130k后严格无效”。
4. 另一 [107统一187复评](../107/02_model_benchmarks.md) 的 v68@200k 约 **.6169**，与该原生ledger **.614868**不同；两个都可保留，但它们来自**不同的评测记录/实现或采样链**，未经逐图核对不能替换、混算或推算校正系数。

**可检验理论修正**：[数学目标](../math/03_LATENT_DECODER_AND_OBJECTIVES.md) 中“STD条件不是开集画质保证”得到更细的曲线证据——v70早期会受益，后期落后，因此下次应**按训练时间和复杂度层次**检查真实STD的有效性，而不是仅选一个最差终点指责结构条件。

## 3. P75 原始CSV存在两个不能忽略的独立问题

### 3.1 不同评测实现的分数不能互抄

该CSV的 `ssim` 列在 `protocol=heldout` 下写入：

| model（CSV原名） | `p125_metrics_full_4090.csv` 的 SSIM | 原生serial/val100叙述的SSIM | 数值差 |
|---|---:|---:|---:|
| P0@50k | **.5642** | **.5573** | **+.0069** |
| P1@55k | **.5729** | **.5636** | **+.0093** |
| P2@120k（实际累计70k） | **.5965** | **.5857** | **+.0108** |

来源：表的左列为仓库实际原始 `p125_metrics_full_4090.csv`，右列来自 [1030p125.md `2.8/`2.9](../../1030p125.md)。CSV本身将感知列明名为 `lpips_vgg`，不得简单与serial其它 LPIPS列混用。**目前没有核查两者的逐图名单、SSIM窗口与完整方法版本，不能宣称差异全部由某个固定函数参数造成**；应保留 `metric_impl+input_ids` 双重身份。P75的单协议内部趋势仍可看，但旧serial的绝对值不能直接替换成p125_metrics_full的绝对值。

S0@85k同样：当前CSV的 `heldout187_seed0` 记录 `ssim=.6081`、`lpips_vgg=.3421`、`eg_auc=85.99%`；而此前serial的正式SSIM记录 `.5963`。这个案例已经由仓库记录为指标实现差异；不同测量必须在数据仓库里共存。

### 3.2 数据结构有真实错误，不能直接读成整洁的15列CSV

按标准CSV引号转义解析，文件有**15个表头字段**；7条数据记录的列数为：

| 数据行（不含表头） | Arm/协议 | 实际列数 | 问题 |
|---:|---|---:|---|
| 1 | S0@85k / heldout187_seed0 | 15 | 与表头一致 |
| 2 | S0@85k / heldout187_seed1 | 15 | 与表头一致 |
| 3 | S0@85k / heldout187_seed0 | **18** | **重复seed0**，附有3个未命名字段 |
| 4 | S0@85k / heldout187_seed1 | **18** | **重复seed1**，附有3个未命名字段 |
| 5 | P0@50k / heldout | **18** | 额外3个无表头字段 |
| 6 | P1@55k / heldout | **18** | 额外3个无表头字段 |
| 7 | P2@120k / heldout | **18** | 额外3个无表头字段 |

那三项额外列的数字/对象可能包含额外Writer统计，但**仅从CSV无法安全赋予字段名**。不能根据排列猜测直接建表，须回读生成脚本输出schema，再写全字段表头、重新导出，按 `arm+真实ckpt_sha+protocol_id+seed+eval_impl_hash` 去重。若两次同键是真正不同运行，就应明确版本而不是删除。

这个问题**不意味着前面的15列数值必定错误**：在这些行里前15列仍有对应的SSIM/LPIPS/IoU读数；但它会使常见 `csv.DictReader` 对多余列填入 `None` 键、造成自动ETL/聚合错位或重复加权。记录可靠性的修复应先于“画更多趋势曲线”。

## 4. 长期归档与新的验证脚本要求

为每张导入表新增：
- `source_path`、`source_commit`、`source_row`、`instance`；
- `metric_impl_hash`、`protocol_csv_sha`、`valid_n`、`solver/NFE/CFG/seed`；
- `step_internal` 和 `step_real_cumulative` **两列**；
- `model_arch_resolved`、`VAE/Decoder版本`、`writer_remap_sha`；
- 缺字段为NA、冲突**保留两行**；不以同 `model+step` 名称覆盖。

**最小CPU复核**：
1. 从 `points_all.csv` 按 `run+instance+set+step` 精确查询上述同step 187轨迹，避免 v68 的两个5k instance被无意混合；
2. 对 `p125_metrics_full_4090.csv` 断言所有数据行字段数等于表头字段数；当前应触发失败，修schema后再用于合并；
3. 复评同一批 PNG 的两个 SSIM实现与 LPIPS网络，输出逐图差/窗口参数，在新的 `metric_impl_hash` 之下保存结果；
4. 不声称“val100全seen/一半unseen”已核对，先以训练字符并集重建。

## 5. 对下一步训练的直接影响

**不需要为了解释旧 v68/v70/v72 再重训**。已有结构化轨迹已经说明：条件表示/Decoder/数据配方的效果高度依赖训练阶段；训练时间是强混杂变量。实际下一步应把已有A/B/C分别按**相同真实更新数、目标Writer有效曝光、固定指标实现**比较，再用按时间分段的Writer速度场诊断验证“条件是否学足”。

如果现成原始CSV未清洗、真实架构不明，继续扩大 P0 或新训练 Writer模块只会制造更多难以归因的数据。本阶段优先把**评价器和实验身份恢复为可复现的科学测量**。

归档结论参见 [1031证据矩阵](01_EVIDENCE_MATRIX.md)、[理论修订账本](../math/00_THEORY_REVISION_LEDGER.md)、[下一轮验收](05_NEXT_EXPERIMENTS.md)。
