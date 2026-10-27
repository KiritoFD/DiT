# 磁盘台账

2026-10-27 15:18 `df -h`：4090仓库盘约1.5T可用；48仓库盘约101G可用。48仍有S组和J延长checkpoint待写入，串行每任务前要求至少25 GiB可用；此快照不保证后续容量。

更新：2026-10-27 03:45（Asia/Shanghai）。规则：**先归档、确认后清理；训练依赖与关键ckpt永不删除**。数据缺失优先本地重建（见[AGENTS.md](AGENTS.md)、[Data.md](Data.md)）。

## 15:18 48机候选范围

优先：旧`_archive_20261027/48_v2_execution_runs/`内J0@1k/2.5k/5k、J1@575、J2@50五个checkpoint，约10.7 GiB；当前matrix64对应副本独立inode且`.done` SHA一致，ready未引用归档。另有48机不完整P1残片约1.3G，当前ready未引用。旧`results_nonkey`约22G包含可能唯一的历史结果，只能逐项确认价值后选取。Qwen 8B共享缓存约16G不列为首批。具体路径、证据和排除项见[J组Loss及磁盘审计](ANALYSIS/style_repr_injection_matrix_20261026/j_loss_disk_20261027.md)。本次仅盘点，尚未清理；严格保护当前训练/评测、封存依赖及关键ckpt。

## 当前磁盘与队列写入预估

| 机器 | 总量 | 归档后可用 | 队列预计写入 | 清理archive后可用（预估） |
|---|---|---|---|---|
| 4090 | 1.8T | 254G（85%已用） | ~100G（5×50k S组 + 延长 + 评测） | ~370G |
| 48 | 1.8T | 129G（93%已用） | ~100G（J组 + 4×50k S组 + 12k延长 + 评测） | ~163G |

03:45时按当时估算写入余量足够；15:18当前可用约101G，后续写入仍需复核。48是较紧的一端，queue内建25G空闲保护（低于即停）。注意：`mv`为同盘改名，归档本身不释放空间，释放发生在确认后的删除。

## 归档清单（assets/_archive_20261027/，同盘mv，2026-10-27 03:40执行）

4090（合计130G）：

| 归档项 | 大小 | 内容 |
|---|---|---|
| `runs_legacy`（原assets/runs） | 49G | 旧时代训练runs（unet_decoupled、clean_cohorts、v11_pretrain、moyi、v15等） |
| `style_matrix_20261026/` | 14G | 旧矩阵/warm-start时代（joint_real_checks、final_checks、portable48、plan_final等） |
| `three_day_20261026/` | 11G | 旧实验资产 |
| `generic114_kxl_top10_20261023/` | 11G | 114M时代Top10数据（当前队列不引用） |
| `style_24h_20261026/` | 8.6G | 旧实验 |
| `u100_p1p2_20261025/`（p1_runs、p2_runs等，**p2_train.csv留在原位**） | 11G | **含关键ckpt P1@10k、P2@20k——永不删除** |
| `u100_pretrain_20261024_old_runs/`（runs_100k、ft_runs、ft_ext_runs、rasters256.pt） | 9.8G | 旧预训练runs；**bank/tables/val100留在原位（封存依赖）** |
| `decoder_only_20261026/` | 4.8G | 旧实验 |
| `reference_kxl_20261023/` | 4.5G | 旧数据 |
| `p2_std_consolidation_20261025/` | 3.5G | 旧实验 |
| `u100_style48_20261024/` | 2.2G | 旧实验 |
| `brush114_top10_20261023/` | 2.0G | 旧数据 |
| `cold50k_4090_execution/`（原cold50k/4090/execution） | 1.8G | 旧S0计划run（已迁移进matrix64）；plan/specs/ready留原位作记录 |

48（合计34G）：

| 归档项 | 大小 | 内容 |
|---|---|---|
| `48_v2_execution_runs/`（原48_v2/execution/runs） | 12G | 旧J0/J1/J2 run（已迁移进matrix64）；**calibration.json、train_unaugmented.csv留原位（封存依赖）** |
| `results_nonkey/`（unet_dic、ablation、v84D、v90A、calliflow_v2_120k、v81、v83、A1–A4、v71、cfg_sweep、eval_4090_b768、qwen_cond_eval） | 22G | 非关键历史结果 |
| `deep85_decoder/48/` | 5M | 首次配置失败目录 |
| `capacity_audit_test/`、`audits/`、`style_24h48_20261026/`、`style_space_audit_20261026/`、`stroke_writer_20261026/`、`three_day_20261026/`、`writer_latent_data_20261016/` | ~1.5G | 旧审计/实验 |

## 永不删除清单

- 两机 `assets/style_matrix_20261027/matrix64/`（运行中，含全部当前checkpoint与评测）。
- 两机 ready.json 封存路径（构建自封存清单，4090 38目录、48 48目录）：p2_train.csv、generic48_kxl latents与std_bbox64、u100_pretrain_20261024 的bank/tables/val100、Deep85 85k EMA、48_v2 的calibration与train_unaugmented.csv、eval CSV、VAE、GT图像目录。
- 关键ckpt（**在archive中的也不得删除**）：4090 `_archive_20261027/u100_p1p2_20261025/{p1_runs,p2_runs}`（P1@10k、P2@20k）；48 `assets/results/unet_deep_sym4_114m_60m`（Deep 50k/75k/85k，未移动）、`results/unet_connectivity_100000`（U100 Base 55k，未移动）、`results/qwen_cond_arm_{a,c}`（90k，未移动）。

## 清理确认清单（待用户确认后执行删除）

删除范围 = 两机 `_archive_20261027/` 中**除上表标注"永不删除"外**的全部内容；预计释放：4090 ~119G、48 ~26G。执行前逐项复核：目标路径仍不在封存清单、matrix64运行目录无引用、`exp.md`/`ckpts.md` 已标注归档位置。

## 变更记录

- 2026-10-27 03:40：归档执行（4090 130G、48 34G），归档后两队列计步复核正常（S0@7200、J1@1200）。清理未执行，等待确认。
