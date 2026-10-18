# experiments/ —— 实验数据与报告

统一入口：[根目录实验总账](../../exp.md)（全量run、设置/数据/结果/图表）与[当前状态](../../state.md)。
本目录旧报告是各日期的证据快照，不直接代表当前机器状态；逐run明细见[ledger](ledger/)。

## 最近专题报告

- [V89 风格编码器完整实验，2026-10-16](2026-10-16-v89-style-encoder-complete.md)：独立的 20k 编码器训练、最佳点（3k）验收与留出评估；确认训练与泛化差距（训练 CE 99.77% vs 字符测试 30.07% vs 来源测试 13.64%），明确暂不接入作为生成监督。
- [后训练收尾与v88基模审查，2026-10-15](2026-10-15-posttraining-and-v88-review.md)：A100/B25/B50/C/D实际run来源、统一原生指标、CPU/GPU独立评分边界及v88早期快照；附[v86源码归档](../../archive/experiments/v86_20261015/README.md)和[代码维护说明](../development/recent-training-maintenance.md)。v88 最终在 170k 步平稳收官（SSIM 0.5776，收敛平台期），详见 [Infra Profile 实测报告](../infra-1015-measurements.md) §9。
- [Infra Profile 实测与评测 2.4× 提速报告，2026-10-16](../infra-1015-measurements.md)：单卡 4090 阶段级吞吐分析、批次/内存/编译开销；§10 真机实测评测耗时由 178–194s 压降至 73–87s（拔除 3 处 empty_cache、批次 8→64、尾批 pad、LPIPS GPU、多线程 CPU 指标）。
- [架构重构与评测正确性修复报告，2026-10-16](../refactor-report-2026-10-16.md)：PR0~PR4 重构、代码去冗余（瘦身 1000+ 行）以及六大评测协议正确性不变量落地。
- [从 SkelNet 到字符表与 VAE 的阶段回顾，2026-10-13](2026-10-13-vae-and-related-changes-report.md)：五个阶段、八张海报；补充 SkelNet 实际失败拼图、VAE 各代 DiT run 与跨 run SDE 推理结果墙。详细数据保留在 [技术附录](2026-10-13-vae-and-related-changes-technical-appendix.md)，附 [数据快照](recent_vae_20261013/)。

## 主面大表（2026-10-09 重抽, 统一 187 协议）★

| 文件 | 说明 |
|---|---|
| `mainline_20261009.csv` | **17 行主面大表**：v54~v71 + 容量阶梯 + moyi 复现，全部 `eval200_fixed` N=187 统一协议（ssim/lpips/mse + ckpt 位置）。README §2.5 表格的机器可读版 |
| 抽取脚本 | `_sync_work/extract_metrics_4090.py`（扫 4090 `assets/runs` 的 eval_auto json）、`_sync_work/extract_metrics_48.py`（ladder/v61/v60/v56/v71/moyi） |
| 主图 | `../system/imgs/fig_mainline_eval200_20261009.png`、`fig_capacity_ladder_20261009.png`、`fig_v68_family_20261009.png`（生成脚本 `_sync_work/plot_readme_figs{,2}.py`） |

## 全量实验表（2026-10-05 生成）

| 文件 | 说明 |
|---|---|
| `all_runs_20261005.csv` | **每 run 一行**，499 行。列分四组：① 位置（`run_dir`/`exp_dir`/`cfg_path`）② 训练量（`ckpt_n`/`ckpt_last`/`max_steps`）③ 条件构成 ④ 评测结果 |
| `all_runs_curves_20261005.csv` | 长表 `(run_id, run_dir, experiment_name, step, ssim, mse)`，870 点（来自 `eval_auto_*.json` 与 `eval_stdskel_summary.csv`） |

### 关键列

| 列 | 含义 |
|---|---|
| `eval_protocol` | **评测集口径**：`eval200fix+seen`（当前）/ `eval_seen_v10` / `fame_strict_clean_v8` / `other:...` / `unknown` ⚠ 跨协议数值不可比 |
| `best_ssim`, `best_step` | 该 run 的最优点 |
| `ssim_at_5000` … `ssim_at_150000` | 标准步点抽查（缺失为空） |
| `skel_latent_shards_dir` / `skel_kind` / `skel_cond_on` | 骨架条件来源（std / inst / gt）及是否启用 |
| `char_dino_embeddings` / `char_table` / `char_cond_on` | 字表（DINO 嵌入表）与 char 条件开关 |
| `callig_*` / `callig_used` | 书家表映射 |
| `glyph_inject_layers` / `w_glyph_cond` / `skel_as_glyph_cond` | 字形注入通路 |
| `w_latent_skel` / `inst_skel_shards_dir` | 骨架**预测目标**（结构损失）开关 |
| `n_eval` | 评测点数（0 = 该 run 无评测：smoke/中断） |

### 常用查询

```python
import csv
rows = list(csv.DictReader(open("all_runs_20261005.csv")))

# 1) 当前口径 + 有评测，按 best 排序
cur = [r for r in rows if r["eval_protocol"] == "eval200fix+seen" and r["n_eval"] != "0"]
for r in sorted(cur, key=lambda r: -float(r["best_ssim"]))[:10]:
    print(f'{float(r["best_ssim"]):.4f} @{int(r["best_step"])//1000}k  {r["experiment_name"]}')

# 2) 条件签名统计
from collections import Counter
sig = Counter((r["skel_kind"], r["char_table"], r["callig_used"], r["eval_protocol"])
              for r in rows if r["n_eval"] != "0")
print(sig.most_common())
```

生成器：`_sync_work/build_experiments_csv.py`（扫描 `exp-std/`、`assets/results/`、`_archive/`）
分析器：`_sync_work/analyze_csv.py` → `_sync_work/tables_20261005.md`
结论解读：`docs/system/76_experiments_20261005.md`

## 单次实验报告（保留原处）

近期报告以日期命名，例如 `2026-10-03-v46-singlestage-experiment-summary.md`、
`2026-10-03-twostage-archive.md`。更早的报告见 `docs/archive/`。
