# experiments/ —— 实验数据与报告

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
