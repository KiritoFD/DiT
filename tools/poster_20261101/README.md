# 跨血统 poster 与定量图工具

产出一张把 **近期 P125/abc 链**、**Moyi 两代**、**历史 CalliVAE / style_matrix 血统**
放在同一行轴上的 poster，以及配套的定量指标图。

结果与结论见 [docs/1031/poster_crosslineage_20261101.md](../../docs/1031/poster_crosslineage_20261101.md)。

## 关键约束：两族评测面板不同

| 族 | 面板 | 行数 | 代表 |
|---|---|---|---|
| 近期 | `assets/u100_pretrain_20261024/validation_val100.csv` | 100 | P3A、A@43429 |
| 历史 / Moyi | `exp-std/csv/eval200_fixed.csv` | 187 | v80、v85A、S9、S0、Moyi 12ch |

`g{i}.png` ↔ 面板 CSV 第 i 行（0 基），由 `in_mem_eval_sets="eval200fix:<csv>:<N>"` 的 N 截断。
两族**不能按行号硬拼**，因此 `pick_cells.py` 按 (书家, 字) 求交集，得到 21 个共有单元；
这 21 格在两面板中的 **GT 源图完全相同（21/21）**，故整表只需一列 GT。

## 流程

```bash
# 0) 取面板清单（val100 只在 4090，KB 级，允许直传）
scp 4090:/root/Workspace/xy/DiT/assets/u100_pretrain_20261024/validation_val100.csv scratch/panels/

# 1) 选格：写出 tools/poster_20261101/cells.json
python tools/poster_20261101/pick_cells.py

# 2) 校验本地三列（v80 / v85A / Moyi）与 GT 是否齐备
python tools/poster_20261101/validate_cells.py

# 3) 从 4090 取远端各列的 21 张（约 4.7MB）
scp tools/poster_20261101/pull_columns.sh 4090:/tmp/ && ssh 4090 'bash /tmp/pull_columns.sh'
scp 4090:/tmp/poster_imgs.tar.gz scratch/panels/ && tar xzf scratch/panels/poster_imgs.tar.gz -C scratch/panels/

# 4) 出图
python tools/poster_20261101/build_poster.py     # -> assets/posters/poster_crosslineage_21{,_raw}.png
python tools/poster_20261101/build_charts.py     # -> assets/posters/fig_p125_*.png
```

## 文件

| 文件 | 作用 |
|---|---|
| `pick_cells.py` | 求两面板的 (书家, 字) 交集，输出 `cells.json`（含 `idx200` / `idx100` 与 GT 同源校验） |
| `build_gt.py` | 从面板 CSV 的 `image_path` 取**原始真迹**做 GT 列，并断言每格 (书家, 字) 与 `cells.json` 一致 |
| `validate_cells.py` | 校验索引不越界、本地列与 GT 齐备 |
| `build_poster.py` | 合成 poster（9 列 × 21 行，行标签按书家着色，K 列描红） |
| `build_charts.py` | 两张定量图；指标只读官方 `eval_stdskel_summary.csv` / `eval.json`，**不重算** |
| `pull_columns.sh` | 按 `cells.json` 的索引从 4090 精确取图 |
| `pull_metrics.sh` | 取各列官方评测汇总（27KB） |
| `list_panels_remote.sh` / `locate_columns.sh` / `check_remote_panels.sh` | 远端面板盘点与定位 |
| `show_s1.sh` | 打印 F 计划 s1 消融结果 |

## 注意

- **GT 别用 moyi 那份 `gt{i}.png`**：那是减细/骨架化产物（墨量只有真迹的 39%、笔宽 1/3）。
  用 `build_gt.py` 从 `image_path` 取原始真迹。自检办法：GT 列 ink% 应与其他列同量级
  （实测 19.7% vs 15.6–20.5%）；若 GT 明显更淡，就是取错了。
- **不重算指标**：`build_charts.py` 只消费官方评测内核的输出。自己实现一遍 SSIM/LPIPS
  会和历史数字不可比。
- **不猜缺失数字**：v80 / v85A 的历史汇总在两机都不再留存，图上直接不画，而不是补一个估计值。
- Moyi 4ch@80k 只有一张 40 字官方海报、无逐行面板，故不进入行轴（避免拼错），
  在 poster 页脚注明。
- 面板图目录名虽叫 `eval200fix`，但 100 行面板也写这个目录名——**以 `g*.png` 数量判断面板**。
