# 历史 eval 图归档（从 4090 拉回，2026-11-01）

**来源**：4090 `/root/Workspace/xy/DiT`，只读抓取，未修改远端任何文件。
**规则**（`scratch/_stage_eval_imgs.sh`）：169,789 张历史 eval PNG 全量是 GB 级，不适合入库，
因此按「信息密度」而不是「数量」取：

1. **海报 / 网格 / 汇总图**：全部拉回（`poster*` `*summary*` `*grid*` `*panel*` 以及 `assets/posters/`）。
2. **小型评测元数据树**：`eval_hwd_20261029`、`full_eval_raw`、`eval_strata` 整树。
3. **面板样本**：每个 `eval_samples_ctrl/<step>/<set>/` 取前 6 张生成图 `g{i}.png` + 前 6 张真迹 `gt{i}.png`（确定性排序，可复现）。
4. **墨迹评测**：`assets/ink_eval/<子目录>/` 各取前 6 张。

合计 **2,767 张 / 126 MB**。

## 分布

| 区域 | 文件数 | 大小 |
|---|---|---|
| `docs/archive`（历史报告海报） | 36 | 30.3 MB |
| `assets/posters` | 10 | 17.0 MB |
| `assets/audits`（cohort/extreme 矩阵） | 936 | 15.0 MB |
| `assets/eval_hwd_20261029`（heldout 面板） | 62 | 12.3 MB |
| `assets/u100_control48_20261024` | 300 | 10.2 MB |
| `assets/generic48_kxl_top10_20261022` | 210 | 7.7 MB |
| `docs/experiments` | 2 | 5.5 MB |
| `assets/ink_eval` | 96 | 4.7 MB |
| `assets/p125_20261030`（P125 六段） | 216 | 4.1 MB |
| `assets/abc_20261031`（A/B/C 链） | 144 | 2.4 MB |
| 其余（style_matrix、style_focus、u100_pretrain、reference_top10、f_20261101 等） | — | 剩余 |

## 命名约定（便于对照）

- `eval_samples_ctrl/step{step:07d}/{eval200fix|seen|strict}/g{i}.png` — 生成样本，`i` 与面板 CSV 行序一致。
- 同目录 `gt{i}.png` — 对应真迹。
- `assets/p125_20261030/.../posters/` — 每次 in-queue 评测生成的拼图。

## 未拉回的部分（有意）

- 完整面板（每步 100–200 张）：体积是这里的两倍以上，且可用 `tools/p125/eval_cpu.py`
  在任何机器上按同一协议重算，重算比存图更可信。
- 权重（`.pt`）：不属于图像归档，见 [ckpts.md](../../ckpts.md)。

## 复现抓取

```bash
# 远端暂存（不修改源目录）
scp scratch/_stage_eval_imgs.sh 4090:/tmp/ && ssh 4090 'bash /tmp/_stage_eval_imgs.sh'
ssh 4090 'cd /tmp && tar czf evalimgs.tar.gz evalimgs'
scp 4090:/tmp/evalimgs.tar.gz scratch/
tar xzf scratch/evalimgs.tar.gz -C docs/1031/eval_imgs --strip-components=1
```
