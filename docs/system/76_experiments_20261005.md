# 76 · 全量实验结果整理与结论（2026-10-05）

> 本文把三个结果目录里**所有**实验抓成结构化数据，然后在**同协议内**做单变量对照，给出结论。
> 旧的实验编年史 `docs/04_experiments/00_exhaustive_experiment_chronicle.md`（144KB）与其同目录文档已归档到
> `docs/archive/20261005_slim/`；system/ 下 10~75 号历史文档同样已归档（见 §9）。

## 0. 本次产物（可复核）

| 文件 | 内容 |
|---|---|
| `docs/experiments/all_runs_20261005.csv` | **每个 run 一行**：条件构成（skel/字表/书家表/注入层/损失权重…）+ 曲线关键点 + best（499 run × 63 列） |
| `docs/experiments/all_runs_curves_20261005.csv` | 长表 `(run_id, step, ssim, mse)`，870 个评测点，供画图 |
| `_sync_work/build_experiments_csv.py` | 生成器（扫描 `exp-std/`、`assets/results/`、`_archive/`） |
| `_sync_work/analyze_csv.py` | 分析器 → `_sync_work/tables_20261005.md`（按协议 × 条件分组表） |

扫描结果：**499 个 run 目录**（`exp-std/` 各 `runs*` + `assets/results/` 98 个实验 + `_archive/` 两个批次），
其中 **71 个有可用评测曲线**；其余为 smoke / 中断 / 未开评测的 run（CSV 里 `n_eval=0` 可筛掉）。

## 1. ⚠ 先说口径：四套评测协议**不可混比**

同一个 SSIM 在不同评测集上能差 0.3，跨协议比较必然得出错误结论。CSV 的 `eval_protocol` 列已标注：

| 协议（评测集） | 使用者 | SSIM 量级 |
|---|---|---|
| **`eval200fix+seen`（当前口径）** | v46 / v50-A / v52 / v53 / v54 | 0.51 ~ 0.59 |
| `eval200.csv` | v46-p0.2/0.4/0.5 若干 | 0.52 ~ 0.535 |
| `eval_seen_v10.csv` | v10b-stdskel-fame3 系列、v11_* | 0.47 ~ 0.77 |
| `fame_strict_clean_v8.csv` | v8a / v9a | 0.44 ~ 0.52 |
| 内嵌 json 老口径（seen/strict） | v10a / v10b pretrain | 0.78 ~ 0.85 |

**本文所有结论只在同一协议内比较。**

## 2. 结论一：内容表是主要贡献者，骨架不是

当前协议 `eval200fix+seen` 内（同数据、同评测集、同启动器）：

| 内容条件 | 实验 | best SSIM | @step |
|---|---|---|---|
| 骨架(std) + 书家表，**无内容表** | v50-A（xattn×12） | 0.5553 | 55k |
| 骨架(std) + 书家表，**无内容表** | v46（adaLN4）p1.0 | 0.5516 | 35k |
| 内容表 + 书家表，**无骨架** | v53（三表） | **0.5875** | 100k |
| 内容表 + 书家表，**无骨架** | v54（最小表 + 冻结） | **0.5875** | 95k |
| DINO 字表 + 书家表，**无骨架** | v52 | 0.5827 | 100k |

**+0.027 ~ +0.032**：把骨架换成内容表，收益远大于骨架侧任何调参
（v46 各 p 值、v50-A 架构替换都挤在 0.535~0.555）。

## 3. 结论二：有内容表后，骨架冗余（严格单变量）

| | v53 | v54 |
|---|---|---|
| 条件 | 三表 + **std 骨架输入** | 表条件、**无骨架** |
| 5k | 0.5106 | 0.5125 |
| 50k | 0.5736 | 0.5613 |
| 65k | 0.5801 | 0.5689 |
| 95k / 100k | 0.5871 / **0.5875** | **0.5875** |

差异只在**前中段 ≤0.012**，终点完全相同 → 骨架输入在表条件之上**不提供终点收益**。
（若目标是"结构可控生成"，骨架仍有意义；但作为**字形内容学习**的手段，它已被表取代。）

## 4. 结论三：「只给 skel 不给表」并没有更难学会（老协议单变量配对）

同代、同数据、同配方，只差字表：

| 实验 | 条件 | best SSIM | @step |
|---|---|---|---|
| v10a-skel-cond-pretrain | 1px 骨架 + **字表 ON** | 0.8476 | 127k |
| v10b-skel-only-pretrain | 1px 骨架 + **字表 OFF** | 0.8404 | **85k** |

**差 −0.0072，且去掉字表的版本更早到达高点**（85k vs 127k）。
即：**骨架条件本身已足够支撑字形内容学习，去掉内容表不会显著变难**
（与 `docs/system/36_v10b_no_char_follow.md` 记录的"中段 −0.005"一致；该文档已归档）。

## 5. 结论四：**真**「裸骨架」路线（无任何表）才是慢且低上限的那条

`eval_seen_v10` 协议下、std 骨架 + 无任何表（`inj12` / `inj4`）共 **34 个 run**：

| 实验 | best SSIM | @step | 备注 |
|---|---|---|---|
| v11_struct-loss | 0.7685 | **470k** | 结构损失加持，跑满 500k |
| v10b-stdskel-fame3-c41x | 0.7642 | 172k | |
| v10b-stdskel-fame3-c41x（另一次） | 0.7638 | 212k | |
| v10b-stdskel-fame3-sp2 / v11-* inj4 组 | 0.47 ~ 0.48 | 17~45k | 同协议内多数 run 长期 < 0.55 |

**特征：要 170k~470k 步才摸到高分区，且 run 间方差极大。**
与结论一/三合起来自洽：**内容表是"信息源"，骨架只是"结构提示"**。
把信息源拿掉（既不给写好的字、也不给表），模型只能反复从数据硬啃字形 → **又慢又不稳**。

所以：**你的方向是对的 —— 去掉表确实更难，但难在"收敛速度与方差"，不是"学不会"**；
且"只有骨架 vs 骨架+表"的差距（−0.007）**远小于**"表 vs 骨架"的优势（+0.03）。

## 6. 结论五：要让 skel「更难学会」，正确形态已有实现

仓库里已有「**骨架作为预测目标**（而非输入条件）」的完整机制，只是 v54 没开：

| 组件 | 位置 | 作用 |
|---|---|---|
| `skel_head` | `src/train/dit.py` | 从主干特征并行解出 1 通道 latent 骨架（**预测**） |
| `w_latent_skel` + `LatentSkelStructureLoss` | `src/train/train.py`（~1485-1520） | 在**训练好的 probe** latent 空间算结构损失 |
| `inst_skel_shards_dir` | 数据侧 | GT 图 → skeletonize → 膨胀 → VAE encode 的**实例骨架**，与条件 `g` 完全解耦 |
| 硬拦截 | `train.py` 同处 | target 指向条件 `g` 本身 → **拒绝启动**（防"预测输入"的假学习） |

即：**不给骨架输入，但要求模型产出骨架** —— 这才是"更难学会"且有意义的形式。
v54 两条都关（`skel_as_glyph_cond=False`、`w_latent_skel` 缺省 0），是纯"表条件"基线。

## 7. 运行中 / 未完成

| 实验 | 状态 |
|---|---|
| **v54 minimal-tables-S-noskel-freeze** | 运行中（S + 冻结最小表 + 无骨架，150k，batch 384）；截至 2026-10-05 07:00 约 100k/150k，best 0.5875@95k |
| v53 triple-tables-noskel | 已完成 100k，best 0.5875 |
| v52 noskel-callig-char | 已完成 100k（+续 150k），best 0.5827 |
| v50-A / v51-B（AB 臂） | v50-A 停在 58k；v51-B 未启动 |
| 字表 SupCon（`tools/pretrain_char_supcon.py`） | 已完成（v2 跨视图 InfoNCE，对齐 0.60） |

## 8. 复现

```bash
# 1) 重建全量 CSV（在数据机上，数据只存在远程）
python _sync_work/build_experiments_csv.py
python _sync_work/analyze_csv.py

# 2) 只看当前协议、按 best 排序
python - <<'PY'
import csv
rows = [r for r in csv.DictReader(open("docs/experiments/all_runs_20261005.csv"))
        if r["eval_protocol"] == "eval200fix+seen" and r["n_eval"] != "0"]
for r in sorted(rows, key=lambda r: -float(r["best_ssim"]))[:10]:
    print(f'{float(r["best_ssim"]):.4f} @{int(r["best_step"])//1000}k  {r["experiment_name"]}')
PY
```

## 9. 文档归档说明（2026-10-05）

为避免"文档比代码还庞杂"，本次把历史文档统一移入 `docs/archive/20261005_slim/`：

| 原位置 | 新位置 |
|---|---|
| `docs/system/{10..21,32..36,55,74,75}_*.md`（21 篇） | `docs/archive/20261005_slim/system/` |
| `docs/04_experiments/*`（含 144KB 编年史 + leaderboard） | `docs/archive/20261005_slim/04_experiments/` |
| `docs/01_architecture/`、`docs/02_dataset/`、`docs/03_training/` | `docs/archive/20261005_slim/` 同名子目录 |
| `docs/922/97_dual_channel_analysis.md` | `docs/archive/phase_922/` |
| `docs/{HANDOVER_2026-08-15,HISTORY_REVIEW_2026-09-22,STATUS_2026-09-30}.md` | `docs/archive/20261005_slim/` |

**`docs/` 现在只保留权威集**：`docs/system/00-09` + 本文(76) + `docs/system/README.md`、
`docs/experiments/`（本次 CSV 与近期报告）、`docs/BUG_dynamo_recompile_oom_2026-10-04.md`、`docs/archive/`。
索引见 `docs/README.md`。
