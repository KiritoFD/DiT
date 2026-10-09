# 108-04 · 本地仓库物理盘点（`G:\GitHub\DiT`）

> 采集时间 2026-10-09 19:52–20:13。**只读**：未修改/删除/移动任何既有文件；临时脚本与中间结果仅在 `scratch/_audit/`（已被 `.gitignore` L84 覆盖）。
> 原始产物：`scratch/_audit/{dirs,exts,exts_nogit,exts_tracked,docs_dirs,docs_md,big50,risky}.csv`、`{zero,emptydirs}.txt`、`lstree.txt`、`reachable.txt`、`ignored.txt`。

## 1. 三条最重要的修正

1. **本地不是"代码+文档+小台账"，而是 75 GB 的数据仓**：全树 **1,806,787 文件 / 76,977 MB**；剔除 `.git` 后仍有 **516,696 文件 / 48,365 MB**。
2. **`.git` 是本机最大单体（28,760 MB / 1,290,129 文件），其中 26.72 GiB 是 loose 对象**；而全历史可达对象只有 **12,589 个**（531 commits）。最大 9 个 loose 对象（0.5–1.5 GB）经 `git rev-list --objects --all` 逐一核对**全部不可达** → **约 26 GB 纯垃圾**。
3. **工作树在 19:52 是完全干净的**（`git diff-index --quiet HEAD` → exit 0）：此前 README/losses.py/latent_dataset.py 的改动已被 commit **`f80c609`（10-09 14:26）** 收走。盘点期间唯一新增 untracked 是 `assets/ledger/`（本轮的台账 CSV）。
   > ⚠ **说明还有别的 agent/进程在同一仓库写文件** —— 任何清理动作前后都要重新确认 `git status`。

## 2. 顶层目录（按体积降序，节选）

| 名称 | 文件数 | 体积 | Tracked | 作用 |
|---|---|---|---|---|
| **.git** | 1,290,129 | **28,760 MB** | — | 对象库；26.72 GiB loose 垃圾，pack 仅 755 MB |
| `archive/` | 356,604 | **18,499 MB** | 0 | 本地垃圾场（`/archive/` 忽略）：`dataset.zip` 13.2 GB、`final/` 2.2 GB |
| `assets/` | 5,180 | 6,716 MB | 336 / 336 MB | 结果资产 + 台账 + 海报 |
| `_sync_work/` | 3,280 | 6,222 MB | **829 / 10.3 MB** | 从 4090 拉回的镜像；**目录被 ignore 却有 829 文件被强加跟踪** |
| `pretrained_models/` | 23 | 4,845 MB | 0 | 权重（DiT-XL 2.6 GB） |
| `_ot_scratch/` | 4,415 | 1,767 MB | 0 | OT scratch（`s32c_best_80000.pt` 1.15 GB） |
| **`docs/`** | 3,714 | 1,689 MB | 3,276 / 409 MB | 见 §5；`docs/archive/` 独占 1,510 MB |
| `data/` | 46,361 | 1,664 MB | 0 | 本地数据缓存 |
| `_zs/` | 4,979 | 1,294 MB | 0 | 旧产物（`main_s19.pt` 602 MB） |
| `tools/` | 23,042 | 1,171 MB | 1,566 / 404 MB | 工具+产物；30 条规则才盖住 481 项 |
| `_diag/` | 336 | 600 MB | 0 | 诊断转储 |
| `exp/` | 88 | 555 MB | 0 | 旧产物 |
| `final_imgs_256/` | 14,872 | 528 MB | 0 | 图像产出 |
| `legacy/` | 218 | 489 MB | 200 / 100 MB | `legacy/data/train.csv` 39.3 MB **已入库** |
| `final_skel_latents_fame/` | 20 | 339 MB | 0 | latent 产物 |
| **`src/`** | **39,170** | 313 MB | **430 / 3.45 MB** | 38,635 文件是 `src/utils/std_glyph_latent_v2/`（306 MB，靠**名字**规则忽略） |
| `results/` `_review/` `eval/` `scripts/` | 6,522 / 3,409 / 2,545 / 397 | 297 / 178 / 100 / 90 MB | 0 / 215 / 0 / 396 | — |
| `_research/` | 14 | 23 MB | **14 / 23 MB** | **无目录规则，含 13.2 MB `.npy` 入库** |
| `exp-std/` `logs/` | 11 / 279 | 17 / 6 MB | 10 / **1** | `logs/` 仅 1 个 `.log` 被 `!` 白名单强留 |
| `experiments/` `final_skeleton_d3/` | **0** | 0 | 0 | **空目录** |
| 根散落文件 | 49 | 627 MB | 26 / 0.09 MB | 见 §4 |

**入库内容真实构成**（7,441 文件 / 1,343.5 MB）：**.png 558 MB + .csv 433 MB = 74%**，真正的代码（`.py/.sh/.js/.md/.json`）只有 **~35 MB**。

## 3. 文件类型 Top（工作树，剔除 `.git`）

`.pt` 30 个/16.0 GB · `.zip` 4/13.7 GB · `.png` 443,112/7.5 GB · `.npz` 40/2.5 GB · `.safetensors` 7/1.8 GB · `.pth` 6/930 MB · `.gz` 20/918 MB · `.tgz` 25/793 MB · `.csv` 638/**728 MB** · `.npy` 58,838/646 MB · `.json` 5,160/458 MB · `.txt` 755/298 MB · `.pdf` 7/180 MB。

全树含 `.git` 时，扩展名 **`(none)`（即 `.git/objects/??/*`）以 1,290,101 文件 / 28,054 MB 绝对第一** —— 这条本身就是"git 未整理"的直接证据。

## 4. 根目录散落文件（49 个 / 627 MB）

| 类别 | 内容 | 建议 |
|---|---|---|
| **A. 应归档（626 MB，占 99.8%）** | `chinese-calligraphy-dataset-with-calligrapher-221030.zip` 289 MB、`Chinese Calligraphy Styles by Calligraphers_datasets(1).zip` 176 MB、`dit_imgs_50k_20260918.7z` 123 MB、`eval_all_models.tar.gz` 38 MB | 移远端/外部存储 |
| **B. 应归档（26 个已跟踪脚本）** | `run_v21…v42` 整条启动器序列 + `run_pipeline_v36_to_union.sh` + `run_skelnet_standalone_v2.sh` + `probe_loss_ablation.sh` + `check_rows.py` | → `tools/launchers/`（根目录散落导致无法按版本检索：v30/v32/v33/v36 缺号，v24/v37 各有两变体） |
| **C. 应删除候选（19 个 / 0.82 MB，均被 ignore，不影响 git）** | `tmp_col_{0..5}_{出,禮,簿,靜,紹,陞}.png`（同时间戳一次性取字）、`tmp_strict_poster_first8_full.png` 630 KB、`test_shards_img_0.png`、`poster_sample0_row0..3`/`render_sample0_row0..2`/`v34_sample0_row0..2`（同 sample 不同管线，已过时）、`paseo.json`（**3 字节**，写入中断残骸） | 本轮已隔离（见 §7） |
| **D. 保留** | `README.md`、`LICENSE.txt`、`.gitignore` | — |

## 5. `docs/` 结构与重复

`docs/` 共 3,714 文件 / 1,689 MB，**md 251 个**。

| 子目录 | md | 体积 | 定性 |
|---|---|---|---|
| `docs/archive/` | 207 | **1,510 MB** | **归档**（`legacy_reports` 1,357 MB 是体积元凶；`system_notes` 94 md；`20261005_slim` 45 md） |
| `docs/106moyi/` | 1 | 141 MB | 现行（内嵌第三方仓库镜像 132 MB） |
| `docs/04_experiments/` | 3 | 33 MB | 现行（12 张海报 32.6 MB） |
| `docs/experiments/` | **11** | 2.0 MB | **现行（最新，10-09）** |
| `docs/107/` | 5 | 1.4 MB | 现行 |
| `docs/system/` | 14 | 0.6 MB | 现行 |
| `docs/105/` | 7 | 0.06 MB | 现行但轻量 |

### 确认的重复（MD5 实测）

| 关系 | 结论 |
|---|---|
| `docs/archive/system_notes/*` ↔ `docs/archive/20261005_slim/system/*` | 20 对同名中 **17 对逐字节相同**；3 对分叉（`15_progress`、`17_training_lineage`、**`74_fewshot_handover` 12.0 KB vs 27.5 KB —— slim 版被截断一半以上**） |
| `docs/system/*` ↔ `docs/archive/system_notes/*` | 12 对同名中 **6 对 MD5 相同**（`01_code_layout/02_diffusion/05_dataset/06_training/07_eval/08_experiments`），5 对微改（差 0.1–0.2 KB），1 对实质分叉（`10_*`：archive 是 diagnosis、现行是 assets_cleanup） |
| `legacy_reports/experiments/2026-08-15-factorized-3cond.md` ≡ `docs/experiments/` 同名 | 相同 |
| `legacy_reports/HISTORY_REVIEW_2026-09-22.md` ≡ `20261005_slim/` 同名；`HANDOVER_2026-08-15.md` ≡ 同名 | 相同 |

> **可压缩**：三处并存（`system_notes` + `20261005_slim/system` + `system`）可减为"现行 + 一份归档"。
> **⚠ 勿删**：`docs/04_experiments/00_exhaustive_experiment_chronicle.md` 现行版 **17.5 KB** vs 归档版 **141.3 KB（8 倍差）** —— **归档版是全文，含独有信息**。

## 6. `.gitignore` 缺口（198 行，实测）

| 目录 | 目录级规则 | 体积 | 缺口 |
|---|---|---|---|
| `assets/` | ❌ **无** | **6,716 MB** | **最大缺口**。实测未覆盖 `assets/structure_probe_cache/`(1,192 MB，只因文件恰为 `.npz`)、`registry/`、`triple_tables_best_minimal/`、`condition_probes/`(26.7 MB)、`backups/` … 且 **`.csv` 全仓无 ignore（638 文件/728 MB）**，`assets/train_full.csv`(14.4 MB) **已入库** |
| `docs/` | ❌ 无 | 1,689 MB | `docs/archive/**` 完全无规则 → 1,510 MB 全在 git 视野内（已跟踪 409 MB） |
| `tools/` | ❌ 无 | 1,171 MB | 用 **30 条**逐路径规则打补丁 |
| `src/` | ❌ 无 | 313 MB | 靠 `std_glyph_latent_v2/` **按名字**盖住 38,635 文件/306 MB，改名即涌入 |
| `_sync_work/` | ✅ L108 | 6,222 MB | **规则在却无效**：829 文件被 `add -f` 强加跟踪 |
| `logs/` | ❌ 无 | 6.1 MB | 只有 `*.log` + 一条 `!` 白名单；任何非 `.log` 落进即 untracked |
| `exp-std/` `legacy/` `_research/` | ❌ 无 | 17 / 489 / 23 MB | `legacy/data/train.csv`(39.3 MB)、`_research/.../std_kai_cls.npy`(13.2 MB) **已入库** |
| `.codeartsdoer/` `.arts/` `.freebuff/` `.commandcode/` `.workbuddy-ai/` | ❌ 无 | <1 MB | IDE/工具残留 |

**最小补丁建议**（本轮未执行）：
```gitignore
/assets/
!/assets/triple_tables_best_minimal/*.npy
/docs/archive/
/logs/
/exp-std/
/src/utils/std_glyph_latent_v2/
/.codeartsdoer/  /.arts/  /.freebuff/  /.commandcode/
```

## 7. 异常项

1. **`.git` loose 垃圾 26.72 GiB**（P0，见 §8）。
2. **`.git/lfs/objects` 638 MB 孤儿**：仓库根**无 `.gitattributes`**、`.git/config` **无 `filter.lfs.*`**、HEAD 内 grep `git-lfs.github.com` **无命中** → 无任何引用。
3. **真误提交的大文件仅 2 个 >50 MB**：`tools/experiment_progress/README.pdf` **79.0 MB**、`report.pdf` **60.8 MB**（后者无任何规则保护）。>20 MB 共 7 个（另有 `strict_struct.png` 42.1、`legacy/data/train.csv` 39.3、`strict_poster.png` 32.2、两张 eval_poster 27.7/21.5 MB）。
4. **空目录 1,016 个**（`assets/` 891）、**0 字节文件 1,164 个**（`assets/` 1,116 = 96%，是评测流水线的 `.done` 占位）。
5. **顶层空目录 2 个**：`experiments/`、`final_skeleton_d3/`。
6. **非 ASCII 路径 2,835 个**，其中 **`logs/` 279 个文件里 253 个是中文名** —— Windows↔Linux 之间 `rsync/scp/tar` 最易乱码。含空格/括号的仅 2 个（其一 `Chinese Calligraphy Styles …datasets(1).zip` 已专门写规则，说明踩过坑）。
7. **`archive/` 混入 2 个无关文件**：`塞壬唱片下载器.exe` 17.6 MB、`①群心情管家·助手 new.rar` 19.5 MB。
8. **截断产物** `paseo.json`（3 字节）。

## 8. 本地-远程职责划分建议（本地应"代码 + 文档 + 小台账"）

| 优先级 | 内容 | 体积证据 | 动作 |
|---|---|---|---|
| **P0** | `.git` loose 对象 | **26.72 GiB / 1,290,054 个**，可达对象仅 12,589；最大 9 个（`d60d3de…` 1,540.6 MB 等）**全部不可达** | `git gc --prune=now` → 回收 ≈26 GB（**本轮未执行，需你确认**） |
| **P0** | `.git/lfs/objects` | 638 MB 孤儿（无 `.gitattributes`/无 `filter.lfs`/HEAD 无引用） | 可删；若确要 LFS 需补配置 |
| P1 | `archive/` | **18,499 MB / 356,604 文件**，0 入库（`dataset.zip` 13.2 GB 等） | 本地留 0，只留 manifest 摘要 |
| P1 | `_sync_work/` 权重 | 目录 6,222 MB，仅 10.3 MB 入库；4 个 >500 MB 权重 | 权重归远端；**829 个强加跟踪文件需 `git rm --cached`** |
| P1 | `assets/` | 6,716 MB，入库仅 336 MB（`results/` 4,787 MB、`structure_probe_cache/` 1,192 MB） | 只留 summary 表 + 海报 |
| P1 | `pretrained_models/` | 4,845 MB（可重新下载） | 留 SHA 清单 |
| P2 | 其余 run 输出/缓存 | `_ot_scratch` 1,767 + `data` 1,664 + `_zs` 1,294 + `_diag` 600 + `exp` 555 + `final_imgs_256` 528 + `final_skel_latents_fame` 339 + `results` 297 + `eval` 100 MB ≈ **7.3 GB** | 全移远端 |
| P2 | 已入库但该瘦身 | 入库 1,343 MB 中 991 MB（74%）是 png+csv | 目标让 git 仓回到 ~50 MB 级 |

**一句话**：本地 75 GB 里，**代码 + 现行文档只有约 55 MB**，其余 99.9% 是 `.git` 垃圾、数据副本与 run 输出。

## 9. 本轮对本地做的实际动作（保守、可回滚）

见 [05_local_quarantine.md](05_local_quarantine.md)：把根目录 §4-C 的 19 个一次性图片/残骸移入 `_quarantine_20261009/`（本地同名隔离区），并留 MANIFEST。**未执行** `git gc`、未动 `.git/lfs`、未删任何跟踪文件。
