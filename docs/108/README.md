# 108 · 双机资产总账与同步对账（2026-10-09）

> **本轮做什么**：把"本地开发机 ↔ 4090 训练机"的**文件、数据、代码**做一次全量对账，补齐 10-09 那份资产整理记录的遗留项，并落成可复核的结构化产物。
> **本轮不做什么**：不占 GPU、不起训练、不删数据（只做**可回滚隔离**）。在跑的 `v72` 全程未受影响。
> **上游文档**：[`docs/system/10_assets_cleanup_20261009.md`](../system/10_assets_cleanup_20261009.md)（同日的资产整理记录）、[`docs/107/`](../107/README.md)（现状规约与实验计划）。

---

## 0. 一页结论（TL;DR）

1. **发现一个高优先级风险：训练机代码落后本地 git 约 3 天。**
   远程 4090 上 `sde_gamma`（SiT SDE 采样器）出现 **0 次**、`train_vae_encoder`/`calli_vae_path`（REPA-E）**0 次**、`calli_decode_noise`（VAE 约定开关）**0 次**；而 `use_c2ot`（10-06 C2OT）两侧一致。
   即远程同步停在 **~10-06**，10-08/10-09 的 **SDE 采样器、REPA-E、VAE 解码约定**三项改动**从未落到训练机**。
   → 后果：**远程评测口径 ≠ 仓库默认口径**（仓库默认已是 `heun + 50 + cfg 0.8 + γ 0.5`，远程是 `cfg 1.0 + 纯 ODE`）。v72 的曲线与 README/107 里引用的默认协议**不同源**。
2. **代码分叉只有 14 处（不是 95 处）**：95 个内容差异里 **81 个是纯换行符（CRLF/LF）**，算术可验证（如 `8964 + 207 CRLF = 9171`）。真实分叉 14 处详见 [02](02_sync_reconciliation.md)。
3. **远程已有 30 项死件移入 `_quarantine_20261009/`（43.01 GiB，可逐行回滚）**，含 26.59 GiB 的 `.git` 孤儿临时包。原位置校验已清空，v72 训练无感。
4. **指标台账补齐**：全量 `eval_auto_*.json` → **616 个评测点 / 36 个 run**（含在跑的 v72），产出 `points_all.csv` / `runs_all.csv` / `aligned_curve.csv`，与 `docs/experiments/mainline_20261009.csv` 互为校验。
5. **对齐曲线给出当前最强信号**：**v72（noaug + 新 Calli-VAE）只用 50k 步（0.5999）就超过了 v54/v66 跑满 150k（0.5960/0.5955）**；65k 时 0.6034，而对标的 v68 同步数只有 0.5833（**+0.0201**）。
6. 远程仍有 **547 GiB 属"需人工决策"**（历史归档 355G + git 松散对象 49G + 可重建副本 33G 等），**不是垃圾**，本轮一律未动。

---

## 1. 双机角色与"真相源"（必须先钉死，否则一切整理都无意义）

| | 本地 `G:\GitHub\DiT` | 远程 4090 `/root/Workspace/xy/DiT` | 48 dserver |
|---|---|---|---|
| 角色 | **代码 + 文档真相源** | **唯一 GPU 训练/数据机**（968G 空闲） | 次要算力，**网络故障中**（丢包 50%，scp ~1.4KB/s） |
| git origin | `KiritoFD/DiT`（项目本体） | `facebookresearch/DiT`（**上游**） | — |
| HEAD | `f80c609`（10-09 14:26） | `ac2c25e`（**10-01**） | — |
| 提交数 | **531** | 28 | — |
| 有效性 | 真实版本历史 | **git 无意义**：2086 untracked，工作区由 scp/rsync 覆盖 | — |

> ⚠ **由此产生的一个断链**：`logs/v72_launch_manifest.json` 里记录 `git_commit: ac2c25e`，那是**上游 DiT 的提交**，不是本项目代码状态。
> 溯源清单里的这个字段目前**自证无效**；应改为记录"本地 commit hash + 关键源文件 md5"（见 §5 行动项 A3）。

---

## 2. 远程（4090）文件总账

- 总量：**1.00 TB / 7,438,846 文件**（PNG 6,955,559、npy 437,017、py 10,382、json 10,152）。
- 磁盘：**968 GB 空闲**（无空间压力 → 本轮以"归位与台账"为主，不以删除为主）。

### 2.1 三大巨头与体积分布

| 目录 | 体积 | 说明 |
|---|---|---|
| `archive_experiments/` | **356 G** | s1–s32 世代历史产物（261G）+ 失败归档（59G）+ 历史快照（35G）；均**唯一副本** |
| `assets/` | **276 G** | `runs/` 231G（run 家族）+ 表/探针/日志若干 |
| `data/` | **213 G** | `dino_cache/` 139G + 原始/中间数据 |
| `exp-std/` | 49 G | latent shards（5.5G，**活跃**）+ 历史 runs（AB/smoke/purestd…） |
| `_models/` | 33 G | Qwen2.5-VL-7B / Qwen3-VL-4B（在用）；3B（零引用） |
| `dataset/` `baseline/` | 各 24 G | 旧按类目切分集 / 三方基线工作区 |
| `_backups/` `_rescued/` | 15 G / 6.4 G | 旧备份与其抢险产物（`purge_old_backup.sh` 的**强制前置**，高风险，未动） |

### 2.2 run 家族化结构（10-09 已完成，本轮只补台账）

```
assets/runs/                       # assets/results 是指向 runs 的符号链接（10-09 05:48 建）
  v10b/(11G) v11_pretrain/(33G) v12/(5.1G) v13/(8.0G) v14/(2.3G) v15/(13G) v16/(153M)
  v17/(18G)  v18_19/(1.6G) v54_67_tables/(28G) v68_family/(66G) v69_70/(16G)
  moyi/(4.0G) ablation_modern/(15G) probes_smoke/(877M)
  v72_callivae_kl1e6_s12500_frozen_sp_c2ot/(12G, **在跑**)
  _archive/(60G, → archive_experiments/results_archive_failed)  _loose_ckpts/  MANIFEST.md
```
- 结论：`assets/runs/_archive` 与 `archive_experiments/results_archive_failed` **不是重复**，前者是后者的符号链接（`readlink -f` + 目录列表 `diff` exit 0）。同理 `assets/results → runs`、根 `_archive → _archive_historical`。

### 2.3 本轮隔离区（新增，43.01 GiB，可回滚）

- 位置：`_quarantine_20261009/`（**保留原相对路径**），清单 [`MANIFEST.csv`](../../assets/ledger/quarantine_manifest_20261009.csv)。
- 内容 30 项，最大三项：`.git/objects/pack/tmp_pack_uGPgSj` **26.59 G**（被中断 repack 的孤儿包，无 `.idx/.rev`，正规 pack 仅 6.6 MB）、`exp-std/runs_smoke` **14.18 G**（59 个 smoke run，src/tools/configs 零引用）、`exp-std/_attempt1` **1.13 G**。
- 其余：`assets/dino_feat_raw.BROKEN.npz`、`char_script_supcon_table.npy.v1_random.bak`、`latent_missing`（已归档 tar）、`_trash`、6×`flip_backup*`、`retro_posters(_latest)`、`gen_curves_S`、`gen_eval_*_imgs`、`compose_test_imgs`、`example_100`（已归档 tgz）、4 个**上游残留**（`run_DiT.ipynb`/`environment.yml`/`CONTRIBUTING.md`/`CODE_OF_CONDUCT.md`）、`train_dashboard.html`（与 `tools/` 副本 md5 相同）、根 `results/`（ckpt 为空）、零字节文件 `" 2"`、空目录 `exp/`、以及一个**目录名以回车 0x0D 结尾**的污染目录。
- **注意**：`mv` 在同一文件系统内是改名，**不释放空间**；空间在你最终确认删除时才回收（`rm -rf _quarantine_20261009`）。
- 回滚：按 MANIFEST 逐行 `mv <quarantine_path> <original_path>`。

### 2.4 明确**未动**的高价值/高风险项（547 GiB）

| 项 | 体积 | 为什么不动 |
|---|---|---|
| `archive_experiments/results_legacy_s1_to_s32` | 261 G | 唯一副本，含 14 个 ≥5GB 的 XL ckpt |
| `.git/objects` 松散对象 | 49 G | 953,330 个松散对象；正确处置是 `git gc`/历史重写，不是删 |
| `results_archive_failed` / `_archive_historical` | 59 G / 35 G | 被 `exp_milestones/*/meta.json` 引用的 **milestone ckpt 唯一副本** |
| `data/dino_cache/v11_sym_full/feats.f32` | 31 G | `feats.f16` 的精确 2× 升精度副本；loader 优先 f32、缺失回落 f16；仅 1 个 9 月配置引用 |
| `_backups/` + `_rescued/` | 21 G | `purge_old_backup.sh` 删除 622GB 旧备份的**强制前置校验** |
| `baseline/`(23.6G) `dataset/`(23.5G) `_models/3B`(7.1G) | 54 G | 三方基线/旧数据集/未引用模型；需人工确认后再谈 |

---

## 3. 本地 ↔ 远程 源码对账（★ 本轮最重要的产出）

方法：对远程 `src/ tools/ tests/ scripts/ _sync_work/ docs/` + 根文件的 `*.py|*.sh|*.json|*.md|*.yaml|*.txt` 计算 md5，与本地同名文件逐一比对；差异再用"LF 归一化 md5"二次判定。

| 结果 | 数量 |
|---|---|
| 两侧一致 | **2,371** |
| 内容不同 → 其中**纯换行符差异** | **81** |
| 内容不同 → **真实分叉** | **14** |
| 仅远程有（本地缺失） | **203** |
| 仅本地有（远程缺失） | **3,896** |

### 3.1 真实分叉 14 处

| 文件 | 远程 | 本地 | 性质 |
|---|---|---|---|
| `src/train/train.py` | 171,728 | **176,545** | 12 处 hunk（REPA-E/VAE 相关段全缺） |
| `src/loss/flow_matching.py` | 26,661 | **27,673** | **整个 SiT SDE 采样器缺失** |
| `src/train/cli.py` | 92,708 | **95,097** | 缺 `--sde-gamma`、`--train-vae-encoder`、`--calli-vae-path` |
| `src/eval/cli.py` | 78,565 | **79,784** | 缺同批参数 |
| `src/eval/in_mem_eval.py` | 62,706 | **63,464** | 缺 calli 解码约定分支 |
| `tools/audit_minimal_best.py` | **3,519** | 1,920 | ⚠ **远程更大 → 远程有未回传改动** |
| `_sync_work/peek_A.sh` | **1,819** | 1,719 | ⚠ 同上 |
| `README.md` / `docs/README.md` / `docs/system/00_README.md` | 旧 | 新 | 文档，本地为 10-09 版 |
| `src/train/configs/_probe_inject12.json`、`tests/test_arch_v2.py`、`_sync_work/v50_A_*.json`、`v51_B_*.json` | — | — | 细粒度配置/测试差异 |

**关键标识检索（决定性证据）**：

| 标识 | 本地 | 远程 |
|---|---|---|
| `sde_gamma` | 13 | **0** |
| `sde-gamma` | 2 | **0** |
| `train_vae_encoder` | 2 | **0** |
| `calli_vae_path` | 1 | **0** |
| `calli_decode_noise` | 7 | **0** |
| `use_c2ot` | 11 | 11 ✅ |
| `c2ot_mode` | 10 | 10 ✅ |

### 3.2 仅远程有（203）——**未回传的资产，需抢救**

- `src/train/configs/ab_mod_{ctrl,ln,mlp,qk,rope}.json`（5 个）——**README §5 消融表的配置，本地仓库缺失**；
- `src/train/configs/{v68_pix,v68_dino_raw,v68_randtab,v64_skel_callig_script_4ch,v65_tables_condinj2468}.json`；
- `tools/{build_top10_sym_aug_5x,encode_top10_aug_5x,dino_expand_top10_aug_5x,probe_ceiling_exhaustive}.py`；
- `src/utils/std_glyph_latent_v2/{coverage_report,manifest}.json`；
- `_sync_work/` 若干仅远程脚本 + `_AB_patch`/`_patch*_bak` 打补丁备份（18+5+4+4+2）；
- **`docs/system/10…75` 共 20 篇旧编号文档 + `docs/01_architecture/`、`docs/922/`、`docs/HANDOVER_*`** —— 本地已在 10-05 归档，**远程仍是旧布局**（远程 docs 内容过期）；
- `scripts/legacy_runners/` 71 个。

### 3.3 仅本地有（3,896，前几位）

`docs/archive` 2,084（10-05 归档的历史文档，远程未清）、`tools/legacy` 197、`tools/poster_archive` 182、`_sync_work/baseline_src` 161、`tools/remote_pulled` 92、`scripts/ops` 63、`docs/106moyi` 62、`src/train` 52 …

> **口径提醒**：`src/train/configs/` 里那批 `v13_fs_*/v15_fs_*/v16_fs_*` 中文名配置在两侧都存在，但因文件名编码（GBK vs UTF-8）被判为"仅远程有"——**属误报**，不构成缺口。

---

## 4. 指标台账（本轮补齐）

- 产物：[`assets/ledger/points_all.csv`](../../assets/ledger/points_all.csv)（616 行 × 33 列，逐评测点）、[`runs_all.csv`](../../assets/ledger/runs_all.csv)（36 run × 摘要+配置）、[`aligned_curve.csv`](../../assets/ledger/aligned_curve.csv)（step 对齐矩阵）。
- 与既有 [`docs/experiments/mainline_20261009.csv`](../experiments/mainline_20261009.csv)（17 行 187 协议主表）互为校验：v68 旗舰 **0.6149@200k** 两侧一致 ✅。

### 4.1 主线 step 对齐（`eval200fix`, N=187, 远程 in-mem eval）

| step | v68 (aug+SD-VAE) | **v72 (noaug+Calli-VAE)** | Δ | v66 (150k 线) | v70 (stdskel) |
|---|---|---|---|---|---|
| 5k | 0.5225 | 0.5247 | +0.0022 | 0.5140 | 0.5341 |
| 10k | 0.5300 | **0.5414** | +0.0114 | 0.5341 | 0.5349 |
| 20k | 0.5429 | **0.5650** | +0.0221 | 0.5397 | 0.5429 |
| 30k | 0.5495 | **0.5844** | +0.0349 | 0.5504 | 0.5457 |
| 40k | 0.5621 | **0.5945** | +0.0324 | 0.5602 | 0.5466 |
| 50k | 0.5701 | **0.5999** | +0.0298 | 0.5661 | 0.5496 |
| 60k | 0.5791 | **0.6032** | +0.0241 | 0.5729 | 0.5535 |
| 65k | 0.5833 | **0.6034** | +0.0201 | 0.5767 | 0.5535 |

**读法与限定**：
1. **v72@50k = 0.5999 > v54@150k = 0.5960 与 v66@150k = 0.5955** —— 在字表 S/2 这条线上，换新 Calli-VAE 潜空间后用 **1/3 的步数**达到了旧线的终点水平。
2. **但这不是干净的单变量对照**：v72 = noaug 语料 + 新 Calli-VAE；v68 = 3x 增强 + SD-VAE + C2OT。两者的正确对照臂是 48 上的 `noaug_c2ot`（SD latent + noaug，见 [决策文档 §5](../experiments/2026-10-09-frozen-vae-vs-repa-e-decision.md)），**该臂数据不在 4090**，故此处只能说"v72 显著更强"，**不能把全部增益归因给 VAE**。
3. 口径提醒：这批数字是**远程 in-mem eval（cfg=1.0，无 SDE）**；仓库默认口径已改为 `cfg 0.8 + γ 0.5`。**跨口径不可直接比**（见 §0.1）。

### 4.2 消融台账（README §5 的机器可读版，20k 同频）

| 臂 | 20k seen SSIM | Δ vs 基准 |
|---|---|---|
| 基准（v68 配置 @20k） | 0.5429 | — |
| 去 RoPE | 0.5408 | −0.0021 |
| 去 RMSNorm | 0.5405 | −0.0023 |
| 去 QK-Norm | 0.5375 | −0.0054 |
| 去 SwiGLU | 0.5366 | −0.0063 |
| 表内容：40万切片 | 0.5453 | +0.0024 |
| 表内容：像素表 | 0.5365 | −0.0064 |
| 表内容：完全随机 | 0.5323 | −0.0106 |

→ 与 README §5 结论一致：**四个现代组件全部正贡献；表内容"学过 vs 随机"是最大杠杆（~0.011）**。

---

## 5. 行动项（按优先级，A 类为风险修复）

| # | 行动 | 依据 | 风险 |
|---|---|---|---|
| **A1** | **决定是否把 14 处真实分叉同步到 4090**：12 处本地较新（含 SDE/REPA-E/VAE 约定）应 push；2 处（`tools/audit_minimal_best.py`、`_sync_work/peek_A.sh`）应 pull 回本地。**在跑的 v72 已加载代码，改文件不影响当前进程，但会影响下一次启动** | §3.1 | 中（需在 v72 完训或确认可重启时做） |
| **A2** | 明确口径：远程评测若继续用 `cfg 1.0 + 纯 ODE`，则所有与 README/107 的横向比较都要标注；反之应同步 SDE 后统一为 `cfg 0.8 + γ 0.5` | §0.1 | 中（影响结论正确性） |
| **A3** | 修 `launch_v72.sh` 的溯源字段：`git_commit` 改记**本地 commit + 关键文件 md5**（当前记的是上游 commit，无效） | §1 | 低 |
| **A4** | 把 §3.2 的 203 个"仅远程有"资产回传本地（尤其 `ab_mod_*.json` 5 个 + `v68_{pix,dino_raw,randtab}.json`，它们是 README §5 与 107 引用的配置） | §3.2 | 低 |
| **A5** | 远程 `docs/` 过期（仍是 10-05 前布局）：下次同步时用本地 `docs/` 覆盖，避免"在训练机上读到旧结论" | §3.2 | 低 |
| **B1** | 547 GiB "需人工决策"清单逐项裁决（历史归档/git gc/可重建副本） | [01](01_remote_inventory.md) | — |
| **B2** | 48 的 `_pull_archive/` 195M 归档包（网络恢复后拉回）；`tier4_l` 补评测 | system/10 遗留 | — |
| **B3** | 建立正式同步契约（当前只有临时 scp）：`tools/remote_sync/` 里是 8 月的旧文件；建议新增 `tools/remote_sync/sync.sh` + 排除清单 + 上表这种对账脚本 | §3 | 低 |

---

## 6. 产物索引

| 文件 | 内容 |
|---|---|
| [01_remote_inventory.md](01_remote_inventory.md) | 远程目录总账、家族化结构、隔离清单、547G 待裁清单 |
| [02_sync_reconciliation.md](02_sync_reconciliation.md) | 完整对账表：一致/EOL/真实分叉/单向文件 |
| [03_metrics_ledger.md](03_metrics_ledger.md) | 台账字段说明、对齐曲线、消融台账、查询示例 |
| [appendix_quarantine_audit.md](appendix_quarantine_audit.md) | 隔离审计原始报告（含引用检查方法与局限） |
| `assets/ledger/points_all.csv` | 616 个评测点（33 列） |
| `assets/ledger/runs_all.csv` | 36 个 run 摘要 + 配置快照 |
| `assets/ledger/aligned_curve.csv` | step 对齐矩阵 |
| `assets/ledger/quarantine_manifest_20261009.csv` | 30 项隔离清单（回滚用） |
