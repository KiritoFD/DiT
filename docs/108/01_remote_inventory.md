# 108-01 · 远程（4090）目录总账

> 采集时间：2026-10-09 19:49–20:15 CST。全部数字来自实跑 `du/find/stat/ls`，**未占用 GPU**（v72 训练全程存活）。
> 工作区：`/root/Workspace/xy/DiT`，磁盘 **968 GB 空闲**。

## 1. 总量与体积分布

| 项 | 值 |
|---|---|
| 总文件数 | **7,438,846** |
| 总体积 | **1004 GB**（`du` 去重后） |
| PNG | 6,955,559 |
| npy / json / py | 437,017 / 10,152 / 10,382 |
| `__pycache__` 目录 | 9 |

| 目录 | 体积 | 定性 |
|---|---|---|
| `archive_experiments/` | **356 G** | s1–s32 历史产物 261G + 失败归档 59G + 历史快照 35G |
| `assets/` | **276 G** | `runs/` 231G + 表/探针/日志 |
| `data/` | **213 G** | `dino_cache/` 139G + 原始/中间数据 |
| `exp-std/` | 49 G | latent shards 5.5G（**活跃**）+ 历史 runs |
| `_models/` | 33 G | Qwen2.5-VL-7B(16G) + Qwen3-VL-4B(8.3G) + ocr(2G) + 3B(7.1G，零引用) |
| `dataset/` `baseline/` | 各 24 G | 旧按类目切分集 / 三方基线（DG-Font 19G） |
| `_backups/` | 15 G | `dataset_20260923.tar.gz`（purge 脚本强制前置） |
| `_rescued/` | 6.4 G | 622GB 旧备份的抢险产物（同上） |
| `tools/` `experiments/` `_venv_qwenvl/` `archive/` `_sync_work/` `MCCD/` | 3.1G / 2.6G / 1.9G / 1.7G / 1.5G / 1.3G | 工具、Calli-VAE 训练区、Qwen-VL venv、旧归档、运维脚本、MCCD 数据 |
| `_quarantine_20261009/` | **44 G（新）** | 本轮隔离区（见 §3） |

## 2. run 家族化结构（10-09 已完成）

```
assets/runs/                          # assets/results -> runs  (符号链接, 10-09 05:48)
  v10b/(11G) v11_pretrain/(33G) v12/(5.1G) v13/(8.0G) v14/(2.3G) v15/(13G)
  v16/(153M) v17/(18G) v18_19/(1.6G) v54_67_tables/(28G) v68_family/(66G)
  v69_70/(16G) moyi/(4.0G) ablation_modern/(15G) probes_smoke/(877M)
  v72_callivae_kl1e6_s12500_frozen_sp_c2ot/(12G, 在跑)
  _archive/(60G -> archive_experiments/results_archive_failed) _loose_ckpts/ MANIFEST.md
```

**符号链接关系（已用 `readlink -f` + 目录 `diff` 验证，**不是重复**）**：

| 链接 | 目标 |
|---|---|
| `assets/results` | `assets/runs` |
| `assets/runs/_archive` | `archive_experiments/results_archive_failed` |
| 根 `_archive` | `archive_experiments/_archive_historical` |
| `archive_20260903` | `archive_experiments/archive_20260903` |
| `pretrained_models` | （空） |

**checkpoint 现状**：`assets/runs` 下 `.pt` 321 个、`.pt.done` 527 个（`.done` 是异步落盘完成标记）。
v68 基座另有硬链接备份 `/root/Workspace/BACKUP_v68_base/v68_aug_sp_c2ot/`（43 ckpt，零额外空间）。

## 3. 本轮隔离区 `_quarantine_20261009/`（43.01 GiB / 30 项）

保留原相对路径，回滚 = 按 MANIFEST 逐行 `mv`。

| 项 | 体积 | 判定证据 |
|---|---|---|
| `.git/objects/pack/tmp_pack_uGPgSj` | **26.59 G** | 无同名 `.idx/.rev`；正规 pack 仅 6.6 MB（08-12）→ 被中断 repack 的孤儿 |
| `exp-std/runs_smoke` | **14.18 G** | 59 个 smoke run；src/tools/scripts/configs **零命中** |
| `exp-std/_attempt1` | 1.13 G | 仅 `_sync_work` 两张表提及 |
| `assets/dino_feat_raw.BROKEN.npz` | 0.575 G | 与 `dino_feat_raw.npz` 字节数完全相同，名自带 BROKEN |
| `assets/char_script_supcon_table.npy.v1_random.bak` | 0.108 G | 同字节数 `.bak`，零引用 |
| `latent_missing` | 0.10 G | 已归档 `archive/tarballs/latent_missing.tar.gz` |
| `_trash` | 0.13 G | 已隔离的 predskel 脏目录 |
| `flip_backup{,_fame2..5,_manual}` ×6 | 0.14 G | 极性翻转前备份，仅注释提及 |
| `retro_posters{,_latest}` `gen_curves_S` `gen_eval_{C,S_old}_imgs` `compose_test_imgs` | 0.08 G | 全库 grep 零命中 |
| `example_100` | 0.007 G | 已归档 `archive/tarballs/example_100.tgz` |
| `train_dashboard.html` | 31 KB | 与 `tools/` 副本 **md5 相同**；12 处引用均按 `HERE` 解析到 `tools/` |
| `run_DiT.ipynb` `environment.yml` `CONTRIBUTING.md` `CODE_OF_CONDUCT.md` | ~10 KB | **上游 facebookresearch/DiT 残留**，零引用 |
| 根 `results/` | 33 KB | 2 个被弃 run，`checkpoints/` 为空 |
| `" 2"`（文件名=空格+2） | 0 B | 0 字节 |
| `exp/`、`exp-std/signal_arena` | 空目录 | — |
| `assets/d1_logs<CR>` | 空目录 | 目录名以 **0x0D 回车**结尾（与正常 `assets/d1_logs` 并存） |

> **`mv` 同盘不释放空间**——回收发生在 `rm -rf _quarantine_20261009` 时。
> 已自检：原位置全部消失，`logs/v72_train_*.log` 继续推进（20:14 已达 step 70,400，3.6 steps/s）。

## 4. 明确**未动**的 547 GiB（"review ≠ 垃圾"）

| 项 | 体积 | 不动的原因 | 正确处理 |
|---|---|---|---|
| `archive_experiments/results_legacy_s1_to_s32` | 261 G | 唯一副本；含 14 个 ≥5GB XL ckpt（78 G） | 需人工裁决 |
| `.git/objects` 松散对象 | 49 G | 953,330 个；**远程 origin 是上游仓库** | `git gc` / 历史重写 |
| `results_archive_failed` | 59 G | `assets/runs/_archive` 链接目标；被 `exp_milestones/03_v21/meta.json` 引用（milestone ckpt 唯一副本） | 保留 |
| `_archive_historical` | 35 G | 同上（`04_v23`） | 保留 |
| `data/dino_cache/v11_sym_full/feats.f32` | 31 G | `feats.f16`(16G) 的精确 2× 升精度副本；`src/utils/dino_cache.py:32-34` 优先 f32、缺失回落 f16；仅 `v11_pretrain_M432_adaln4_sym_noise400k.json` 引用该 cache（**在跑的 v72 用 `top10_aug_v1`**） | 可删（有兜底），但属性能产物 |
| `baseline/` | 23.6 G | 顶层零引用（31 处命中全是 `assets/results/v17_s2_s2z_baseline`） | 需人工裁决 |
| `dataset/` | 23.5 G | 被 `src/eval/eval_full_3cond.py:18`、`tools/check_mccd_calligs.py:19` 引用；主线已迁 `data/` | 保留 |
| `_backups/` + `_rescued/` | 21 G | `_sync_work/purge_old_backup.sh` 删除 622GB 旧备份的**强制前置校验** | **不要动** |
| `exp-std/runs_AB` | 15.8 G | 4 个历史配置的 `results_dir` | 保留 |
| `exp-std/runs_purestd` | 7.7 G | 6 个 `v49_branch_*.json` 的 `resume_full` 指向其中 ckpt | 保留 |
| `_models/…/Qwen2.5-VL-3B-Instruct` | 7.1 G | 代码只用 4B 与 7B | 可删（可重下） |
| `_sync_work/` 内 4 个大件 | 1.13 G | `_v15_start_s2_60k.pt`(560M) / `_dino_patch_feats.npy`(376M) / `eval_backup.tar.gz`(225M) + 1 | 需人工裁决 |
| `archive/`、`MCCD/`、`_venv_qwenvl/`、`assets/pixmap/` 等 | ~8 G | 自引用或工具依赖 | 保留 |

## 5. 数据资产（latent shards）——**三套约定严禁混用**

| shards | VAE | 存盘 | 解码 | 用于 |
|---|---|---|---|---|
| `exp-std/data/shards_img_aug` | sd-vae-ft-ema | `mode()*sf` | `decode(lat/sf)` | v68、消融 4 臂 |
| `exp-std/data/shards_img_aug_calli`（**48 上**） | 旧 calli（step 25000，非标准） | 裸 `sample()` | `decode(sample/sf)` + 注噪补丁 | v71 |
| **`exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample`** | **新 calli（kl1e6, step 12500, 标准约定）** | **`sample()*sf`** | **`decode(lat/sf)`** | **v72（在跑）** |

- 冻结制品：`experiments/vae_frozen/calli_vae_kl1e6_s12500_stdconv/`（`MD5SUMS.txt` + `PROVENANCE.md`）。
- 自检（`tools/vae/verify_calli_latent_shards.py`）：16 shards / 77,823 行；存储 latent std **0.1955 ≈ sf 0.18215**（证明存的是 `sample*sf`）；`decode(lat/sf)` L1 = **0.0072**，对照 `decode(lat)` = 0.6646 → 约定唯一正确 ✓
- `dino_cache/` 7 个 cache 共 139 G，6 个仅 f16、1 个（`v11_sym_full`）f32+f16 双份。
