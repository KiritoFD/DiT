# DiT 工作区「垃圾/冗余/重复」普查 + 隔离计划（只读审计）

- 远程：`4090:/root/Workspace/xy/DiT`，总占用 **1.1 TiB（du -sh，硬链接已去重）**
- 审计时间：2026-10-09 19:5x–20:1x（服务器本地时间 UTC+8）
- 只读手段：`ls / find / du / stat / cat / head / sed / grep / wc / md5sum / readlink`；全部结论来自实际命令输出
- 体积单位：**GiB**（1 GiB = 1024³ B），取自 `du`（磁盘占用）；括号内给出精确字节或 `du -sh` 原始值
- 未触碰：`assets/runs/v72_callivae_kl1e6_s12500_frozen_sp_c2ot/` 与 `logs/v72_train_*.log` 仅做 `ls`/`find -name`/`grep -l`；未运行 nvidia-smi / python / kill / tmux；未访问 48

---

## 1. 主表（按建议释放空间降序）

| 路径 | 体积 | 文件数 | mtime | 是什么(证据) | 引用检查结果 | 建议动作 | 风险 |
|---|---|---|---|---|---|---|---|
| `archive_experiments/results_legacy_s1_to_s32/` | 261.0 GiB (267151.5 MB) | 508,242 (+2,330 dir) | 2026-09-06 20:11 | s1–s32 世代历史产物，67 个顶层目录；扩展名直方图 png 835,822 / json 4,436 / pt 372 | 全库 grep 无 src/tools/scripts/configs 命中；只有 exp_milestones 引用另两个 archive 子目录 | **review** | 中（唯一副本，含 14 个 ≥5GB XL ckpt = 78.04 GiB） |
| `.git/objects/` 松散对象 | 49.0 GiB（.git 76 GiB − pack 27 GiB） | 953,330 个松散对象 | 2026-10-09 | 256 个 loose 前缀目录；最大松散对象 0.26–0.33 GB（疑似历史上误提交的 .pt）；`find -links +1` = 0（无硬链接，全部实占） | N/A（git 元数据）；`.git/config` 的 origin = `https://github.com/facebookresearch/DiT.git` | **review** | 中高（需 `git gc`/`filter-repo`，非删除） |
| `archive_experiments/results_archive_failed/` | 59.1 GiB (60507.0 MB) | 250,542 (+2,125 dir) | 2026-10-01 20:32 | 31 个 v8–v10b 失败/归档 run | **是 `assets/runs/_archive` 符号链接的目标**（`readlink -f` 解析到它，目录列表 diff 完全一致）；另被 `assets/runs/MANIFEST.md:27`、`exp_milestones/03_v21/meta.json:4` 引用（v21 milestone ckpt 在此） | **review** | 中（milestone ckpt 唯一副本） |
| `archive_experiments/_archive_historical/` | 35.0 GiB (35867.3 MB) | 84,248 (+1,562 dir) | 2026-10-07 17:38 | `20261003_twostage` 历史快照（含 src/tmp_revert/misc 等备份代码与 run） | 根目录 `_archive` 符号链接目标；被 `exp_milestones/04_v23/meta.json:4` 与 `milestones_manifest.json:38` 引用 | **review** | 中（v23 milestone ckpt 唯一副本） |
| `data/dino_cache/v11_sym_full/feats.f32` | 31.18 GiB (33,484,308,480 B) | 1 | 2026-09-13 19:52 | **f16 的精确 fp32 升精度副本**：33,484,308,480 = 2 × 16,742,154,240 = 85155×256×384×4B；同一 `meta.json`（n=85155, patches=256, dim=384, csv=assets/train_fame3_sym_full.csv）；比 f16 晚 40 分钟写入；`tools/convert_dino_cache_f32.py` 默认目标就是该目录 | `src/utils/dino_cache.py:32-34` **优先读 f32、缺失回落 f16**；仅 1 个配置引用该 cache：`src/train/configs/v11_pretrain_M432_adaln4_sym_noise400k.json:71` + 5 个 tools（bench_gather/bench_gather2/diag_repa_dev/build_dino_cache/convert_dino_cache_f32）；**在跑的 v72 用的是 `data/dino_cache/top10_aug_v1`**（见 v72 `resolved_config.json:260`） | **review** | 低-中（删后功能回落 f16，仅损失 pinned-fp32 快路径；重生成需 CPU 全量升精度） |
| `.git/objects/pack/tmp_pack_uGPgSj` | 26.59 GiB (28,545,506,512 B) | 1 | 2026-10-04 00:17 | **被中断的 repack 残留临时包**：无同名 `.idx`/`.rev`，命名 `tmp_pack_*`；pack 目录里唯一的正规包只有 6,674,622 B（2026-08-12） | 无任何引用（git 不读取 tmp_pack） | **quarantine** | 低 |
| `baseline/` | 23.6 GiB (24183.4 MB) | 12,009 (+78,158 符号链接) | 2026-09-29 09:17 | 三方基线对比工作区：DG-Font 19G / FontDiffuser 4.3G / VQ-Font 174M / adapter / results / serial_logs | **顶层 `baseline/` 零引用**——grep `baseline/` 的 31 条命中全部是 `assets/results/v17_s2_s2z_baseline/...`；仅 baseline 内部脚本自引用 | **review** | 中（论文对比结果/三方代码，可重取但耗时） |
| `dataset/` | 23.5 GiB (24042.3 MB) | 1,454,153 (+734,533 dir) | 2026-08-13 23:23 | 旧版按类目切分的原始集：images 15G / latents 5.0G / canny 2.2G / skeleton 2.2G；`ASSETS.md` 记为「原始数据集 24G」 | 被 `src/eval/eval_full_3cond.py:18`（注释明确 root_dir 必须是 "dataset"）、`tools/check_mccd_calligs.py:19`、`scripts/legacy_*` 引用；业务主线已迁到 `data/` | **review** | 中（仍被 eval 脚本引用） |
| `exp-std/runs_AB/` | 15.8 GiB (16195.1 MB) | 8,686 (+95 dir) | 2026-10-04 23:56 | 54 个 v50/v51/v52/v53/v54「AB 世代」run | 被 `configs/v50_A_space...json:4`、`v51_B_joint...json:5`、`v52_char_supcon_table.json`、`v53_triple...json` 的 `results_dir` 引用 + `tools/backfill_in_mem_eval.py:12` | **review** | 低-中（重跑旧配置会写回这里） |
| `_backups/` | 14.4 GiB (14736.7 MB) | 1 (+1 log) | 2026-09-23 01:44 | `dataset_20260923.tar.gz` 15,452,503,336 B + 5B 的 `dataset_pack.log` | `_sync_work/purge_old_backup.sh:22,341` 把它作为**删除 622GB 旧备份前的强制前置校验**；`.gitignore` 含 `_backups/`；`prep_cleanup.sh:31` 列它 | **review** | 高（删掉则 purge 前置失效） |
| `exp-std/runs_smoke/` | 14.2 GiB (14521.5 MB) | 444 (+120 dir) | 2026-10-05 18:29 | 59 个 smoke/探针 run（smoke_A/B/B2/A192/A160…） | 仅被 `_sync_work/` 的 smoke/launch/probe shell 与 tables md 提及；**src/tools/scripts/configs 零命中** | **quarantine** | 低 |
| `exp-std/runs_purestd/` | 7.7 GiB (7920.0 MB) | 5,867 (+42 dir) | 2026-10-03 19:20 | 23 个 v46 p=1.0 纯标准字 run | `configs/v49_branch_{A_adaln_cont,A_xattn_cont,B_adaln4,B_xattn,C_every_layer,D_k4_styleca}.json:74` 的 `resume_full` 指向其中 0037500.pt + `tools/summarize_purestd.py`、`initmix_sweep.py` | **review** | 中（被 resume 依赖） |
| `_models/models/Qwen--Qwen2.5-VL-3B-Instruct/` | 7.1 GiB | 15 (+3 dir) | 2026-09-21 | HF 快照目录（同层还有 Qwen2.5-VL-7B 16G、Qwen3-VL-4B 8.3G） | 代码只指向 **Qwen3-VL-4B**（`tools/eval_extra_metrics.py:146`、`tools/vlm_compare_std_gt.py:35`）与 **Qwen2.5-VL-7B**（`tools/vlm_ocr_acc.py:28` 等）；**未发现任何 3B 引用** | **review** | 低（可重下） |
| `_rescued/` | 6.3 GiB (6459.1 MB) | 237,985 (+2,999 dir) | 2026-09-13 18:36 | `20260913_backup/{eval_imgs/5script 6.3G, records 64M}` 从已删 622GB 备份中抢救出的记录 | `purge_old_backup.sh:7,347` 以「records 文件数 > 100」作为删除前置；`rescue_backup_records.sh:10` 为其产出脚本 | **review** | 高（抢救唯一副本） |
| `assets/pixmap/` | 3.4 GiB | 4 (+1 dir) | 2026-08-22 | 像素图缓存（`ASSETS.md` 记为 assets/pixmap 3.4G） | `src/tools/scripts/_sync_work` grep `pixmap` **零命中** | **review** | 低（无引用，但用途不明） |
| `exp-std/runs/` | 2.3 GiB (2357.2 MB) | 3,845 (+28 dir) | 2026-10-03 12:55 | 16 个 v46 p0.2/p0.4 run（含 eval_stdskel_batch.csv） | 未找到任何 `results_dir: exp-std/runs` 配置；仅 `_sync_work` 表格提及 | **review** | 低 |
| `_venv_qwenvl/` | 1.84 GiB (1887.0 MB) | 4,874 (+781 dir, 65 零字节) | 2026-09-22 16:04 | 独立 venv（`pyvenv.cfg` 87B，bin/include/lib）供 Qwen-VL OCR 工具 | `_sync_work/setup_qwenvl.sh` 创建/使用；`docs`/脚本 18 处提及 | **review** | 低-中（可重建，但依赖联网） |
| `archive/` | 1.61 GiB (1646.3 MB) | 586 (+84 dir) | 2026-10-07 15:36 | 旧代码/诊断/探针归档；其中 `archive/tarballs/` 1.4G（含 `top10_latents_train_eval.tar.xz` 1.23 GB） | archive 内部脚本自引用；`archive/create_example.py` 产 `example_100`、`archive/tarballs/latent_missing.tar.gz` 产 `latent_missing` | **review** | 低 |
| `MCCD/` | 1.29 GiB (1321.3 MB) | 4,306 (+47 dir) | 2026-08-12 15:19 | `MCCD/MCCD/{MCCD-Calligrapher/calligrapher_dataset(4301 png), dual_task/{char_calligrapher_lmdb,char_dynasty_lmdb}(5 mdb)}`；**不存在 `MCCD_Character` 子目录** | `_sync_work/probe_mccd.py:14` 读 dual_task LMDB；`src/eval/gradio_controlnet.py:47` 指向的 `MCCD/MCCD/MCCD_Character/character_dataset` **已不存在（失效引用）**；代码里 384 处 `MCCD` 多为 `MCCDDataset` 类名 | **review** | 中（原始标注数据） |
| `exp-std/runs_5050/` | 1.29 GiB (1323.5 MB) | 6,660 (+40 dir) | 2026-10-03 16:03 | 5050 混合比 run | `configs/v46_5050_80k.json:4` 的 results_dir + `tools/summarize_remote_v46.py:19-32` | **review** | 低 |
| `assets/eval_isolated/` | 1.1 GiB | 5 (+2 dir) | 2026-09-23 | 隔离评测 ckpt 目录 | `_sync_work/run_head.sh:25` 引用 `glyph_gate_hard_15k/0015000.pt` | **review** | 低 |
| `exp-std/_attempt1/` | 1.13 GiB (1154.3 MB) | 1,018 (+12 dir) | 2026-10-03 03:34 | v46 attempt-1 run（含 v46-std-adaln4-top10-p0.2） | 仅 `_sync_work/tables_20261005.md:82`、`all_runs_table.md:342` 表格提及 | **quarantine** | 低 |
| `assets/dino_feat_raw.BROKEN.npz` | 0.575 GiB (616,987,290 B) | 1 | 2026-10-07 21:49 | 与 `assets/dino_feat_raw.npz` **字节数完全相同**（616,987,290 B，后者 2026-10-08 11:46），文件名自带 `BROKEN` | 无脚本引用 `.BROKEN` | **quarantine** | 低 |
| `exp-std/v54/` | 0.51 GiB (527.0 MB) | 423 (+8 dir) | 2026-10-05 00:02 | v54 世代 checkpoints/eval | 未找到配置引用 | **review** | 低 |
| `latent_missing/` | 0.142 GiB (145.8 MB) | 12,414 (+1 dir) | 2026-08-13 23:59 | 每图一个 8,320 B 的 `.npy` 缺失 latent 清单 | **已存在归档包** `archive/tarballs/latent_missing.tar.gz`（90,257,186 B，2026-08-14 00:03）；唯一引用是注释 `scripts/legacy_py/make_latent_shards.py:28` | **quarantine** | 低 |
| `_trash/` | 0.135 GiB (137.9 MB) | 43 (+38 dir) | 2026-10-01 18:25 | 已隔离的 predskel 脏目录（38 个 `shard_00000.npz`） | 仅注释 `_sync_work/launch_v36_formal.sh:2`「predskel* 脏目录已隔离到 _trash/」 | **quarantine** | 低（本身即垃圾区，确认无需回滚） |
| `assets/char_script_supcon_table.npy.v1_random.bak` | 0.108 GiB (116,390,016 B) | 1 | 2026-10-04 | 与 `char_script_supcon_table.npy` 同字节数的 `.bak`（另有小体积 `char_script_supcon_index.json.v1_random.bak`） | 无引用 | **quarantine** | 低 |
| `flip_backup/`+`flip_backup_fame2..5`+`flip_backup_manual` | 0.142 GiB (149,272 KB 合计) | 5,061 | 2026-08-14 ~ 2026-09-02 | 极性翻转前的原始图备份（fame3 集） | 文档引用：`docs/system/13_fame_zero_shot.md:12`、`docs/system/14_data_cleaning.md:43`、注释 `tools/build_fame_dataset.py:12` | **quarantine** | 低-中（文档化的溯源备份，仅 146 MiB） |
| `retro_posters/` + `retro_posters_latest/` | 0.046 GiB（33.4 + 13.7 MB） | 111 + 10 | 2026-09-11 00:46 / 07:43 | 复古海报出图 | 全库 grep **零命中** | **quarantine** | 低 |
| `gen_curves_S/` | 0.022 GiB (22.4 MB) | 336 (+17 dir) | 2026-08-15 12:14 | 曲线 eval 出图（0001000.json + 0001000_imgs/…） | 唯一引用 `scripts/legacy_sh/_eval_S_curves.sh:4`（`mkdir -p gen_curves_S`） | **quarantine** | 低 |
| `example_100/` | 0.007 GiB (7.1 MB) | 301 (+668 dir) | 2026-08-11 18:23 | 示例数据集（dataset/ + example.csv） | **已归档** `archive/tarballs/example_100.tgz`（3,585,989 B）；其生成脚本 `archive/create_example.py` 已归档 | **quarantine** | 低 |
| `gen_eval_S_old_imgs/` `gen_eval_C_imgs/` `compose_test_imgs/` | 0.003 GiB（1.2+1.3+0.8 MB） | 20+20+4 | 2026-08-15 | 旧 eval / 测试拼图 | 全库 grep **零命中** | **quarantine** | 低 |
| `train_dashboard.html`（根目录） | 32 KB | 1 | 2026-08-24 17:48 | 与 `tools/train_dashboard.html` **md5 完全相同**：`96d5e3a0942328aa5e22ea1eeb93a976` | 12 处引用全是 `os.path.join(HERE,"train_dashboard.html")`，HERE ∈ {tools, tools/evaluation, tools/dashboard, scripts/legacy_py} → 解析到 `tools/` 副本；根副本从不被加载 | **quarantine** | 低（重复副本） |
| `run_DiT.ipynb` | 8 KB | 1 | 2026-08-12 | 内容为 **上游 facebookresearch/DiT 的采样 notebook**（首 cell 引用 wpeebles.com/DiT、GitHub facebookresearch/DiT） | 全库 grep **零命中** | **quarantine** | 低（上游模板残留） |
| `environment.yml` | 4 KB | 1 | 2026-08-12 | 上游 conda 环境（python>=3.8 / pytorch-cuda=11.7），与实际 .venv 不符 | 全库 grep **零命中** | **quarantine** | 低（上游残留） |
| `CONTRIBUTING.md` / `CODE_OF_CONDUCT.md` | 3.6 KB + 3.6 KB | 2 | 2026-08-12 | 上游仓库模板文件 | 全库 grep **零命中** | **quarantine** | 低（上游残留） |
| ` 2`（文件名 = 空格 + "2"） | 0 B | 1 | 2026-10-04 13:22 | `find . -maxdepth 1 -type f -size 0` → `./ 2`；`ls` 显示 `-rw-r--r-- 1 root root 0 Oct 4 13:22  2` | 无引用 | **quarantine** | 低 |
| `exp-std/signal_arena/`、`exp/` | 0 B | 0 | — | **空目录**（`find . -type d -empty` 命中） | signal_arena 是 `tools/latent_signal_arena.py:21` 的输出目录；`exp` 仅被 `stop_and_survey.sh:35` 列为 ckpt 家底 | **quarantine** | 低 |
| `results/`（根目录） | 52 KB | 7 (+5 dir) | 2026-09-25 20:01 | 两个被弃的 `DiT-2Cond-S-2` run 目录，**checkpoints 子目录为空** | 无引用；`stop_and_survey.sh` 列 `exp`/`exp-std/ckpt` 但不含 `results/` | **quarantine** | 低 |
| `exp-std/ckpt/` | 0 B | 0 | — | **空目录** | `_sync_work/stop_and_survey.sh:35,553` 的 ckpt 盘点列表含它 | **review** | 低（空目录，删无收益） |
| `exp-std/reeval/` | 26 MB | 642 (+15 dir) | 2026-10-03 18:39 | ckpt 重评输出 | `tools/reeval_ckpts.py:80` 的 `--out-dir` 默认值；另有空目录 `reeval/failed_skelnet_runs__0000000` | **review** | 低 |
| `exp-std/signal_arena_full/` | 4.8 MB | 6 | 2026-10-03 19:57 | vae_space_probe 输出 | `tools/vae_space_probe.py:44` 的 `--out` 默认值 | **review** | 低 |
| `_pull_archive/` | 1.7 MB | 1 | 2026-10-09 07:48 | `dit_results.tar.gz`，**当天新拉入的包** | 零引用 | **review** | 低（刚拉取，可能在用） |
| `assets/d1_logs<CR>`（目录名结尾是回车 0x0D） | 4 KB | 0 | 2026-09-22 22:38 | `ls -labd` 输出 `assets/d1_logs\r`；**空目录**，与正常的 `assets/d1_logs`（10 files, 44K, 2026-09-23 18:43）并存 | 零引用 | **review** | 低（需人工 rename/删，命名污染） |
| `exp-std/data/` | 5.5 GiB (5680.1 MB) | 302 (+13 dir) | 2026-10-09 14:15 | latent shards（shards_img_aug_calli_kl1e6_s12500_rsample 等） | **活跃**：在跑 v72 的 `resolved_config.json:275` = `exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample`；另被 v48/v49/v51/v64 配置引用 | **keep** | — |
| `exp-std/csv/` | 49 MB | 9 | 2026-10-09 14:15 | 训练/评测 csv（train_top10_noaug.csv / eval200_fixed.csv 等） | **活跃**：v72 `resolved_config.json:2,173` 直接引用 | **keep** | — |
| `data/dino_cache/` 其余 6 个 cache | 91.7 GiB (96189084 KB) | 18 | 2026-10-07 04:57 | 全部只有 `feats.f16` + ids + meta（无 f32） | 被 `src/train/configs/*.json` 的 `repa_cache_dir` 大量引用；**在跑 v72 用 `top10_aug_v1`** | **keep** | — |
| `_models/`（除 3B 外） | 26.3 GiB | 170 | 2026-09-22 16:50 | Qwen2.5-VL-7B 16G + Qwen3-VL-4B 8.3G + ocr 2.0G | `tools/vlm_*.py`、`tools/eval_extra_metrics.py`、`_sync_work/setup_qwenvl.sh:35,42`、`dl_qwen.sh:11,16` | **keep** | — |
| `gradio_stdskel.py` | 21 KB | 1 | 2026-09-21 15:16 | std-skel-g DiT (c41x_cos_e) 的 gradio 前端 | 42 处命中/27 文件：`_sync_work/start_gradio.sh:2` + 10 个 `patch_gradio*.py` + `assets/archeology_survey.json:5429` 索引 `logs/gradio_stdskel.log` | **keep** | — |
| `data_path.json` | 4 KB | 1 | 2026-09-13 16:48 | 活跃数据资产清单（文档，内容为 v8 世代路径） | 唯一引用 `_sync_work/reorg_data.sh:42` 的根文件白名单 | **keep** | — |
| `ASSETS.md` | 4 KB | 1 | 2026-09-13 16:48 | 大资产清单（部分数据已过期，如 data 记为 412G） | 仅自引用 | **keep** | — |
| `exp_milestones/` | 34.9 MiB | 855 (+48 dir) | 2026-10-07 17:18 | eval200 milestone 交付物（checkpoint.pt 均为符号链接指向 archive） | `milestones_manifest.json`、`_sync_work/*` 表格 | **keep** | — |
| `eval_export/` | 38.8 MiB | 1,049 (+9 dir) | 2026-10-07 17:16 | 各版本 eval 出图导出（gt/v13/v21/…/v70） | 近期交付物，无脚本引用 | **keep** | — |
| `_sync_work/` | 1.49 GiB | 1,325 (+32 dir) | 2026-10-09 19:53 | 活跃运维脚本目录；内含 4 个大备份：`_v15_start_s2_60k.pt` 560M、`_dino_patch_feats.npy` 376M、`eval_backup.tar.gz` 225M、`code_data_backup.tar.gz` 33M | 自身即引用源 | **review**（仅针对 4 个备份大件） | 低 |

---

## 2. 隔离计划 CSV

```csv
action,path,size_GiB,reason
quarantine,.git/objects/pack/tmp_pack_uGPgSj,26.590,"被中断的 repack 残留临时包：无同名 .idx/.rev，正规 pack 仅 6674622B(2026-08-12)，mtime 2026-10-04 00:17"
quarantine,exp-std/runs_smoke,14.181,"59 个 smoke run；仅 _sync_work 的 smoke/probe shell 与表格提及，src/tools/scripts/configs 零命中"
quarantine,exp-std/_attempt1,1.127,"v46 attempt-1 run；仅 _sync_work/tables_20261005.md:82 与 all_runs_table.md:342 提及"
quarantine,assets/dino_feat_raw.BROKEN.npz,0.575,"与 dino_feat_raw.npz 字节数完全相同(616987290B)，文件名自带 BROKEN，无引用"
quarantine,latent_missing,0.142,"12414 个 8320B npy；已归档为 archive/tarballs/latent_missing.tar.gz(90257186B,2026-08-14)，仅遗留注释引用"
quarantine,_trash,0.135,"已隔离的 predskel 脏目录(38 个子目录)；仅 launch_v36_formal.sh:2 注释提及"
quarantine,assets/char_script_supcon_table.npy.v1_random.bak,0.108,"与正式文件同字节数的 .bak，无引用"
quarantine,flip_backup,0.054,"极性翻转前原始图备份(2751 文件)；仅 docs/system/13,14 与 build_fame_dataset.py 注释提及"
quarantine,flip_backup_fame2,0.039,"同上(757 文件)"
quarantine,flip_backup_fame4,0.029,"同上(896 文件)"
quarantine,flip_backup_fame5,0.016,"同上(515 文件)"
quarantine,flip_backup_fame3,0.005,"同上(97 文件)"
quarantine,flip_backup_manual,0.0002,"同上(39 文件)"
quarantine,retro_posters,0.033,"111 张复古海报；全库 grep 零命中"
quarantine,gen_curves_S,0.022,"336 文件曲线出图；仅 scripts/legacy_sh/_eval_S_curves.sh:4 mkdir 提及"
quarantine,retro_posters_latest,0.013,"10 张海报；全库 grep 零命中"
quarantine,example_100,0.007,"301 文件；已归档为 archive/tarballs/example_100.tgz(3585989B)，生成脚本已归档"
quarantine,gen_eval_C_imgs,0.001,"20 张旧 eval 图；全库 grep 零命中"
quarantine,gen_eval_S_old_imgs,0.001,"20 张旧 eval 图；全库 grep 零命中"
quarantine,compose_test_imgs,0.001,"4 张测试拼图；全库 grep 零命中"
quarantine,train_dashboard.html,0.00003,"与 tools/train_dashboard.html md5 相同(96d5e3a0942328aa5e22ea1eeb93a976)；12 处引用均按 HERE 解析到 tools/ 副本"
quarantine,run_DiT.ipynb,0.00001,"上游 facebookresearch/DiT 采样 notebook 残留，零引用"
quarantine,environment.yml,0.00001,"上游 conda 环境残留(pytorch-cuda=11.7)，零引用"
quarantine,CONTRIBUTING.md,0.00001,"上游模板残留，零引用"
quarantine,CODE_OF_CONDUCT.md,0.00001,"上游模板残留，零引用"
quarantine,results,0.00005,"根目录 2 个被弃 DiT-2Cond-S-2 run，checkpoints 子目录为空，无引用"
quarantine,./ 2,0.000,"文件名为空格+2 的 0 字节文件(2026-10-04 13:22)"
quarantine,exp-std/signal_arena,0.000,"空目录"
quarantine,exp,0.000,"空目录"
review,archive_experiments/results_legacy_s1_to_s32,261.000,"s1-s32 世代历史产物(508242 文件)，无代码引用，但为唯一副本且含 14 个 >=5GB XL ckpt(78.04GiB)"
review,.git/objects,49.000,"953330 个松散对象(953363 文件总计)；需 git gc/历史重写，remote origin 为上游 facebookresearch/DiT，非删除项"
review,archive_experiments/results_archive_failed,59.060,"assets/runs/_archive 符号链接目标；exp_milestones/03_v21 与 MANIFEST.md 引用 v21 milestone ckpt"
review,archive_experiments/_archive_historical,35.030,"根 _archive 符号链接目标；exp_milestones/04_v23 与 milestones_manifest.json 引用 v23 ckpt"
review,data/dino_cache/v11_sym_full/feats.f32,31.180,"f16 的精确 2x 升精度副本(33484308480B=85155*256*384*4)；loader 优先 f32 缺失回落 f16；仅 v11_pretrain_M432 配置引用，live v72 用 top10_aug_v1"
review,baseline,23.610,"三方基线工作区(DG-Font 19G 等)；顶层 baseline/ 零代码引用(命中均为 assets/results/v17_..._baseline)"
review,dataset,23.480,"旧版原始集(1454153 文件)；被 src/eval/eval_full_3cond.py:18 与 tools/check_mccd_calligs.py:19 引用，主线已迁 data/"
review,exp-std/runs_AB,15.820,"v50-v54 AB 世代 54 个 run；被 4 个历史配置的 results_dir 与 backfill_in_mem_eval.py 引用"
review,_backups,14.390,"dataset_20260923.tar.gz(15452503336B)；purge_old_backup.sh 删除 622GB 旧备份前的强制前置校验"
review,exp-std/runs_purestd,7.730,"23 个 v46 p=1.0 run；6 个 v49_branch 配置的 resume_full 指向其中 ckpt"
review,_models/models/Qwen--Qwen2.5-VL-3B-Instruct,7.100,"代码只用 Qwen3-VL-4B 与 Qwen2.5-VL-7B，未发现 3B 引用；可重下"
review,_rescued,6.310,"从已删 622GB 备份抢救的 eval_imgs/records；purge_old_backup.sh 以 records 完整性为前置，属唯一副本"
review,assets/pixmap,3.400,"无任何代码引用，用途不明(ASSETS.md 记为 3.4G)"
review,exp-std/runs,2.300,"16 个 v46 p0.2/p0.4 run；未找到 results_dir 配置引用"
review,_venv_qwenvl,1.840,"Qwen-VL OCR 工具用独立 venv(_sync_work/setup_qwenvl.sh)；可重建但需联网"
review,archive,1.610,"旧代码/诊断/探针归档，其中 archive/tarballs 1.4G"
review,MCCD,1.290,"MCCD-Calligrapher 4301 png + dual_task 5 个 lmdb；probe_mccd.py 读取，gradio_controlnet.py 的引用路径已失效"
review,exp-std/runs_5050,1.290,"5050 混合 run；被 v46_5050_80k.json results_dir 与 summarize_remote_v46.py 引用"
review,assets/eval_isolated,1.100,"隔离评测 ckpt；_sync_work/run_head.sh:25 引用"
review,exp-std/v54,0.510,"v54 世代 checkpoints/eval；未找到配置引用"
review,_sync_work/_v15_start_s2_60k.pt,0.547,"560M 训练快照备份，位于活跃脚本目录内"
review,_sync_work/_dino_patch_feats.npy,0.367,"376M DINO patch 特征中间产物"
review,_sync_work/eval_backup.tar.gz,0.220,"225M eval 备份包"
review,exp-std/reeval,0.026,"ckpt 重评输出(non-fatal 空子目录 failed_skelnet_runs__0000000)；tools/reeval_ckpts.py 默认输出目录"
review,_pull_archive,0.002,"dit_results.tar.gz，2026-10-09 07:48 当天新拉入，零引用"
review,assets/d1_logs<CR>,0.004,"目录名结尾为回车符 0x0D 的空目录，与正常 assets/d1_logs 并存，命名污染"
review,exp-std/ckpt,0.000,"空目录，_sync_work/stop_and_survey.sh 盘点列表包含它"
```

（注：路径 `assets/d1_logs<CR>` 中 `<CR>` 是真实的 0x0D 字符；若直接写入 CSV 会破坏行结构，建议以 `assets/d1_logs$'\r'` 形式在 shell 中处理。）

---

## 3. 空间回收小结

| 类别 | 体积 |
|---|---|
| **立即可回收（低风险，全部 `quarantine`）** | **≈ 43.1 GiB** |
| ↳ 其中 workspace 内垃圾 | ≈ 16.5 GiB（`runs_smoke` 14.18 + `_attempt1` 1.13 + `.BROKEN.npz` 0.575 + 其余 0.6） |
| ↳ 其中 `.git` 孤儿临时包 | 26.59 GiB（`tmp_pack_uGPgSj`，纯死文件） |
| **需人工确认（`review`）** | **≈ 547 GiB** |
| ↳ 历史归档三兄弟（legacy/failed/historical） | 355.1 GiB |
| ↳ `.git` 松散对象（需 gc/历史重写） | 49.0 GiB |
| ↳ 类数据集/三方件（baseline / dataset / _models 3B / MCCD / pixmap） | 35.4 GiB |
| ↳ 旧世代实验产物（runs_AB / purestd / runs / 5050 / v54 / reeval） | 28.6 GiB |
| ↳ 备份与抢救（_backups / _rescued） | 20.7 GiB |
| ↳ 可重建数据副本（feats.f32 / _venv_qwenvl） | 33.0 GiB |
| ↳ 其他（archive / _sync_work 大件 / _pull_archive / 空目录 / 命名异常） | 3.2 GiB |

**重要说明**：355 GiB 的历史归档与 49 GiB 的 git 松散对象都不是「垃圾」，而是「唯一副本/需要决策」的空间；`review` 不等于建议删除。真正零价值、可无脑回收的是 **43.1 GiB**。

**重复项结论（明确回答提问）**
1. `assets/runs/_archive` 与 `archive_experiments/results_archive_failed` **不是重复**：`readlink -f` 显示前者就是后者的符号链接，两侧目录列表 `diff` 完全一致（exit 0）。同理 `_archive` → `_archive_historical`、`archive_20260903` → `archive_experiments/archive_20260903`、`assets/results` → `runs`（`ln -s runs`）。
2. `data/dino_cache/v11_sym_full/feats.f32` 与 `feats.f16` **是同一份数据的两种精度**：字节数精确 2 倍（33,484,308,480 = 2 × 16,742,154,240），meta.json 完全一致（n=85155/patches=256/dim=384），f32 晚 40 分钟生成，且存在专门的升精度脚本 `tools/convert_dino_cache_f32.py`（默认目标即该目录）。**f32 仍被 loader 优先使用**（`src/utils/dino_cache.py:32-34`），但只有 1 个 9 月 13 日的配置引用该 cache；在跑的 v72 用 `top10_aug_v1`。删 f32 有 f16 兜底，属"可回退的性能优化产物"，故列 `review` 而非 `quarantine`。
3. `train_dashboard.html` 根目录副本与 `tools/` 副本 **md5 相同**，且所有引用按 `HERE` 解析到 `tools/`。
4. `assets/dino_feat_raw.BROKEN.npz` 与 `assets/dino_feat_raw.npz` 字节数相同；`assets/char_script_supcon_table.npy` 与 `.v1_random.bak` 字节数相同。
5. `.git` 之内无硬链接（`find -links +1` = 0，953,363 个文件全部实占）；`/root/Workspace` 的 1.1T 与各部分之和一致。

---

## 4. 引用检查方法说明（可复核）

**A. 全库主扫描（一次 find + 一次 xargs grep）**
```bash
cd /root/Workspace/xy/DiT
PATS='_trash|flip_backup|gen_curves_S|gen_eval_S_old_imgs|gen_eval_C_imgs|compose_test_imgs|example_100|latent_missing|retro_posters|MCCD|_venv_qwenvl|exp_milestones|eval_export|_pull_archive|_rescued|_backups|archive_experiments|results_archive_failed|results_legacy_s1_to_s32|_archive_historical|runs_AB|runs_smoke|runs_purestd|_attempt1|runs_5050|signal_arena|exp-std/ckpt|exp-std/reeval|v11_sym_full|feats\.f32|feats\.f16|check_dino_dtype|train_dashboard|gradio_stdskel|data_path\.json|run_DiT\.ipynb|ASSETS\.md|environment\.yml|CONTRIBUTING\.md|CODE_OF_CONDUCT\.md|dino_cache|baseline/|dataset/|_models/|_sync_work|tmp_push|tmp_sync'
find . -xdev \( -path ./.git -o -path ./data -o -path ./assets -o -path ./archive_experiments -o -path ./archive \
  -o -path ./.venv -o -path ./_venv_qwenvl -o -path ./baseline -o -path ./dataset -o -path ./_models -o -path ./logs \
  -o -path ./_trash -o -path ./exp-std/data -o -path ./_rescued -o -path ./_backups -o -path ./latent_missing \
  -o -path ./flip_backup -o -path ./flip_backup_fame2 -o -path ./flip_backup_fame3 -o -path ./flip_backup_fame4 \
  -o -path ./flip_backup_fame5 -o -path ./flip_backup_manual \) -prune -o -type f \
  \( -name '*.py' -o -name '*.json' -o -name '*.sh' -o -name '*.yaml' -o -name '*.yml' -o -name '*.md' \
  -o -name '*.ipynb' -o -name '*.txt' -o -name '*.toml' -o -name '*.cfg' -o -name '*.html' -o -name '*.js' \
  -o -name '*.ts' -o -name '*.ini' \) -print0 | xargs -0 -r grep -HnE "$PATS"
```
另有 3 段补充扫描：`assets` (maxdepth 2 + *.py/*.json/*.sh/*.md/*.csv)、`exp_milestones eval_export _pull_archive _rescued _backups _sync_work _review _ot_scratch scratch`、根文件 `ls -la` 与 `.gitignore`。原始输出 32,392 行，已按 token 去重统计（见下）。

**B. 缺口补齐扫描（针对 A 中被 prune 的目录）**
```bash
for D in archive data assets baseline dataset _models logs exp-std _pull_archive; do
  find "$D" -type f \( -name '*.py' -o -name '*.sh' -o -name '*.json' -o -name '*.md' -o -name '*.txt' \
    -o -name '*.ipynb' -o -name '*.yaml' -o -name '*.yml' -o -name '*.cfg' \) -print0 \
    | xargs -0 -r grep -lE "$PATS"
done
```
命中文件数：`assets` 387（几乎全是 `assets/runs/**/resolved_config.json` 的历史快照）、`exp-std` 153（同上，run 目录内的 resolved_config/log）、`archive` 48、`baseline` 48、`data` 9（dino_cache 的 meta.json 自指）、**`dataset` 0、`_models` 0、`logs` 0、`_pull_archive` 0**。

**C. 定向校验（关键结论的直接证据）**
- `src/utils/dino_cache.py` 第 4–14 行（格式说明）与第 32–34 行（`feats.f32` 优先、`feats.f16` 回落）
- `tools/convert_dino_cache_f32.py` 第 8、129–144 行（默认 `data/dino_cache/v11_sym_full`，f16→f32 升精度）
- `grep -rn '"repa_cache_dir"' src/train/configs/*.json` → 143 个配置文件全量映射（v11_sym_full 仅 1 个命中）
- 在跑 run 的 `resolved_config.json`：`repa_cache_dir=data/dino_cache/top10_aug_v1`、`data_csv=exp-std/csv/train_top10_noaug.csv`、`latent_shards_dir=exp-std/data/shards_img_aug_calli_kl1e6_s12500_rsample`
- `md5sum train_dashboard.html tools/train_dashboard.html` → 同一 md5
- `readlink -f assets/runs/_archive _archive archive_20260903 pretrained_models assets/results`
- `diff <(ls -1 assets/runs/_archive) <(ls -1 archive_experiments/results_archive_failed)` → exit 0
- `find .git/objects -type f -links +1 | wc -l` → 0；`stat -c '%s' .git/objects/pack/tmp_pack_uGPgSj` → 28,545,506,512
- `_sync_work/purge_old_backup.sh`（622GB 备份删除脚本的前置校验链：`_backups/…tar.gz` + `_rescued/records`）、`_sync_work/prep_cleanup.sh`（.gitignore 生成逻辑）

**D. 未覆盖/局限（请复核者注意）**
1. 未运行任何 git 命令（避免 `git status/gc` 触碰索引），因此**无法判定 `.git` 里 0.26–0.33 GB 的松散对象是否可达**；只能确认 `tmp_pack_*` 是死文件、松散对象实占 49 GiB。
2. `data/`、`dataset/`、`baseline/`、`_models/`、`logs/`、`.git/` 只做了"文件名级"grep（B 段），未逐字节判断；`assets/runs/**/resolved_config.json` 的历史路径快照不算活跃依赖，但**重跑旧配置时会需要对应目录**（已逐条标注在"引用检查结果"列）。
3. 文件名相同的目录 `assets/d1_logs` 与 `assets/d1_logs\r` 只能确认后者为空目录，未推断产生原因。
4. 体积均为 `du`（磁盘占用，硬链接已去重）；`archive_experiments` 与 `assets` 的 356G/276G 不重复计算符号链接目标。
5. **本次唯一远程写入**：`/tmp/` 下的 13 个审计脚本 + 1 个 `/tmp/out_a.txt`（首轮 nohup 重定向产物）。工作区零写入；`assets/runs/v72*` 与 `logs/v72_train_*.log` 只读，未运行 nvidia-smi / python / kill / tmux，未访问 48。
