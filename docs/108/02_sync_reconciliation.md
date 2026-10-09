# 108-02 · 本地 ↔ 远程 源码对账（2026-10-09）

> **目的**：在没有正式同步契约（`tools/remote_sync/` 里是 8 月的旧文件）的情况下，回答"训练机上跑的代码，和版本控制里的代码，到底是不是同一份"。
> **答案**：不是。**核心训练链路有 14 处真实分叉，其中 12 处是本地较新（含 10-08/10-09 三项改动）**。

## 0. 方法

1. **取指纹**：对远程 `src/ tools/ tests/ scripts/ _sync_work/ docs/` 与根目录的 `*.py|*.sh|*.json|*.md|*.yaml|*.yml|*.txt`（排除 `__pycache__` 与 `*/logs/*`）计算 `md5sum` → **2,669 个文件**。
2. **本地同域枚举**：同规则枚举本地 → **6,362 个文件**。
3. **一级对账**：按相对路径比对 md5 → 一致 / 内容不同 / 仅一侧有。
4. **二级定性**：对"内容不同"的每个文件，在远程计算 **LF 归一化 md5**（`tr -d '\r' | md5sum`），与本地同样归一化后比对 —— **归一化后一致 = 纯换行符差异**，仍不一致 = **真实分叉**。
5. 脚本：`scratch/sync_manifest.sh`（远程指纹）、`scratch/norm_md5.sh`（归一化）、`scratch/compare_sync.ps1`、`scratch/classify_divergence.ps1`、`scratch/diff_divergence.ps1`。

## 1. 一级对账结果

| 结果 | 数量 | 说明 |
|---|---|---|
| **两侧一致** | **2,371** | 同步良好 |
| 内容不同 | 95 | 见 §2 拆解 |
| 仅远程有 | 203 | 见 §3（**含未回传资产**） |
| 仅本地有 | 3,896 | 见 §4（多为 10-05 归档与分目录重组） |

## 2. 95 处内容差异的拆解

| 类别 | 数量 | 例子与算术验证 |
|---|---|---|
| **纯换行符（CRLF vs LF）** | **81** | `tools/encode_aug_latents.py`：远程 8964 B，本地 9171 B，CRLF=207 → **8964+207=9171** ✅；`_sync_work/extract_metrics_4090.py`：4162，本地 4260，CRLF=98 → **4162+98=4260** ✅ |
| **真实分叉** | **14** | 见 §2.2 |

> 结论：**"95 个文件不一致"的说法是误导性的；真实分叉只有 14 个**。本地侧由 Windows 工具写出的文件带 CRLF，远程侧是 LF，这是 md5 差异的主因（无害）。

### 2.2 真实分叉 14 处（`scratch/sync_gap/real_divergence.csv`）

| 文件 | 远程字节 | 本地字节 | 方向 | 差异性质（hunk 数） |
|---|---|---|---|---|
| `src/train/train.py` | 171,728 | **176,545** | 本地新 | 12 hunks；`train_vae_encoder`/`calli_vae_path` 相关段缺失 |
| `src/loss/flow_matching.py` | 26,661 | **27,673** | 本地新 | 11 hunks；**整个 SiT SDE 采样器缺失**（见下） |
| `src/train/cli.py` | 92,708 | **95,097** | 本地新 | 4 hunks；缺 `--sde-gamma` / `--train-vae-encoder` / `--calli-vae-path` |
| `src/eval/cli.py` | 78,565 | **79,784** | 本地新 | 3 hunks；缺同批参数 |
| `src/eval/in_mem_eval.py` | 62,706 | **63,464** | 本地新 | 5 hunks；缺 calli 解码约定分支 |
| `src/train/configs/_probe_inject12.json` | 3,867 | 4,385 | 本地新 | 3 hunks |
| `tests/test_arch_v2.py` | 17,692 | 17,699 | 本地新 | 1 hunk |
| `README.md` | 32,365 | 41,318 | 本地新 | 本地为 10-09 版 |
| `docs/README.md` | 3,676 | 5,173 | 本地新 | 同上 |
| `docs/system/00_README.md` | 5,453 | 5,946 | 本地新 | 同上 |
| `_sync_work/v50_A_space_xattn_style_adaLN_top10.json` | 3,031 | 3,031 | — | 同长不同内容 |
| `_sync_work/v51_B_joint_kv_k4_top10.json` | 4,215 | 4,215 | — | 同长不同内容 |
| `tools/audit_minimal_best.py` | **3,519** | 1,920 | ⚠ **远程新** | 9 hunks；**远程有未回传改动** |
| `_sync_work/peek_A.sh` | **1,819** | 1,719 | ⚠ **远程新** | 远程更完整 |

**决定性证据（关键标识在两地的出现次数）**：

| 标识 | 本地 | 远程 | 含义 |
|---|---|---|---|
| `sde_gamma` | **13** | **0** | 远程**完全没有** SiT SDE 采样器 |
| `sde-gamma` | 2 | **0** | 同上（CLI 层） |
| `train_vae_encoder` | 2 | **0** | 远程**没有** REPA-E 联合微调开关 |
| `calli_vae_path` | 1 | **0** | 同上 |
| `calli_decode_noise` | **7** | **0** | 远程**没有** Calli-VAE 解码约定分支 |
| `use_c2ot` | 11 | 11 ✅ | 10-06 的 C2OT 已同步 |
| `c2ot_mode` | 10 | 10 ✅ | 同上 |

**`flow_matching.py` 的实际差异（远程缺的就是这段）**：

```python
# 本地有、远程没有 —— SiT SDE 随机纠偏采样器
if self.sde_gamma > 0:
    h = -dt
    drift = v_eff * (1.0 + (self.sde_gamma ** 2) * t_i)
    noise_scale = self.sde_gamma * th.sqrt(th.clamp(2.0 * t_i * (1.0 - t_i) * h, min=0.0))
    x = x + dt * drift + noise_scale * th.randn_like(x)
else:
    x = x + dt * v_eff
```
远程对应位置只是 `x = x + dt * 0.5 * (v1_ + v2)` 的朴素 Heun 更新。

## 3. 仅远程有（203）——**未回传的资产**

| 类别 | 数量 | 明细 |
|---|---|---|
| **消融/变体配置** | 8 | `src/train/configs/ab_mod_{ctrl,ln,mlp,qk,rope}.json`（**README §5 消融表的配置，本地缺失**）、`{v64_skel_callig_script_4ch,v65_tables_condinj2468,v68_pix,v68_dino_raw,v68_randtab}.json` |
| 工具脚本 | 4 | `tools/{build_top10_sym_aug_5x,encode_top10_aug_5x,dino_expand_top10_aug_5x,probe_ceiling_exhaustive}.py` |
| 数据清单 | 2 | `src/utils/std_glyph_latent_v2/{coverage_report,manifest}.json` |
| **过期文档** | 26+ | `docs/system/10…75` 共 20 篇旧编号 + `docs/01_architecture/`(2) + `docs/922/`(1) + `docs/HANDOVER/HISTORY_REVIEW/STATUS`(3) —— 本地已在 10-05 归档，**远程仍是旧布局** |
| 运维/探针脚本 | ~10 | `_sync_work/{check_eval_oov,extract_rgb_feats_raw,find_python,pretrain_char_supcon,probe_table_slice_schema,check_eval_oov}.py`、`evalonly_v50a.sh`、`all_runs_table.md`、judge json ×4 |
| 补丁备份 | 33 | `_sync_work/_AB_patch`(18) + `_patch*_bak*`(19) |
| 历史运行器 | 71 | `scripts/legacy_runners/` |
| 遗留/其他 | ~40 | `scripts/legacy_scripts/`(6)、根 `ASSETS.md`/`data_path.json`/`gradio_stdskel.py`/`train_dashboard.html` |

> ⚠ **最该抢救的是那 8 个配置**：它们是 README §5 与 `docs/107` 引用的实验定义，**本地仓库里没有**，一旦远程磁盘出事就无法复现那批消融。

## 4. 仅本地有（3,896）——多为有意的本地归档

| 目录 | 数量 | 说明 |
|---|---|---|
| `docs/archive` | 2,084 | 10-05 归档的历史文档（远程未清理） |
| `tools/legacy` | 197 | 本地整理后的历史工具 |
| `tools/poster_archive` | 182 | 海报归档 |
| `_sync_work/baseline_src` | 161 | 三方基线源码镜像 |
| `tools/remote_pulled` | 92 | 从远程拉回的文件 |
| `scripts/ops` | 63 | 运维脚本 |
| `docs/106moyi` | 62 | 墨意复现图文 |
| `src/train` | 52 | 本地独有的配置/脚本 |
| `scripts/debug` `tools/analysis` `_sync_work/baseline_adapter` `src/eval` `tools/archive` | 44/39/37/33/31 | — |
| 根目录 | 24 | v21→v42 启动器序列（另见 [04](04_local_inventory.md) §3） |

**口径提醒（误报）**：`src/train/configs/` 里 `v13_fs_*/v15_fs_*/v16_fs_*` 共 28 个中文名配置在**两侧都存在**，但因文件名编码（远程 GBK 字节 vs 本地 UTF-8）被判为"仅远程有"——**不构成缺口**。

## 5. 结论与建议

1. **同步停在 ~10-06**：C2OT（10-06）在两侧一致；此后的 **SDE（10-08）→ REPA-E（10-08）→ VAE 解码约定（10-09）** 三项**从未落到训练机**。
2. **直接后果是口径问题，不只是版本问题**：仓库默认已是 `heun + 50 + cfg 0.8 + γ 0.5`（commit `b28e393`），而远程只能跑 `cfg 1.0 + 纯 ODE`。**v72 在远程产出的曲线因此与 README/107 引用的默认协议不同源**；任何"v72 vs v68"的横向比较都必须注明口径。
3. **反向也有缺口**：`tools/audit_minimal_best.py`、`_sync_work/peek_A.sh` 是**远程较新**，说明存在"只在远程改、从未回传"的工作方式。
4. **建议动作**（本报告不擅自执行）：
   - **A1**：等 v72 完训（或确认可重启）后，把 12 个"本地较新"文件同步到 4090；把 2 个"远程较新"文件拉回本地。
   - **A2**：明确远程评测口径并在所有表格标注；若要与 README 对齐，需先同步 SDE 再重测。
   - **A3**：修 `tools/launch_v72.sh` 的溯源字段（`git_commit` 当前记的是**上游 facebookresearch/DiT 的提交** `ac2c25e`，无效）。
   - **A4**：把 §3 的 8 个配置 + 4 个工具脚本 + 2 个清单回传本地。
   - **A5**：建立正式同步契约（脚本 + 排除清单 + 本对账流程），替代临时 scp。
