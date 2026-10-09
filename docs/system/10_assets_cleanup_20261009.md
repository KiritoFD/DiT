# 机器资产盘点与清理记录（2026-10-09）

## 48（dserver，`/home/ds/Workspace`）

磁盘：清理前 249G 可用（86% 满）→ 清理后 **583G 可用**（回收 ≈330G）。

### 已删除（非 best ckpt / 中间产物）

| run | 原有 ckpt | 保留 | 回收 |
|---|---|---|---|
| results/v61_triple_prior_sota_b1024 | 22 个（5k→110k） | **0060000.pt（best 0.6018@60k）+ 0110000.pt（final）** | ~69G |
| results/v60_randinit_sota | 11 个（5k→55k） | 0055000.pt（final） | ~35G |
| results/v56_classic_sp2 | 16 个（5k→80k） | 0080000.pt（final） | ~20G |
| experiments/capacity_ladder（6 个 tier run） | 35 个 | 每 run 最后一个（tier2 20k / tier3_b 5k / tier3_b_aug 40k / v66route 35k+60k / tier4_l 50k） | ~100G |
| moyi/results（moyi_top10_rf + _4ch） | 34 个（52G+61G） | **全删**（最终结果已归档拉回） | ~113G |

完整删除清单：`_pull_archive/deletion_manifest_48.txt`（48 上）。

### 保留不动

- `assets/results/v71_callivae_sp_c2ot/`（v71 全部，含 3 个 run 目录）
- `experiments/ablation_modern/`（A0 基座 ckpt 3 个）、`experiments/ablation_phase_a/`（exp1 ckpt 2 个）
- `experiments/recurrent_skel_refine/`（v46 单 ckpt）、`experiments/v68_aug_sp_c2ot/`（48 侧 200k ckpt）
- `data/`、`pretrained_models/`、eval187（capacity_ladder 评测图）
- 其他项目（ERV_mouse 等）完全未碰

### 待拉回（在 48:`/home/ds/Workspace/DiT/_pull_archive/`，已从 /tmp 移出防丢）

| 文件 | 大小 | 内容 |
|---|---|---|
| dit_results.tar.gz | 54M | capacity_ladder/{eval187,configs,logs,results非ckpt} + ablation_modern + ablation_phase_a + v68(48侧) + results/ 指标（全部排除 ckpt） |
| moyi_final.tar.gz | 141M | moyi 两个 run 的 eval_samples/posters/eval_ours200fix + 指标 json/csv + exp-std-csv |
| deletion_manifest_48.txt | 11K | 删除清单 |

**⚠️ 拉取暂被阻塞（2026-10-09 07:50 实测）**：dserver 出向网络故障 —— 到同内网 4090 的 ping 丢包 50%、RTT 1.3~1.5s，scp 实测 ~1.4KB/s（当天早些时候 ~50KB/s，持续恶化）。ssh 交互（小包重传）正常，大流量传输必卡死。已尝试并排除：直拉本地、4090 中转（48→4090 同样丢包，临时密钥已撤）。**待链路恢复后再拉**（两个包共 195M，恢复后 ~1 分钟）：
```powershell
scp 48:/home/ds/Workspace/DiT/_pull_archive/*.tar.gz 48:/home/ds/Workspace/DiT/_pull_archive/deletion_manifest_48.txt _sync_work\archive_48\
```
建议顺手让管理员查一下 dserver 的网卡/交换机口（丢包率不像拥塞，像硬件/驱动层问题）。

---

## 补充 (Part 2): 4090 深入重组 + 主表/主图重抽

### 4090 runs 家族化（91 个顶层 run → 15 个家族目录）

```
assets/runs/
  v10b/(11G)  v11_pretrain/(33G)  v12/(5.1G)  v13/(8.0G)  v14/(2.3G)  v15/(13G)
  v16/(153M)  v17/(18G)  v18_19/(1.6G)  v54_67_tables/(28G)  v68_family/(66G)  v69_70/(16G)
  moyi/(4.0G)  ablation_modern/(15G)  probes_smoke/(877M)
  _archive(60G, 符号链接)  _loose_ckpts/  MANIFEST.md
```

- 空壳清除：`_sweep`、`v11_pretrain_Sp2_base_wz2`、`v15_multistyle_k4`、`v19_gq_e0_100k`（合计 <200K，无 ckpt）
- `assets/_scratch/` ← 收纳 `_smoke_rgb`/`_calib_dino`/`_smoke_tab_dino`/`_smoke_callig_raw`/`_smoke_top10_rgb`（昨日的冒烟产物，55M）
- 根目录 4 个 tarball（eval200 相关, ~2G）→ `_archive/root_tarballs/`
- `data/_quarantine_v2/_v3` → `data/archive/quarantine/`；`dino_cache/`(139G) 与 `data/archive/results_legacy`(261G 表观, 多为硬链接) **未动**
- 旧 `MANIFEST.md`（09-13 指标快照）保留，已在新版顶部加"目录结构"节

### 主表/主图重抽（README §2.5）

- 数据源：4090 `assets/runs/**/*/checkpoints/eval_auto_*.json`（509 个）+ 48 侧 `eval_strict/*.json` + `capacity_ladder/**/metrics_40k.json` + moyi eval json
- 产物：`docs/experiments/mainline_20261009.csv`（17 行 × 187 协议）+ 3 张图（`docs/system/imgs/fig_*_20261009.png`）
- 关键校正：v68 旗舰 **0.6149@200k**；v61 三表先验 65k **0.6067** 仍在爬升；阶梯 B/2+aug+route **0.6104@60k**；moyi 4ch **0.6085@80k**

### 未完成项（遗留）

- ⏳ 48 `_pull_archive/` 两个归档包（195M）**因 dserver 网络故障尚未拉回本地**，恢复后执行见上文命令
- L 级阶梯（tier4_l）只有 ckpt 无评测


## 4090（`/root/Workspace/xy/DiT`）

磁盘 1.2T 可用（不紧张），以整理重组为主、基本不删除。

### v68 基座备份（单开目录，不删任何 ckpt）

- `/root/Workspace/BACKUP_v68_base/v68_aug_sp_c2ot/` ← 硬链接复制（`cp -al`，零额外空间），43 个 ckpt 与原件一致
- 原 run 已移入 `assets/runs/v68_family/`，硬链接不因移动失效

### runs 统一目录（`assets/runs/`）

117 个散落 run 全部归位，按组分层：

```
assets/runs/
  v68_family/        (66G: v68_aug_sp_c2ot 基座 + v68_pix/dino_raw/randtab)
  _archive/          (60G, 符号链接 -> archive_experiments/results_archive_failed)
  ablation_modern/   (15G: ab_mod_ctrl/ln/mlp/qk/rope)
  probes_smoke/      (_probe* _smoke* _t_* 等临时)
  _loose_ckpts/      (散装 ckpt: v11_M432_best_strict_152500 / v12_w320_last + 旧日志)
  <91 个顶层 run>    (v66/v70/v65/v11/v10b/moyun/std_*...)
```

- **`assets/results` 现在是指向 `runs` 的符号链接** —— 训练/评测脚本输出路径不变，新 run 自动落进 `assets/runs/`。

### data/ 轻整理

- 删除空壳：`_quarantine`×2、`_quarantine_v4`、`50k_v2_glyph15k`、`50k_v2_augmented`（合计 <1M）
- 保留待查：`_quarantine_v2`(113M)、`_quarantine_v3`(84M)
- 大头未动：`dino_cache/`(139G)、`archive/`(35G)、`skel/`(8G)

### 在跑进程（整理期间与之后都未受影响）

- `train_dino_calli_vae.py`（DDP×4，calli_vae_dino_stdconv_kl1e6，20000 步）全程存活
- 其读写路径（experiments/calli_vae_*、data/pretrained、UNIFIED_RAW）均未移动
