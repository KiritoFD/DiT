# 历史归档代表性 Checkpoint 清单（2026-10-24 深度清理）

背景：4090 磁盘深度清理。归档目录中的全部 checkpoint 已按本清单裁决：
**表内 19 个 ckpt 保留**（连同各自目录下的评测 JSON/CSV/海报/日志/manifest 全部保留），
其余 330 个归档 ckpt 删除（约 260.6 GiB，清单 `assets/disk_cleanup_archive_ckpt_20261024.json`）。
同日更早：`assets/runs` 旧代实验 487 ckpt 删除（423.2 GiB，`assets/disk_cleanup_legacy_20261024.json`，
保留 DiC-768@50k/80k 等现行基线终版）、dino_cache/v90 scratch 等 ~178G 删除。
删除目录：`_quarantine_20261009`（旧仓库快照）、`_models`（Qwen VL 公共模型，可重下）、
`baseline`（DG-Font/FontDiffuser/VQ-Font 第三方基线，公共可重下）、`dataset`、
`data/archive/{legacy_images,legacy_skeletons,legacy_latents,mid_data}`（派生数据，可由 UNIFIED_RAW 重生成）。
**保留**：`data/archive/{legacy_csv,legacy_plans,std_legacy,quarantine}`（小体积标签/计划书）。
主要实验历史以本地机器为准；本文件是 4090 上仍存在的代表性权重的索引。

## 保留理由与索引（4090:/root/Workspace/xy/DiT/ 下相对路径）

### XL 时代（2026-08，DiT-3Cond-XL 系列，当前架构的远祖）

| ckpt | 是什么 | 为什么留 |
| --- | --- | --- |
| `archive_experiments/results_legacy_s1_to_s32/results/exp_xl_head/20260814-184935-DiT-3Cond-XL-2/checkpoints/0037000.pt` | XL 骨干条件头首跑（3Cond-XL） | 条件注入架构演进起点 |
| `…/exp_xl_head_r64|…/0037000.pt`、`exp_xl_head_r32`、`exp_xl_head_r32_attn`、`exp_xl_head_r8` | 头部 rank 64/32/attn/8 消融五连 | LoRA-rank 对条件容量的第一次系统消融；结论直接影响了后来 rank16 的选择 |
| `results_legacy_s1_to_s32/v3a_xl/20260815-232600-v3a-2factor-glyph-XL-lora-cs/checkpoints/0004000.pt` | v3a XL 2-factor glyph + LoRA-cs | 双因子（内容×风格）分解时代的代表 |
| `results_legacy_s1_to_s32/v3a_xl_highdim/*/…0001000|0004000|0005000.pt`（3 runs） | XL highdim 系列（最高容量旧代） | 旧代容量上限证据（三个 run 各留末点） |
| `results_legacy_s1_to_s32/v3a_xl_skelhead/…0001000|0015000.pt` | XL skelhead 引导 | 骨架引导 XL 尝试（dead-end 但有档案价值） |

### s 系列阶段实验（2026-08/09，机制演进链）

| ckpt | 是什么 | 为什么留 |
| --- | --- | --- |
| `s22_char_embed_only/20260830-…/0032500.pt` | 纯 char-embedding 条件 | "只有字 ID 不够"的关键负结果 |
| `s23_glyph_cond/20260831-…/0010000.pt` | glyph 条件注入 | 字形条件通路的首次正面验证 |
| `s30_dino_char_strong_pretrain/…0132500.pt`（+017500 备份） | DINO 字符特征强预训练 | **DINO 风格条件路线的直系祖先**（今天的 B 路线由此演化） |
| `s32_repa_finetune/…0010000|0015000.pt`、`s32c_chain/…0090000.pt`、`…s32d-repa-super/0015000.pt` | REPA 微调/链式/super | REPA 时代的终局证据（后续决策"不再用 REPA"的依据） |

### 完整删除（仅指标/日志留存）的大类

- `results_legacy_s1_to_s32` 其余 50+ 目录（ctrl_fame 系、ctrl_skel 系、s2/s5/s10–s20 组合引擎时代、
  px/pixel-sk 系列、s25/s26/s28/s29/s31、v3b/v3c glyphcond 变体、exp_xl_head 其余 rank）；
- `results_archive_failed` 全部（v8_3stage 19G、v9a/v9c/v10a/v10b 失败臂、failed_skelnet_runs、_aborted）——
  失败教训在实验总账与分析文档中，权重无复用价值；
- `_archive_historical/20261003_twostage` 的 ckpt（34G 中仅留 README/MANIFEST——两阶段实验的直接后继
  是后来的 generic48 curriculum，其演进由现存 generic48 系列承担）；
- `_archive_historical/20261001_bespoke` 的 ckpt。

### 复核方式

任一保留 ckpt 的上下文（config/日志/评测）在其同级目录；全量删除明细在
`assets/disk_cleanup_archive_ckpt_20261024.json`（本次）、`assets/disk_cleanup_legacy_20261024.json`（旧代）、
`assets/disk_cleanup_20261024.json`（今日 u100/reference/generic48 中间 ckpt）。
