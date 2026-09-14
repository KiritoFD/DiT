# 55. fame-tj-uc base 数据线启动：Sp/2 + 新书家表（2026-09-14）

## 1. 动机

### 1.1 为什么建新数据集（fame-tj-uc base）
- fame3 线只有 41 书家 / 28k 原始图；strict 已平台（0.5680@390k），继续在原数据上叠配方收益递减。
- 三个新数据源合并：
  - **Calli-Tongji.zip**（5,000 张，46 书家×书体，每目录 100 字）→ 楷/行/隶 2,992 张 + v4 对称增强 → 8,976 行
  - **UniCalli_dataset**（HF gated，39,105 字符 bbox，93 书家）→ 裁切楷/行/隶、归一名后 27,253 张
  - fame3 原始 28,385（楷 0/行 3/隶 4）
- 合并过滤：<224 张/书家剔除（44 家），strict 泄漏排除（tongji 8 + UniCalli 121），草书 2,000 排除。
- **最终 base：54,892 行、45 书家（含赵孟頫 3,487）、5,405 唯一字、楷 28,704/行 23,378/隶 2,810**。
- fame3 曾丢失赵孟頫/乾隆/李斯（script 标签"六体/篆"不在楷行隶），已查明。
- 数据质量：反相筛查 54,892 张（CV：border 亮度 + ink ratio）→ **UniCalli 1,682 张反色（6.7%，拓片类）已原位反相归一**，fame/tongji 全干净。下游 skel3/canny PNG 已用反相后图重生成。

### 1.2 为什么上 Sp/2（h512, 59M）
- 容量-质量曲线（doc 54）：S/2 30M seen 天花板 ~0.52；**Sp/2 59M 可达 0.67-0.76 seen / 0.5680 strict**。
- base 数据量 54,892 行（近 fame3 两倍）+ 45 书家 → 容量升级合理。

### 1.3 为什么去掉条件噪声增强
见 §2.3 noise400k 结论：**被证伪**。

### 1.4 为什么重建书家表
- 旧表 `callig_emb_pretrained.pt` 是 fame 41 家（(41,128)），base 表 53 行（52+null）——shape 不匹配 assert 会炸。
- `freeze_table()` 会把新书家行冻死在随机 → 赵孟頫等 11 家风格条件失效。
- 方案：**SupCon 同配方重建**（DINO CLS 54,892 特征 + InfoNCE 同书家正对 + 质心锚定，3000 步）→
  `callig_emb_pretrained_base.pt` (52,128)，anchor cos→1.0、pairwise cos 0.074（未塌缩，塌缩基线 0.323）。

## 2. 上一次实验结果：v11_pretrain_M432_adaln4_sym_noise400k

- 配方：M/2 (h432) + 对称增强 sym_full + REPA 0.03 + adaLN4 + **条件噪声增强
  （glyph_noise_scale=0.1, glyph_noise_prob=0.3, glyph_patch_drop=0.05）** + 400k cosine + in-mem eval。
- 结果（strict n=50 / seen n=10, cfg 0.7, Heun50）：
  - strict：122.5k 0.4762 → **best 162.5k 0.4869** → 271k 0.4845（平台）
  - seen：best 45k 0.4764 → 271k 0.4288（持续走低）
- **结论：条件噪声增强被证伪**：
  1. seen -0.048（噪声直接破坏 g 条件保真度；g 是字身份唯一来源）
  2. strict 平台 0.4869 vs 同容量无噪声 sk3 线 0.5108@150k（差 -0.024）
  3. 机制：M/2 容量不足以"抗噪+拟合风格"兼得；鲁棒性收益没兑现
- 432k 前中止于 271k（后续平台无望超越 sk3 线），GPU 转产 base 数据。
- 次要发现：encode 32 进程会把训练 dataloader 饿死（5.2→0.62 sps）——CPU encode 必须限并发。

## 3. 实验 plan：v11_pretrain_Sp2_base（进行中）

- **配方**：Sp/2 (h512/d12/heads8, ~59M) + base 54,892 行 + 新书家表 (52×128) + 无噪声增强 +
  REPA 0.03 (dino cache base_v1, fp16 10.1GiB) + 12ch (canny0.3/skel0.8) + adaLN4 + cosine 400k + batch 192。
- **对照变量**（vs noise400k）：数据（fame3→base）+ 容量（M/2→Sp/2）+ 去噪声增强 + 书家表。非单变量 A/B，是"新数据线"run。
- **目标**：strict 突破 M/2 sk3 线 0.5108@150k；seen 突破 0.571；长期向 0.5680+ 推进。
- eval：in-mem eval 每 2,500 步（seen n=10 + strict n=50 + PNG + poster），无 daemon。
- adaln4 vs adaln12 A/B 仍留作后续单变量。

## 4. 本轮 infra 资产

- 数据：`assets/train_base_noaug.csv`（54,892）、`train_fame_tj_kxl.csv`（94,131，v4 增强）、
  `callig_id_map_base/tj.json`、`data/imgs/{calli_tongji_imgs,calli_tongji_sym,unicalli_chars}`、
  反相修复 `tools/fix_polarity.py` + `base_polarity_bad.csv`
- encode：`tools/encode_base_gpu.py`（vae_io 轮子，GPU 149 img/s，三阶段+std_expand）、
  `tools/cpu_encode_base.py`（CPU 多进程版，shard 512 即时落盘）、`tools/cpu_encode_v9.py`（git 历史找回）
- 派生图：`tools/gen_base_images.py`（skel3/canny 54,892+54,892，87s）、std skel PNG 5,627、key2uid
- 书家表：`tools/extract_dino_cls_base.py`（CLS 54,892×384）、`tools/pretrain_callig_emb_base.py`（SupCon 52×128）
- latent shards：final_latents_base_shards / aux_skel3_latents_base / aux_canny_latents_base 各 11 shards、
  std_skel3_latents_base(+expand 9 shards)、合并目录 `std_skel3_latents_base_full`（42 symlink shards）
- dino cache：`data/dino_cache/base_v1`（54,892, fp16 10.1GiB, 96s）
- eval：`src/eval/in_mem_eval.py`（真 in-mem：EMA 采样+decode+指标+PNG+poster，无 daemon，36s/次）
- 合并目录 + config：`src/train/configs/v11_pretrain_Sp2_base.json`
