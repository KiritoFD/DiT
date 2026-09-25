# 919 / 30 — 训练配方与实验链

> 快照日期 2026-09-19。训练入口 `src/train/train.py`，config 在 `src/train/configs/*.json`。

---

## 1. 当前标准配方（v13/v14 家族，"base recipe"）

| 类别 | 项 | 值 |
|---|---|---|
| **模型** | variant | `DiT-2Cond-S/2`（d12 h384 heads6，36.55M） |
| | 现代化 | `rms` + `swiglu` + `qk_norm=1` + `rope=1`（θ=100） + `sdpa` |
| | 条件融合 | `factorized_cat` + `glyph_vec_cond=true`（128, mean pool） |
| | g 注入 | `adaln`，4 层，`glyph_scale_init=0.6` |
| | 书家表 | `freeze_callig_table=true`，128 维 |
| | char 条件 | **`no_char_cond=true`** |
| **扩散** | 类型 | `flow` matching |
| | t 采样 | `logit_normal`（mean 0, std 1） |
| | 采样器 | `heun`，`shift=1.0` |
| **优化** | lr | `1e-4`（full-ft 阶段 5e-5） |
| | schedule | `cosine`，warmup 3000，`min_lr_ratio=0.1` |
| | weight decay | `0.02`（v13_wd01/fullft：**0.1**） |
| | batch | `360`（S/2 @ 4ch） |
| | EMA | `0.9999`，interval 4 |
| **正则** | cond drop | `all=0.1` / `one=0.0` |
| | **glyph drop** | **`0.0`** ← 见 `20_model.md` §2.5 警告 |
| | REPA | `w_repa=0.03`，layer 8 |
| **数据** | csv | `assets/train_50k_v2.csv`（50,786） |
| | latents | `data/50k/shards_img` + `data/50k/shards_std` |
| | preload | true，16 workers |
| **评测** | in-mem | `seen:20 + strict:250`，cfg 0.7，50 步 |
| | 频率 | `gpu_eval_every=2500` |

### 1.1 长 epoch 重构（2026-09-18/19）

**问题**：`DataLoader` 的 epoch 由 `dataset 长度 / batch` 决定（50,786/360 ≈ 141 步），
而 ckpt/eval 每 5000 步一次 → **loader reset 与 ckpt 点不对齐** →
每 141 步出现一次 re-shuffle 抖动（实测 ~11% 的步进时间锯齿）。

**解**：新增 `LongEpochDistributedSampler`（`src/utils/samplers.py`），
令 **epoch = 固定步数 = `ckpt_every`**（= `epoch_steps` = 5000）→ loader reset 与 ckpt/eval 完全对齐。

**`--resume-full` 步骤推断修复**：原先从 **文件名** 解析步数（陷阱：非 5000 整数倍的文件名）；
现在**读 ckpt 顶层 `train_steps` 字段**。测试：`_review/test_longepoch.py` 通过。

⚠ **`--fresh-scheduler` 语义 bug（已修）**：它现在**也会把步骤计数器归零**，
且发生在调度器构建**之前**。因此：
> **`--fresh-scheduler` + `--resume-full` 会同时归零步数** ——
> 若 `max_steps` 写的是绝对步数，绝对不能一起用。

v14 的解决方案：**不用 `--fresh-scheduler`**，把 `max_steps` 写成**绝对步数**
（stage3: 200000 = 160000 + 40000），让 ckpt 里的调度器按原 cosine 继续衰减。

---

## 2. 实验链（按时间顺序）

### 2.1 史前（v10/v11）：确立"标准骨架条件"路线

| run | 关键 | 结果 |
|---|---|---|
| v10a/v10b | 骨架条件首次引入（`skel_as_glyph_cond`） | 确立 **+0.22** 的收益 |
| `v10b_stdskel_fame3_c41x_cos_e` | **历史最高分线**：Sp/2 + xattn12 + 旧数据(28,385) | strict **0.5680**@360k，seen 0.7630 |
| `v11_pretrain_M432_adaln4_*` | M/2（47M）+ adaLN4 注入 | 152.5k strict **0.5656** 见顶 |
| `v11_pretrain_M432_adaln4_sym_noise400k` | 条件噪声增强 | seen **−0.048**（**证伪**） |
| `v11_pretrain_Sp2_base_wz` | 白底归零 | 见 `20_model.md` §3.2（**放弃**） |
| `v11_pretrain_S2_v8_aux02` / `v11_pretrain_M432_v8_auxc03s08` | 12ch aux 权重实验 | 未突破 |
| `v11_struct-loss` | 实例骨架结构损失 | 未突破 |

### 2.2 v12 系列：干净数据集 + 模型容量单变量

背景：数据从 54,892 → **px60 28,569**（见 `10_data.md` §1.1），
并新增 `eval_skel_latent_shards_dir` 修复 strict g 全零的致命 bug。

| run | 单变量 | 模型 | strict | 结论 |
|---|---|---|---|---|
| `v12_pretrain_S_cat_fame_kxl_tj_px60` | **基线**（S/2 + cat + glyph_vec） | S/2 | 最佳 **0.5603**@95k（n=50） | 100k 回落 0.5599 |
| `v12_d8` | depth 12→**8**（XS/2） | XS/2 | **0.5337**@100k | **−0.026 → 浅层路线失效** |
| `v12_w320` | width 384→**320**（S320/2） | S320/2 | 0.5139@40k（**未跑完即停**） | 早期落后基线 ~0.03 |
| `v12_12ch` | +8 通道 aux 扩散目标 | S/2 | 0.4993@85k | **−0.06 → 证伪** |
| `v12_xattn` | adaln → **xattn** 注入 | S/2 | 0.5146@25k（**早停**） | 未跑完，无结论 |

⚠ v12 系列全部是 **n=50** strict，**不能**与 v13/v14 的 n=250 直接比（差 +0.017）。

### 2.3 v13 系列：换 50k 数据集 + wd

| run | 单变量 vs `v13_base_50k` | 结果 |
|---|---|---|
| `v13_base_50k` | **新基线**（45 书家 / 50,786 行） | in-training strict 全程 **n=50**：0.5450@100k → 0.5547@155k；155k 全量 **n=250 = 0.5703**（seen 0.7580，`--eval-only`） |
| `v13_base_50k_wd01` | wd 0.02 → **0.1** | strict **n=250**；同口径裁决见 `40_results.md` §1.2：85k 前与 base 无差异，85k 后领先（峰值 +0.0135@110k）、125k 收窄到 +0.006；**停在 125k 等 GPU，终审未完成** |
| `v13_styletok`（失败） | 冻结主干 + xattn + 新随机权重 + lr 3.9e-5 | strict 峰值 0.5711@170k 后跌到 0.5495@195k；**seen 从 0.758 崩到 0.604（−0.154）** |
| `v13_12ch_post` | 12ch post-hoc | 未突破 |

**`v13_styletok` 失败的诊断（重要）**：
它**同时**改了 4 件事（注入路径 xattn + 新增随机参数 + 冻主干 + 极低 LR）。
用户裁定：**这是训练协议错误，不是 style token 机制不行** → 需要重新公平测试。

### 2.4 v14 系列：87 (书家×书体) 风格表

三阶段管线（`scripts/ops/run_v14_style87_3stage.sh`）：

```
Stage 1 (离线, 不需 GPU 训练)
  src/utils/callig_script_map.py        → assets/callig_script_id_map.json  (87 对)
  tools/extract_dino_cls_50k.py         → assets/dino_cls_50k.npz (76M)
  tools/pretrain_callig_script_emb.py   → assets/callig_script_emb_pretrained.pt
                                          （层级 SupCon 3000 步，sibling 0.3 + anchor 0.5）

Stage 2  v14_style87_stage2.json   冻结表, 训主干, max_steps 250k
Stage 3  v14_style87_stage3.json   冻结主干, 只微调表 + 锚定 λ=0.005, max_steps 200k
         ★ wd 0.1 → 0.02（只训 11K 表参数；wd 0.1 会把向量往零缩、
           与"锚定往预训练拉"方向冲突，正则化职责交给 λ）
```

| run | 结果 |
|---|---|
| `v14_style87_s2` | 60k: 0.5529 / 80k: 0.5602 / **160k: 0.5703**（seen 0.7664，gap 0.196） |
| `v14_style87_s3` | 162.5k: **0.5764**（seen 0.7698）→ 峰值；165k 后平台 0.5750–0.5756（缓降，187.5k 止） |
| `v14_style87_fullft` | 从 s2@60k 全解冻：strict 0.5529→0.5683@125k 缓升，**160k 终点 0.5691 / seen 0.7642 / gap 0.195** —— 未超 base 0.5703，gap 三代最差 → **裁决：全解冻不是答案**（09-19 21:41 跑完收官） |

**三代同期对照**（strict / seen，n=250）：

| | @115k | 说明 |
|---|---|---|
| `v13_wd01` | 0.5651 / 0.6794 | 45 表 |
| `v14_s2` | 0.5657 / 0.7032 | 87 表，**冻结** |
| `v14_fullft` | 0.5674 / 0.6973 | 87 表，**全解冻** |

→ 87 表带来 **+0.0006 ~ +0.0023**，在 n=250 的 SE(≈0.003) 内 —— **不显著**；
且已查明两代 SupCon 表的 DINO 锚定因特征提取 bug 从未生效（`10_data.md` §3.4），
表的几何是纯标签驱动的 —— "表结构不是瓶颈"的结论进一步加固。

### 2.5 v15 系列：多模态风格 K=4 从头矩阵（当前）

**为什么从头而不是 resume**（2026-09-19 用户裁定，推翻了最初照设计稿跑的
`--train-only-style` 隔离版——该版跑了 ~20 分钟即停，日志留档
`logs/v15_series/v15_train_20260919-221005.log`）：注入方式变了 resume 不公平 ——
① adaLN 条件几何断层（旧 128 维 callig 几何是纯标签的，新 pooled-384 K-Means 几何
完全不同，冻结主干 + 可训 cond_fusion 大概率把新容量压回旧子空间）；
② 冻结主干从未见过"带风格的 g_tok"，不会读新注入通路（styletok 前车之鉴）；
③ 判据不可归因（ratio 上不去分不清是容量还是表达）。

| run | config | 注入方式 | batch/lr | 说明 |
|---|---|---|---|---|
| `v15a_multistyle_k4` | `v15a_multistyle_k4_pool.json` | K=4 池化 → adaLN | 360 / 1e-4 | 消融基线：纯 token 容量 |
| `v15b_multistyle_k4` | `v15b_multistyle_k4_ca.json` | a + CA 书家化骨架 | 360 / 1e-4 | 用户原设计核心 |
| `v15c_multistyle_k4` | `v15c_multistyle_k4_ctx.json` | a + 每层 xattn ctx 可见风格 | **260 / 7.2e-5** | xattn 显存限制 batch；同 step 样本数比 a/b 少 28% |

共享：87×4 可训表（DINO K-Means 初始化 + **λ=0.01 mean 锚定**）、其余 = v14_s2 配方
（REPA 0.03 / wd 0.1 / warmup 3000 / cosine / EMA）、`glyph_drop_prob=0.1`（双轴 CFG 前提）。
各 **150k 步 cosine 收尾**（用户裁定；⚠ 与 v14_s2 的 250k schedule 在 150k 处 LR 不同，
对比时声明）。串行 `bash scripts/ops/run_v15_multistyle.sh serial`（a→b→c，~26h），
判据 = strict/seen vs v13_base/v14_s2 同口径 + 最终 `ratio_style`。

---

## 3. 训练工具链

| 脚本 | 用途 |
|---|---|
| `scripts/ops/run_v14_serial.sh` | v14 串行排队 |
| `scripts/ops/run_v14_style87_3stage.sh` | v14 三阶段一键 |
| `_sync_work/_launch_v14.sh` / `_v14_stage3_inner.sh` | 远端启动包装 |
| `_sync_work/gpu_mon.sh` | GPU 监控 |
| `_review/ckpt_info.py` | 读 ckpt 元信息（步数/指标/配置） |
| `_review/print_v14_cfg.py` | config 打印（含注释） |
| `_review/compare_tables.py` | 两代风格表对比（cos / 有效秩） |
| `_review/smoke_v12.py` / `smoke18_19.py` | 冒烟测试 |
| `tools/eval_diversity.py` | **`ratio_style` 等多样性指标**（关键工具） |

---

## 4. 训练侧的坑（全部踩过）

| # | 坑 | 后果 | 修法 |
|---|---|---|---|
| 1 | **`factorized_cat` 不在 drop guard 元组里** | 全程零条件 dropout → `null_embed` 未训 → CFG 不可信 | 已修（2026-09-17） |
| 2 | **`materialize_lazy_params` 在 `load_state_dict` 之后** | `null_embed` 静默丢弃、恢复随机 | 调到之前 |
| 3 | **`aux_zero_white` 未注册 argparse** | config 值静默丢弃 → 生成图发黄 | 已注册 |
| 4 | **`repa_layers` 未注册 argparse** | config 里的值静默忽略 → 走硬编码 `(8,)` | **仍有踩**（v12 注释记录过） |
| 5 | **`--fresh-scheduler` 不归零步数** | LR 曲线错位 | 已修（同时归零） |
| 6 | **`--resume-full` 从文件名推步数** | 非 5000 整数倍文件名出错 | 改读 ckpt `train_steps` |
| 7 | **epoch 与 ckpt 点不对齐** | 步进时间 ~11% 锯齿 | `LongEpochDistributedSampler` |
| 8 | **旧 eval 集含 5 个 0 样本书家** | `y_callig_embedder(806)` 而表 46 行 → **CUDA 越界崩** | 换 `eval_v13_*` |
| 9 | **dino cache 代际不兼容** | REPA teacher 静默错位 | 每代重建 |
| 10 | **新增 buffer 破 ckpt** | RoPE 表进 state_dict 会报 missing key | `persistent=False` |
| 11 | **架构演进 resume 时形状失配** | `strict=False` 只容忍 key 差异——模型侧崩 RuntimeError、EMA 侧（OptimizedModule）同样崩 | `drop_shape_mismatched`（剔除+并入新模块保持可训，模型/EMA 两处都过）；但**注入方式变了仍应从头**（见 §2.5） |

⚠ **第 3/4 条是同一类"静默丢弃"家族**（config 键未注册 → 值被吞）。
**纪律**：新增任何 config 键必须同时注册 argparse，并加断言/打印。

---

## 5. 训练成本与吞吐

| 项 | 值 |
|---|---|
| S/2 @ batch 360 | **4.14 step/s**（约 3.6M 图/小时） |
| 100k 步 | ≈ **6.7 小时** |
| M/2 @ batch 240 | 5.28 step/s（FLOPs 大但 batch 小） |
| xattn 注入 | **+33% 算力** |
| 完整一次 v14 三阶段 | ~20 小时 |

详细硬件/环境见 `50_infra.md`。