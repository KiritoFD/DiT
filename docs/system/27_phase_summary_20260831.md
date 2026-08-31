# 阶段总结：标准字形 DINO embedding + 清洗数据 + ControlNet 管线（2026-08-31）

> 本文件是 2026-08-31 一轮工作的阶段性收口：代码变更、理论诊断、实验结果、
> 运行中任务、下一步计划。详细研究过程见 `25_dino_embed_direct.md` 与
> `26_dino_embed_pca_landing.md`。

---

## 1. 一句话总结

**这一轮完成了「char/glyph embedding 用标准字形 DINO 特征零参数直通」的完整闭环**：
研究（信号质量验证）→ 落地（PCA 降维表 + 冻结查表 + ln_only 直通）→ 数据清洗 →
训练验证（s28）→ 自动衔接下游 ControlNet（s29, 1px GT skel）。

## 2. 代码变更

### 新增
| 文件 | 作用 |
|---|---|
| `src/model/std_dino_embedder.py` | 标准字形 DINO 冻结查表嵌入器（`StdDinoCharEmbedder`）。加载 PCA 384 表，输入 glyph_id，`% chars_per_script` 取 char_id 查表。可训练参数=0（仅 CFG null token）。 |
| `src/train/configs/s28_std_dino_pretrain.json` | s28 预训练（标准字形 DINO + PCA + OT + 清洗数据）。 |
| `src/train/configs/s29_ctrl_gt_skel_1px.json` | s29 ControlNet 后训练（1px GT skel，基于 s28 主模型）。 |
| `docs/system/25_dino_embed_direct.md` | DINO cls/patch 直接注入研究（判别性评测）。 |
| `docs/system/26_dino_embed_pca_landing.md` | PCA 降维落地 + 训练 infra profile。 |

### 修改
| 文件 | 改动 |
|---|---|
| `src/model/dit.py` | 新增 `use_std_dino_char_embedder` / `std_dino_table_path` / `chars_per_script` 开关；`initialize_weights` 对 StdDino 跳过表重初始化（冻结）。 |
| `src/model/controlnet.py` | `load_main_model` 支持 std_dino 主模型加载。 |
| `src/train/train.py` | 新增 `--use-std-dino-char-embedder` / `--std-dino-table-path` / `--use-ot`；DINO init 跳过 std_dino；`opt.zero_grad(set_to_none=True)`。 |
| `src/train/train_controlnet.py` | 新增 std_dino 参数透传。 |
| `src/utils/latent_dataset.py` | 清洗数据相关（img_id 解析等）。 |
| `gradio_fame_local.py`、`s27_ctrl_std_skel.json` | 推理/配置微调。 |

---

## 3. 模型理论诊断

### 3.1 DINO CLS/Patch 作为 char/glyph embedding（docs/25）

**核心问题**：能不能用 DINO 的 CLS / Patch 特征**零可训练参数**直接注入（不做可训练
投影扭曲信号）？评测标准 = 外形一致性（外形像的字应接近）。

**方法修正**：像素级骨架 IoU 不适合当"外形相似度"真值（对笔画位置敏感，土/士 结构
相似却 IoU=0.07）。改用**判别性评测**：人工形近字对 vs 随机字对，AUC。

**结论（有数据支撑）**：

| 特征来源 | 分离度(形近-随机) | AUC |
|---|---|---|
| 真迹 DINO CLS 768（跨书体平均） | +0.007 | 0.50 |
| 真迹 DINO PatchMean | −0.012 | 0.50 |
| **标准字形 kai CLS** | +0.139 | **0.92** |
| **标准字形 kai PatchMean** | +0.088 | **0.94** |
| **标准字形 li CLS** | +0.163 | **0.96** |

- **DINO 特征本身外形一致性极好**（标准字形下 AUC 0.92-0.96）。
- **真迹跨书体平均不行**：书体主成分把所有字对余弦压到 ~0.83（AUC 0.50）。
- 关键认知：**问题在「真迹跨书体平均」，不在 DINO 特征本身**。

### 3.2 PCA 降维 vs 插值（docs/26）

DINO 768 → 模型 hidden 384，需零参数降维。实测判别性 AUC：
- 768 原始 0.820，线性插值 0.824，截断 0.833，相邻平均 0.824
- **PCA 投影 0.906**（最优，保留主方差方向去噪）

**采用 PCA**：在有真值覆盖的行上做 SVD，投影应用到整表。生成
`std_dino_char_table_384_pca.npy`（7026×384）。

### 3.3 训练 infra profile（docs/26）

torch 1.13（不可升级）下 profile 纯训练步（batch=192）：
- forward 83ms(23%) / **backward 269ms(73%)** / opt 11ms / ema 4ms
- batch 128→768，吞吐稳定 **~525 samples/s**（compute-bound，证实"提高 batch 吞吐不变"）
- xformers 已最优（比 eager 快 30%），bf16 已开
- **无显著工程 free lunch**；仅 `zero_grad(set_to_none)` +0.3%（已应用）

### 3.4 OT-CFM（最优传输）

flow_matching.py 已实现 Minibatch OT（`use_ot=True`，匈牙利重排 noise/data 配对）。
s28 已开启（`use_ot=true`）。代价 O(B³) scipy + 一次 GPU→CPU 同步，收益是轨迹不
交叉、速度场更平滑，无坏影响。

### 3.5 骨架消融：GT vs 标准字形（docs/17 §3）

| step | 1px GT SSIM | 3px GT SSIM | std-skel SSIM | std-skel IoU |
|---|---:|---:|---:|---:|
| 2500 | **0.7355** | 0.7288 | 0.4937 | 0.0154 |
| 5000 | **0.7376** | 0.7337 | 0.4940 | 0.0152 |

- **1px GT 优于 3px**（三项指标全优）
- **std-skel 彻底失败**（SSIM≈base，IoU≈0）：标准字形 skel 与字 ID 冗余，被门控
- **s29 采用 1px GT skel**

---

## 4. 数据清洗

清洗管线（用户执行）产出：
- `final_imgs_256_clean/`：1457 张修复图（denoise 1030 / crop 357 / invert 70 / big_glyph 327）
- `5script/train_fame_clean.csv`：51321 行（丢弃 1 张垃圾图）
- `5script/eval_fame_strict_clean.csv`：500 行（27 张 clean）

**latent 局部替换**（不全部重 encode）：
- 只 encode 1430 张 clean 修复图（29s），按 img_id 替换进 20 个旧 shards
- 生成 `final_latents_fame_clean/`，校验 1430/1430 替换、id 集合一致
- eval 不需要 latent（`prepare_eval_cache` 实时读 csv 的 image_path，自动用 clean 图）

---

## 5. 运行中任务

| tmux session | 内容 | 状态 |
|---|---|---|
| `s28_train` | s28 预训练（clean+PCA+OT） | **运行中**，GPU 97% |
| `s28_metrics` | s28 eval metrics daemon | 运行中 |
| `s28_to_s29` | 等 s28 收敛 → 自动拉起 s29 | 等待中 |

**s28 当前**（20:47 启动）：
- 数据 51,321（clean），Steps/Sec ~3.4，GPU 97%
- step 1000 eval: **ssim=0.4153, lpips=0.5225**（与 s21 真迹 DINO 同期 0.4204 相当）
- loss 已降至 0.48（初始 2.5）

**s29 自动拉起**：`_sync_work/_monitor_s28_to_s29.sh`
- 等 s28 训练进程结束 → 选 best eval ckpt（ssim 最大）→ 填 `s29 main_ckpt` → 启动 s29
- 加载已验证：`missing=0 unexpected=0`，s28 ckpt 完美加载到 s29

---

## 6. 下一步

1. 监控 s28 至收敛（early-stop: ssim_lpips, patience 5, min_steps 20000）
2. s28 收敛后 s29 ControlNet（1px GT skel）自动拉起，对比骨架消融曲线
3. 若 s28 效果好（超 s21），可考虑标准字形 DINO embedding 成为默认 char/glyph 条件

## 7. 复现

- PCA 表：`_sync_work/_build_pca_table.py`
- latent 局部替换：`_sync_work/_patch_clean_latents.py`
- infra profile：`_sync_work/_profile_train.py`
- s28→s29 监控：`_sync_work/_monitor_s28_to_s29.sh`
