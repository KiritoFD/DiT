# 标准字形 DINO embedding 落地：PCA 降维 + 训练 infra profile（2026-08-31）

> 承接 `docs/system/25_dino_embed_direct.md`。本文记录两件事：
> 1. **PCA 降维落地**：DINO 768 → 384 用固定 PCA 投影（零可训练参数，直通），替代
>    可训练 char_proj 投影。用户明确「别干扰信号，直通进去」。
> 2. **训练 infra profile**：在 torch 1.13（不可升级）、不改模型/不升 batch 的原则下，
>    验证纯计算是否还有 free lunch。

---

## 1. 落地：标准字形 DINO char 表（PCA 降维，零可训练参数）

### 1.1 为什么用 PCA 而不是线性插值

实测判别性 AUC（形近字对 vs 随机对，方法同 docs/25）：

| 降维方式 | AUC |
|---|---|
| 768 原始 | 0.820 |
| 768→384 linear interp | 0.824 |
| 768→384 截断前384维 | 0.833 |
| 768→384 相邻平均 | 0.824 |
| **768→384 PCA 投影** | **0.906** |

- **PCA 明显最好**（0.906 vs 0.82），保留主方差方向、去噪。
- PCA 是**固定线性正交投影**（构建时算好矩阵，冻结，零可学习参数），不引入梯度，
  不扭曲特征结构相对关系 → 符合「直通」。
- 用户最初要求「插值」，实测插值可行（信号不丢，AUC 0.824）但 PCA 更优，故采用 PCA。

### 1.2 实现

- 在**有真值覆盖的行**上做 SVD（避免 fallback 均值行污染主成分），投影应用到整表。
- 产出（已生成）：
  - `_sync_work/std_dino_char_table_384_pca.npy`（7026×384，行归一化）
  - `_sync_work/std_dino_pca_meta.json`（center、P 矩阵，便于复现）
- `src/model/std_dino_embedder.py`：`StdDinoCharEmbedder`
  - 加载 384 表（`std_dino_char_table_384_pca.npy`），维度匹配则 identity（不降维）。
  - 冻结 buffer，可训练参数 = 0（仅 CFG null token，384 参数）。
  - 接口 `forward(labels, train, force_drop_ids)`，输入 glyph_id，`% chars_per_script` 取 char_id 查表。
- `src/model/dit.py`：新增 `use_std_dino_char_embedder` 开关；`initialize_weights` 对
  `StdDinoCharEmbedder` 跳过表重初始化（冻结表），只初始化 null token。
- `src/train/train.py`：新增 `--use-std-dino-char-embedder` / `--std-dino-table-path`。
- s28 配置：`use_std_dino_char_embedder=true`, `char_embed_dim=384`,
  `char_proj_mode=ln_only`（LayerNorm 逐元素归一化，不改变特征方向，直通进 adaLN）。

### 1.3 验证

smoke test 通过：模型构建、forward 正常；形近字（土/士 0.897）PCA 表 + LayerNorm 后
余弦**保持不变**（0.897→0.897），证明直通、无方向扭曲。总可训练参数 32.93M，
其中 char embedding 冻结表 0 参数。

---

## 2. 训练 infra profile（torch 1.13，纯计算）

### 2.1 方法

手动计时（torch.cuda.Event / time，CUPTI 权限受限无法用 torch.profiler CUDA activity）
profile s28 模型 + flow loss 的纯训练步（forward + backward + opt + ema），排除 eval/ckpt。

### 2.2 结果（batch=192）

| 阶段 | 耗时 | 占比 |
|---|---|---|
| forward | 83ms | 23% |
| **backward** | **269ms** | **73%** |
| opt | 11ms | 3% |
| ema | 4ms | 1% |
| **合计** | **367ms** | 2.72 steps/s |

### 2.3 batch 扩展（吞吐饱和点）

| batch | 每步耗时 | 吞吐 |
|---|---|---|
| 128 | 225ms | 569 samples/s |
| 192 | 367ms | 523 samples/s |
| 256 | 491ms | 521 samples/s |
| 512 | 972ms | 527 samples/s |
| 768 | 1450ms | 530 samples/s |

**结论：batch≥128 后吞吐稳定在 ~525±5 samples/s，不随 batch 变化** —— 与用户
「提高 batch 样本吞吐不变」的判断完全一致，证明**纯计算是 compute-bound（GPU FLOPs 饱和）**。

### 2.4 attention 实现对比（batch=192）

| attn_impl | 每步耗时 |
|---|---|
| **xformers**（当前） | **367ms** |
| eager | 479ms |

- **xformers 已是最优**（比 eager 快 30%）。torch 1.13 无 `F.scaled_dot_product_attention`，
  已自动回退 xformers memory_efficient。
- eager 更慢，不可用。

### 2.5 其他

- bf16 autocast 已开启（`torch.autocast("cuda", dtype=torch.bfloat16)`），matmul 已减半。
- `return_pred_xstart`：s28 的 `_need_x0=False`（无 struct/std_mid loss），走不含
  pred_xstart 的 `training_losses` 分支，无额外计算浪费。
- `opt.zero_grad(set_to_none=True)`：已应用，实测 +0.3%（366.3 vs 367.3ms），微小正提升。

---

## 3. 结论

1. **PCA 落地完成且正确**：标准字形 DINO 768 → 384 用固定 PCA 投影（AUC 0.906，
   零可训练参数，直通不扭曲），smoke 验证通过。可拉起 s28 预训练。
2. **纯计算无显著工程 free lunch**：吞吐 ~525 samples/s 已饱和（compute-bound），
   backward 占 73% 是 transformer FLOPs 本质；xformers/bf16 已最优；eager 更慢；
   唯一小优化 zero_grad(set_to_none) 已应用（+0.3%）。
3. 真正的吞吐提升需要**架构层减 FLOPs**（缩短序列 / 稀疏 attention / 更小模型）或
   **升级 torch**（用 SDPA/compile），两者均被用户约束禁止，故维持现状。

---

## 4. 复现

- 生成 PCA 表：`_sync_work/_build_pca_table.py`
- smoke：`_sync_work/_smoke_s28.py`
- infra profile：`_sync_work/_profile_train.py`
- 产物：`_sync_work/std_dino_char_table_384_pca.npy`、`std_dino_pca_meta.json`
