# 47. 阶段总结（2026-09-10）：数据 / 模型设计 / 结果 / 实验线

状态: v10b 主线训练中（容量扫描 sty16→sty32→sty64）
前置: 44（stdskel_sp）、45（c41x 系列）、46（风格 token 设计）
覆盖期: 2026-09-08 ~ 09-10

---

## 0. TL;DR

- **主线模型**：`DiT-2Cond-Sp/2`（h512/d12，~59M）+ **std skel 骨架条件** + **41 书家冻结表** + **xattn 全 12 层骨架注入**。
- **最好结果**：`c41x` 212.5k 步 ssim **0.7638** / skel_iou **0.2227** / lpips **0.1778**（n=10 协议）。
- **当前推进**：新增"风格 token 每层注入"（46 号），三档容量 `n_style ∈ {16,32,64}` 正在串行从零训练。
- **数据侧**：fame v9 去噪线**已被证伪并隔离**（`final_imgs_fame_v9_REJECTED`），训练继续用 `fame v8` + `fame3`，**v8 全程只读未动**。
- **两条硬教训**：① 评估必须用**笔画级**指标（`skeleton_keep`），全图 `ink_delta` 会掩盖损伤；② 结构改动后**不能盲目 resume**（key 表面兼容、语义不安全）。

---

## 1. 数据现状

### 1.1 三条数据线

| 数据 | 规模 | 用途 | 状态 |
|---|---|---|---|
| `final_imgs_256`（原始 MCCD 落盘） | 51,322 | 噪声诊断基线 | 只读 |
| **`fame v8`**（`final_imgs_fame_v8`） | **51,822** | **v9c/v10 系列训练** | **只读未动 ✓** |
| `fame v9_REJECTED` | 51,321 | 去噪实验产物 | **已隔离，勿用** |
| `fame3`（`train_fame3_clean_v8.csv`） | **28,386** | **v10b 主线训练** | 41 书家精选子集 |

### 1.2 fame3：书家收紧 + 骨架条件换代

- 书家 **1013 → 41**（精选，样本/书家 ~700）→ 风格学习密度大增。
- 骨架条件从"GT 实例骨架 latent"换成 **std skel latent**（`build_std_skel1_latents.py`，
  用字帖标准字形生成）——关键动机：**实例骨架推理时拿不到**。
- 条件质量实测（`stdskel_gap.json`, n=3000）：`cos(std, GT_skel)=0.902`，
  `|std−GT|_nmse=0.197` → std 与实例骨架有 ~20% 缺口（笔势/个人风格），但远好于
  std 与最终目标（0.814 nmse）。**遵循度通路从"不可推理"变为可用。**

### 1.3 数据清洗线（30 号）——失败并终止

噪声探查 + 8 轮去噪迭代结论：

| 指标 | 原图 | v8 | v9(v3.7) |
|---|---:|---:|---:|
| `small_cc_n>0` | 34.69% | 0.00% | 0.00% |
| `edge_blob>0`（贴边大块/装裱） | 47.72% | 43.99% | ~43% |
| 综合有噪点 | 54.61% | 44.06% | — |
| **骨架丢 >5%（笔画断裂）** | — | **0** | **5,414（10.55%）** |

**结论**：毛刺(1–2px) 与细笔画(2–4px) **尺度连续重叠**，任何无参考的形态学/骨架方法
都必然两难（3×3 吃细笔画、2×2 削骨架、只 closing 无收益）。8 轮实验（删块/fill/横贯/
距离/剪枝/重建/膨胀/形态学）全部失败，**不是调参问题**。装裱黑边需**精确参考字形 bbox**
（现有 `std_glyph_bbox.json` 全是 `[0,255,0,255]` 全图框，无信息量）才能安全清除。
**v9 已隔离，v8 保留（笔画 100% 完整）。**

---

## 2. 模型设计（当前主线）

### 2.1 任务定义

条件生成：给定 **(g, c)** 生成书法字图。
- `g` = **std skel latent**（标准字形骨架，推理可得）
- `c` = **书家 id**（41 闭集）

部署口径（用户裁定）：**闭集 41 书家**，无未见书家兜底需求；CFG 理论值 `s=1.0`。

### 2.2 主干：`DiT-2Cond-Sp/2`

| 项 | 值 |
|---|---|
| latent | 32×32×4（VAE 8×） |
| hidden / depth / heads | **512 / 12 / 8**（"Sp" = S 的加宽版，1.8× S/2） |
| 参数量 | ~59M |
| 加宽动机 | S/2 (32.7M) 在 fame3 上 train loss 全 t 桶高且平、低 t 桶连训练原图都重建不动 → **逐 token 表达带宽不足**（欠拟合） |

### 2.3 两路条件接入

**c 路（书家）**：`LabelEmbedder(41+1)` + **SupCon 预训练 + 冻结**
- `tools/pretrain_callig_emb.py`：同书家正对 + `temp 0.07` + DINO CLS 质心锚定
  → embedding 继承真实风格语义
- pairwise cos **0.323 → 0.024**（塌缩修复）✓
- `--freeze-callig-table`：null token 拆独立可训练 Parameter
- callig `cond_drop`: 0.5 → **0.1**（冻结表已免疫塌缩；闭集任务先学会）

**g 路（骨架）**：`glyph_embedder_depth=2` + **注入机制演进**

| 版本 | 注入 | 说明 |
|---|---|---|
| v10b 基线 | 输入层 token-add | `glyph_inject_layers=0` |
| fame3-inject/deep | adaLN 4 层 | 固定 1:1 位置调制 |
| **c41x（主线）** | **xattn 全 12 层** | `ZeroCrossAttention`：Q=x 画布, K/V=g_tok(+16×16 sincos 位置) |
| **+ 风格 token（本轮）** | xattn 12 层，K/V=[g_tok; style] | 见 §2.4 |

### 2.4 本轮新增：风格 token 每层注入（46 号）

设计要点：**每个 x 位置（query）依据自身内容 + 空间位置，同时寻址"局部字形"与"书家风格"**
→ 风格调制**局部化**（表达结体差异：颜体宽博 / 欧体紧收）。

实现（`GlyphStyleCrossAttn`，`src/model/dit.py`）：
- context = `[书家化骨架 + 2D sincos 位置 ; 风格 token + 可学习 role]`
- 风格 token 由**共享投影**生成（各层复用）+ 可学习 `role` 促 N 个 token 分化
- `out_proj` **零初始化** → 初始恒等
- 开关：`--style-token-n`（0 = 回退旧 `ZeroCrossAttention`，完全兼容）
- 参数：42.39M → 43.99M（n=32，+1.6M）

**冒烟验证（1000 步，从零）**

| 项 | 实测 | 判读 |
|---|---|---|
| `glyph_injections.*.out_proj` | 12 层全部 0→非 0（层0 1.16e-3 / 层11 5.80e-4） | 通路真的在学 ✓ |
| style token intra-cos | **0.064** | < 0.3 不塌缩 ✓ |
| **有效秩** | **27 / 32** | 容量真实可用 ✓ |
| inter-cos（书家间） | 0.149（> intra 0.064） | 书家可分 ✓ |
| Diff / 显存 | 0.628→0.480 / 19.4G / 24G | 健康 ✓ |

---

## 3. 结果

### 3.1 当前最优（`eval_auto` 口径：n=10, cfg 0.7, ddim 50, 引擎 cpu_2sock_g）

| 实验 | step | ssim | skel_iou | lpips | mse |
|---|---:|---:|---:|---:|---:|
| c41x | 150k | 0.6970 | 0.1287 | 0.2288 | 0.3943 |
| **c41x** | **212.5k** | **0.7638** | **0.2227** | **0.1778** | **0.2755** |
| c41x_cos | 172.5k | 0.7642 | 0.2144 | 0.1816 | 0.2569 |
| c41x_cos_clean | 175k | 0.7510 | 0.1810 | 0.1900 | 0.2799 |

c41x @150k 的 `strict` 子指标（n=237）：`ssim_mean 0.5304`、`skel_iou_mean 0.0160`、
`lpips_mean 0.3816`、`mse_mean 0.8633`。

### 3.2 曲线趋势与解读（谨慎）

- **c41x 仍在上升**：150k→212.5k ssim 0.697→0.764。这是"Sp 加宽 + xattn 注入 + 41 冻结表"有效的证据。
- **S/2 系全部平台在 0.61–0.62**（旧口径）→ **容量瓶颈假设被支持**，这是加宽到 Sp 的直接依据。
- ⚠ **口径提醒**：44 号文档里的 `0.6216@105k` 是**另一套口径**（strict/旧协议），
  **不可与上表 0.76 直接比较**。上表 n=10、波动 ±0.01 量级，**单点差异（如 c41x_cos 0.7642 vs
  cos_clean 0.7510）不足以定论**。
- ⚠ c41x @150k 那个点 `elapsed_s=1706.8`（vs 212.5k 的 62.6s）→ 该点是在**训练高负载**下评测的，
  数值可信度低于其它点，勿单独引用。

---

## 4. 实验线状态（2026-09-10）

| 实验 | 配方 | 状态 |
|---|---|---|
| `c41x` | Sp/2 + 41冻结表 + xattn×12 + drop0.1 | 212.5k，主线，**仍升** |
| `c41x_cos` | c41x + callig 相关正则变体 | 172.5k |
| `c41x_cos_clean` | 上述 + 干净词表 | 175k |
| **`c41x_sty16/32/64`** | c41x + 风格 token 每层注入，n=16/32/64，**30k 从零** | **sty16 训练中（step ~4.8k, Diff 0.3551, 4.23 steps/s, Mem 19.58G）** |

容量扫描机制：
- config `src/train/configs/c41x_sty{16,32,64}.json`（唯一变量 = `style_token_n`）
- 串行链 `_sync_work/run_sty_scan_chain.sh`（含每实验 `cpu_eval_daemon` 切换 watch-root）
- **起点：从零训练**（不 resume，理由见 §5.2）
- 判据：结构不退化（follow-IoU3 / strict SSIM 不低于 c41x）+ 风格分化 + 有效秩随 n 的饱和拐点

---

## 5. 关键教训（重要）

### 5.1 评估指标必须"笔画级"，不能"全图级"

- ❌ `ink_delta`（全图墨量变化）**掩盖笔画级损伤**：一个 3px×50px 笔画被削成 1px，
  自身掉 67% 墨，但相对全图 8000px 墨仅 −1.25%。v3.7 全图只掉 0.21%，
  实际却有 **5,414 张字被削断笔画**。
- ✅ 核心安全指标 = **`skeleton_keep`（骨架保留率）**：`skeletonize(原图)` 的骨架像素中
  去噪后仍为墨的比例。<0.98 报警，<0.95 判定损伤。辅以 `half_width` 看削细程度。
- ❌ **不能依赖"读图确认"**：读图工具对已分析过的图会返回缓存，本次多处"视觉确认字形完整"
  因此是错的。**结论以量化指标为准，图像只作旁证。**

### 5.2 结构改动后 resume 旧 ckpt：表面兼容 ≠ 语义安全（46 号实测）

| 检查 | 结果 |
|---|---|
| 形状不匹配 | **0**（子模块命名/形状完全一致：`norm_x/norm_c/q/k/v/out_proj`） |
| missing | 仅 5 个新参数（`style_proj.*`、`style_role`）→ `strict=False` **静默通过** |
| 构建后 `out_proj` \|mean\| | 0（zero-init） |
| **load 后 `out_proj` \|mean\|** | **3.398e-02** ⚠ |

**风险**：随机初始化的 style token 会经**已训练的非零 `out_proj`** 直接注入残差流 =
给训好的模型叠加随机噪声。`initialize_weights()` 的 zero-init 发生在**构建时**，load 之后不再执行。
**处置**：容量扫描全部从零；若将来必须 resume，须在 load 后**显式重新 zero-init
`glyph_injections.*.out_proj`**。

### 5.3 其他

- **记忆化**：Sp 后期出现 `(g,callig)→图` 查表化（seen10 0.56 穿越 strict 0.52）→
  用 callig cond_drop 抑制（0.5→0.1）。
- **环境**：训练/评测统一用 **cu121**（`/opt/conda/envs/cu121/bin/python`）；
  base 环境 torch 1.13 + xformers 0.0.16 在 CPU fp32 无 attention kernel（曾误判为"8 头不兼容"）。
- **标注**：`gpu_eval_n` 不是 train.py 的 argparse 参数，config 里的值会被静默丢弃；
  评测样本数按 eval csv 行数确定。

---

## 6. 风险与待办

### 待验证
- [ ] 容量扫描：三组 Diff / strict / IoU3 曲线；有效秩随 n 的饱和拐点（16/32/64）
- [ ] 风格分化是否真的体现为生成图的书家差异（30k 偏短，可能需更长）
- [ ] c41x 与 S/2 在同协议下的 follow-IoU 对比（确认 Sp 的遵循度增益）
- [ ] c41x 训至平台（300k）后全量 GPU eval

### 风险
- **n=10 评测样本量小**，±0.01 波动内的实验对比不可下结论；需扩大 eval 集或多次采样。
- **装裱黑边未清除**（edge_blob ~43%）：需先重建精确参考字形 bbox 才能推进，否则重蹈 v9 覆辙。
- **风格 token 收益未证实**：冒烟只证明"通路在学、容量不塌缩"，**不等于生成质量提升**。

---

## 附：相关产物索引

| 类型 | 路径 |
|---|---|
| 文档 | `docs/system/44~47`（44 阶段、45 c41 三件套、46 风格 token、47 本篇） |
| 主线 config | `src/train/configs/v10b_stdskel_fame3_c41x.json` |
| 容量扫描 config | `src/train/configs/c41x_sty{16,32,64}.json` |
| 串行链 | `_sync_work/run_sty_scan_chain.sh` |
| 书家词表 | `src/utils/callig_map.py` + `5script/callig_id_map.json` |
| 预训练书家表 | `tools/pretrain_callig_emb.py` → `callig_emb_pretrained.pt` |
| resume 兼容性实测 | `_chk_resume_compat.py`（_ot_scratch，本地） |
| 笔画级评估 | `_ot_scratch/_eval_stroke_damage{,_full}.py` |
