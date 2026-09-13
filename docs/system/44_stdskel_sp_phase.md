# 44 — std skel + Sp 模型阶段：数据/条件/模型三换代

> 2026-09-09。覆盖 43 号（五实验链）之后的阶段：v10 主线从"fame_v8 + 实例骨架 g
> + S/2"全面切换到"**fame3 + std skel g + Sp 加宽 + xattn 注入 + 41 书家冻结表**"。
> 本阶段未写文档（一直在实验），此处补全。

## 1. 为什么换代（动机链）

1. **实例骨架不可推理**：v10 系 g 条件用 GT 实例骨架 latent —— 训练/评测可用，
   但真实生成时**没有目标样本的骨架**（只有要写的字）。遵循度 0.57 建立在一个
   推理时拿不到的条件上。
2. **std skel**（`build_std_skel1_latents.py`）：用**标准字形**（字帖骨架）生成
   骨架 latent `std_skel1_latents_fame3_v8` 作 g —— 推理时可得的骨架条件。
   - 信息缺口实测（`stdskel_gap.json`, n=3000）：cos(std, GT_skel)=0.902，
     |std−GT|_nmse=0.197 —— std 与实例骨架有 ~20% 差距（笔势/个人风格），
     但远好于 std 与最终目标 (0.814 nmse)。
3. **fame3 数据**：`train_fame3_clean_v8.csv`，28,386 样本，书家 1013→**41**
   （精选书家子集，样本/书家 ~700，风格学习密度大增）。
4. **41 书家冻结预训练表**：`callig_emb_pretrained.pt` + `freeze_callig_table:
   True` + `callig_id_map.json` —— 书家 embedding 预训练后冻结（呼应 41 号
   callig 链弱梯度诊断：小书家集直接冻结，防过拟合 + 稳定风格向量）。
5. **欠拟合诊断 → Sp 模型**（2026-09-08）：S/2 (32.7M) 在新任务上 train loss
   全 t 桶高且平，低 t 桶连训练原图都重建不动 → 逐 token 表达容量不足。
   GT-g 同深度模型能解抄写（follow-IoU3 0.57），排除深度稀释，缺的是
   骨架→墨映射带宽 → **加宽不加深**：`DiT-2Cond-Sp/2` = h512/d12/heads8, ~59M
   (1.8× S/2)，g 残差流存活系数 1.72 已验证。

## 2. 注入机制演进

| 版本 | 注入 | 说明 |
|---|---|---|
| v10b 基线 | 输入层 token-add | glyph_inject_layers=0 |
| fame3-inject/deep | adaLN 4 层 | glyph_inject_layers=4 |
| **c41x（当前）** | **xattn 全 12 层** | `glyph_inject_mode: xattn`，ZeroCrossAttentionInjection 逐层交叉注意力 + `glyph_embedder_depth: 2` |

## 3. 变体消融矩阵（8 个 config）

| 实验 | 模型 | 书家 | 注入 | drop(-all/one) | 状态 |
|---|---|---|---|---|---|
| fame3 base | S/2 | 1013 | 输入层 | 0.1/0.4 | best 0.6185@50k (25 点) |
| fame3 d01 | S/2 | 1013 | 输入层 | **0.05/0.05** | 0.596@17.5k (7 点, 早停) |
| fame3 deep | S/2 | 1013 | adaLN 4 | 0.05/0.05 | 0.6117@30k (13 点) |
| fame3 mid | S/2 | 1013 | adaLN 4 + w_std_mid | 0.05/0.05 | 1 点, 弃 |
| fame3 di | S/2 | 1013 | 深层 | 0.05/0.05 | 弃 |
| **c41** | **Sp/2** | **41+冻结** | adaLN 4 | 0.1/0.4 | 3k 步, 快速放弃 |
| **c41d01** | Sp/2 | 41+冻结 | adaLN 4 | **0.05/0.05** | 15k, 被 c41x 取代 |
| **c41x（当前主线）** | Sp/2 | 41+冻结 | **xattn 12 层** | 0.05/0.05 | **107.5k 步仍在训** |

公共配方：w_repa **0.03**（41 号诊断后轻量化）、flow heun、logit_normal、
batch 128（c41x）/192（早变体）、lr 1.5e-4、max 300k（c41x）、早停关（c41x）。

## 4. 结果（⚠ 新 eval 协议：eval_seen_v10.csv, n=10, **不可与 v8 的 0.84 直接比**）

| 实验 | best ssim | @step | 趋势 |
|---|---|---|---|
| fame3 base (S/2) | 0.6185 | 50k | 平台后微降 |
| fame3 deep | 0.6117 | 30k | 仍缓升 |
| **c41x (Sp+xattn)** | **0.6216** | **105k** | **仍升（97500→105000: 0.598→0.622）** |

c41x 最新点（105k）：lpips 0.290、mse 0.584、ssim_p10 0.497 / p90 0.725。
skel_iou 0.060（新 metric 口径；注意 GT-g 抄写任务 follow-IoU3 0.57 说明
骨架→墨通路本身可解，std skel 的缺口在逐字细节）。

**关键对比**：c41x > base (+0.003) 且曲线仍在升 —— Sp 加宽 + xattn 注入 +
41 冻结表有效；S/2 系全部平台在 0.61-0.62，容量瓶颈假设被支持。

## 5. 本阶段结论

1. **std skel 可行**：0.90 余弦的条件质量，遵循度通路从"不可推理"变为可用
2. **欠拟合确认**：S/2 容量不足是平台根因之一（呼应 41 号梯度诊断的主干弱更新）
3. **Sp + xattn 是正确方向**：唯一仍在上升的曲线
4. **41 冻结表**：书家小集的标准解法，呼应 callig 弱梯度诊断

## 6. 下一步

- c41x 训到 300k（max）或平台 → 全量 GPU eval + 遵循度矩阵
- 与 S/2 的 follow-IoU 对比（同协议），确认 Sp 在遵循度上的增益
- 若 c41x 平台：glyph_embedder_depth 2→3、xattn 加 KV 投影正交初始化、
  或 std skel gap 弥补（std + 实例骨架混合训练）

存档：`assets/results/stdskel_gap.json`（缺口分析）、各 results 目录 eval_auto json。