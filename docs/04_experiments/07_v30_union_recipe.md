# 07 — v30 Union：生成器+主干端到端联合训练（训练配方）

> 2026-09-30。动机：06 号文档定论两阶段断裂在接口（生成器 loss 不知道 decode 后笔画断没断）。
> 方案（用户提出）：把 latent-w7 生成器与 v26 主干**拼成一条可微链路，全解冻，只用最终图像
> 的 diff loss**——梯度从图像穿过 g 条件回传生成器，接口第一次被纳入优化。
> 无 char 两因子架构下 g 是字形唯一来源，断笔画必然推高 loss → 必然被修复。

## 1. 数据边界（无泄露版，2026-09-30 重训基准）

**泄露审计发现**：历史 `strict84` 的 84 个 (字,书家) 组合有 79-84 个在
`train_top10_style23*` 训练集里（`gt_skel_eval_strict84` 覆盖的 84 行全部散布在训练集）。
v30 起改用**组合级无泄露分割**（字符级剔除过度——strict 语义是组合级"该书家未写过该字"）：

| 文件 | 用途 | 行数 |
|---|---|---|
| `assets/eval_v13_strict84_aligned.csv` | 评测（84 行，与 GT shards 顺序对齐） | 84 |
| `assets/train_top10_style23_minusval_clean84.csv` | 生成器+联合训练 | 34,583 |
| `assets/train_top10_style23_clean84.csv` | v26 主干重训 | 38,398 |
| `assets/val_skelnet_clean84.csv` | 生成器验证 | 3,815 |

剔除 = eval84 的 84 个 (字,书家) 组合 + 对应 img_id（-166/-185/-19 行）。
**verified 零重叠**（组合与 img_id 双查）。seen20 / strict249 原 csv 不动（历史可比）。

## 2. 链路与实现（`tools/train_joint_g2img.py`）

```
g_std (shards_std) --[生成器 bridge Euler 8 步, hide-g]--> g_pred
      --[v26 条件增广: 噪声 0.3@0.5 + patch drop]--> v26 主干 (g 条件) --> flow loss vs 真迹 latent
```

- **生成器**：`DiT_2Cond(32², 4ch, depth6/h256/heads4, inject2 adaln, factorized_cat,
  glyph_vec_cond)` —— 与 `train_skelnet_dit.py` 逐参数一致；ckpt = w7 best（bridge+hide-g
  语义保持：g 输入恒零，z 从 g_std 起步），`--gen-lr 1e-4`
- **主干**：v26 配置逐项复刻（factorized_cat + **split LN**、glyph_inject_layers=4 adaln、
  glyph_vec_cond mean、callig 表冻结、cond_drop_all 0.1、cfg=1.0），ckpt = v26 0030000，`--bak-lr 5e-4`
- 双 EMA (0.9999×4)、warmup 2500 cosine、clip 1.0、batch 160（20.0G/24G，0.93 sps）
- 关键陷阱（都已踩过）：① `gen_sample` 不能挂 no_grad（训练要梯度穿采样链）；
  ② v26 ckpt 的懒参数 `null_embed` 需 `materialize_lazy_params` 先补再 load（否则 CFG
  uncond 被静默随机化）；③ 生成器构造必须与原训练**逐参数一致**（默认 arch），否则权重错位

## 3. 评测（与 06 号文档同尺，复用 `in_mem_eval` 三口径）

- `strict84`：g=GT 骨架（oracle 上界口径）
- `strict84_pred`：g=**EMA 生成器在线产出**（headline；两阶段 w7raw=0.6427 对照）
- `seen20`：重构口径
- 评测前 EMA 生成器对 `eval_v13_strict84_aligned.csv` 的 84 个 std 骨架在线采样 →
  落盘 `predskel_eval_strict84_joint/` shards → `_pred` 口径读表
- **泄露版参考**（`v30_union_LEAKED_20260930`，仅作管线验证，数字不可比）：
  4 样本试跑 strict84_pred ssim 0.7833 / **frag 1.08**（两阶段 5.36）/ lpips 0.12 /
  tgt_spec +0.322 / enrich 6.31×——方向信号极强，正待无泄露重训确认

## 4. 重训序列（本次执行）

1. **w7 生成器重训**（clean84）：`train_skelnet_dit.py` 同配方（depth6/h256/bridge/hide-g/w7 目标，
   40k steps, lr 2e-4, ema 0.999, es patience 按 val dice64）
2. **v26 主干重训**（clean84）：v26 配置 60k steps（GT 条件饱和快）
3. **v30-union 联训**（干净双 ckpt 起步）：本脚本
4. 全部数字只在 `strict84_aligned` 上报告；历史 0.53-0.64 数字带泄露边界，不再同表混排
