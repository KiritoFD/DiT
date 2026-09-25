# 43 — v10b 五实验串行链：设计与运行状态

> 2026-09-07。基于 41 号梯度诊断（REPA 劫持 158% + callig 弱梯度 0.0013），
> 把方案 B/C（40 号）+ 42 号 3 个改进实验合并成 **5 个独立实验串行执行**。
> 上承 41（诊断）、42（实验设计）、40（B/C 方案）。

## 1. 五个实验（串行顺序）

| # | 实验名 | config | 关键改动 | 验证假设 |
|---|---|---|---|---|
| 1 | **v10b-repa** (方案B) | `v10brepa_strong.json` | `w_repa` 0.1→**0.5**，`repa_layers` (8,)→**(8,11)**，`repa_warmup` 2000 | REPA 强化到 v8e 最强配方 → base +0.003~0.008 |
| 2 | **v10b-inject** (方案C) | `v10binject.json` | `glyph_inject_layers` 0→**4**，`glyph_scale` 0.4→**0.6**，`glyph_drop` 0.1→**0.05** | 深层 ZeroAdaLN 注入 → 遵循度 0.57→0.62+ |
| 3 | **v10b-callig** | `v10bcallig_strong.json` | `callig_embed_dim` 128→**256**，`callig_proj_mode` **mlp**，`callig_scale_init` **1.5** | 弱梯度补容量 → 风格保真 |
| 4 | **v10b-norepa** | `v10bnorepa.json` | `w_repa` →**0** | 验证 REPA 劫持假说（41 号核心） |
| 5 | **v10b-shallow** | `v10bshallow_repa.json` | `repa_layers` (8,)→**(2,4)**，`w_repa` **0.05** | 浅层结构先验不过度干预 |

**对照三角**：#1 (w0.5) / #4 (w0) / 基线 (w0.1) → REPA 权重-质量曲线；
#3 是唯一动条件链结构的独立变量；#2 是遵循度杠杆。

## 2. 代码改动（dit.py 新开关，零破坏）

- `callig_proj_mode`: "linear"(默认) | "mlp" —— 两层 MLP（LN→Linear→SiLU→Linear）补 callig 容量
- `callig_scale_init`: 默认 1.0 —— callig_scale 初值可配（1.5 增强风格权重）
- train.py 透传两个新参数；冒烟全过（参数量 32.97M，+325K 正确）

## 3. 串行链机制（run_v10b_5exp_chain.sh）

```
v10bchain tmux 主链
  ├─ launch v10brepa:   tmux v10brepa 训练 + cpu_eval daemon watch→v10brepa_strong
  ├─ wait v10brepa:     while tmux has-session; sleep 120 (训练早停→会话退出)
  ├─ launch v10binject: 同 (daemon watch→v10binject)
  ├─ ... 串行 5 个 ...
  └─ 全部完成: tmux ls
```

**eval 正确性保障**：
- 每个实验启动时 `cpu_eval_daemon --mode pretrain_g --watch-root <对应 results_dir>`
- flat eval_auto_{step}.json（g=GT 骨架 latent, n=100, cfg 0.7）——与 v10b/v10a 基线曲线可比
- 训练 `auto_eval=false`（daemon 接管），早停读 daemon 写的 json
- 每步确认 model built OK + GPU 状态后才 wait

## 4. 运行状态（2026-09-07 13:0x 快照）

- **v10brepa（#1）训练中**：step 2550，Total 0.35（REPA w0.5 主导早期，预期中），
  2.77 sps，GPU 97% / 21.5G（预算内）
- **eval 已产出 step 1000** 第 1 点（daemon 正常）
- chain 正常 WAIT 中，日志在 tmux pane（`tmux capture-pane -t v10bchain`）
- 预计：每实验 8-12h（含早停），5 个串行 40-60h

## 5. 验收标准（每个实验）

1. base SSIM > 0.8476（v10a 记录）→ 该实验成功
2. 遵循度（n=30 GPU 矩阵，训练完补测）≥ 0.570 不降 → 骨架通路未破坏
3. REPA 梯度占比 < 50%（用 debug_v10b_gradients.py 复测）→ 劫持解除

## 6. 后续

- 5 实验全跑完 → 三臂汇总表（含 v10a/v10b/v10a-dino 基线）
- 最优者 + v10b 基线补遵循度矩阵（n=30）
- 出 v10 系列终结文档，定稿架构

存档：configs `src/train/configs/v10b{repa_strong,inject,callig_strong,norepa,shallow_repa}.json`，
chain `_sync_work/run_v10b_5exp_chain.sh`，冒烟 `_ot_scratch/smoke_v10b_callig.py`。