# 32 — v9c: skel 联合预训练（骨架直接进主干的对照实验）

> 2026-09-05 实现。动机与设计讨论见 `21_base_model_properties.md` §8：
> 特征探针证明 s21/v9a 主干内骨架信息严格为零（R² ≈ −2），实例结构 100% 依赖
> ControlNet 外挂分支；v8b 外挂路线 30k 后完全平台。v9c 让主干从 step0 起联训。

## 1. 实验定位

| | v8b / v9b（外挂 ControlNet） | **v9c（联训）** |
|---|---|---|
| 主干 | 冻结（v8a/v9a 预训练产物） | **解冻联训**（main_lr=3e-5，ctrl lr=1e-4 双参数组） |
| 骨架信息入口 | ctrl 分支 → zero-init 注入 | 同左，但主干**从 step0 学会读注入** |
| base 臂（无骨架） | = 纯冻结主干（恒定 0.518/0.525） | = 联训中漂移的主干（监视崩塌用） |
| 目标 | 不动 base，bolt-on 结构 | 检验"结构+风格联合训练"是否突破外挂天花板 |

## 2. 实现（复用已有路径，零新架构）

`train_controlnet.py` 的 warm-start 分支 + `--unfreeze-main true`：

```
train_ctrl_only=true   → load_main_model(v9a best) 硬断言加载
unfreeze_main=true     → 解冻主干 (豁免 freeze_char_table 有意冻结集)
main_lr=3e-5           → optimizer 双参数组: ctrl 1e-4 / main 3e-5
```

配置 `src/train/configs/v9c_skel_joint.json`：其余配方 = v9b
（fame_clean_v8 数据 + 1px 骨架 latent、flow+heun、REPA-early w=0.5 layers 8,11、
batch 128、max 50k、eval cfg 0.7 @ eval_fame_strict_clean_v8 每 2500 步、
early_stop patience 5 / min_delta 0.002 / min_steps 10k）。

## 3. 本次修复的正确性问题（smoke 15/15 验证）

1. **unfreeze-main 解冻 char 表 bug**（`train_controlnet.py`）：
   `main_p = [p if not requires_grad]` 会把 `freeze_char_table` 冻结的
   DINO 字符表（13.59M，含 null_embed 之外全部）一并解冻，语义锚在 main_lr 下漂移。
   修复：包装 ctrl 前快照"有意冻结"参数 id 集合，解冻时豁免。
   日志可验证：`保持冻结 13,588,608 (char 表等)`。
2. **torch.compile 存盘键前缀 bug**（两侧）：
   compiled 模块 `state_dict()` 键带 `_orig_mod.` 前缀，`startswith("main.")` 过滤
   全部失配 → `model`/`ema_model` 存成**空字典**（v9b 亦中招，其 ckpt 下游加载会静默丢权重；
   v9a/train.py 的 ema 键同样带前缀，`load_main_model` 内部有 strip 才没炸）。
   修复：存盘前 `_strip_compile_prefix`；resume 载入未编译原始模块 + 入侧剥前缀。
3. **resume 的 ema_model 错挂**：旧代码把 main-EMA 键载入**原始** ctrl 而非 ema_ctrl。
   修复：resume 按 raw/ema 分别载入对应模块。

## 4. Smoke 验证（`tools/diag/verify_v9c_smoke.py`，GPU 与 v9b 共享，batch 8 × 6 步）

15/15 PASS：
- 存盘结构 `model.*`/`ema_model.*`/`ctrl.*`/`ema` 各 159 键
- optimizer 双参数组，main lr = 0.3 × ctrl lr 精确成立
- **char 表 bit-exact**（联训 5 步未漂移，raw 与 ema 双份验证）
- main block 权重已更新（max|Δ|=2.3e-6，联训生效）；12 个注入 proj 离开零点（梯度到达）
- 重载零 unexpected；skel 条件前向 vs 无条件前向 max|Δout|=1.25（条件通路活）
- 首步 loss 0.337 → 与 v9a 收敛水平一致（零初始注入的不变量：初始=纯主干行为）

## 5. 拉起（自动点火）

```bash
tmux new-session -d -s v9c_waiter 'bash /root/Workspace/xy/DiT/_sync_work/run_v9c_skel_joint.sh'
```

守护每 120s 查 GPU，连续 2 次 <2000MiB（v9b 退出）+60s 稳定后：
自动挑 v9a best ckpt（eval_auto ssim 最高 = 0130000.pt / 0.5204）→ 启动联训。
日志 `/tmp/v9c_joint.log`，结果 `5script/results/v9c_skel_joint/`。

## 6. 评测与汇总（全自动，无需人工）

- 训练内 eval：eval_facade 每 2500 步双臂（ctrl 带 skel + base 纯主干），
  进程内算指标直接写 `eval_auto_ctrl_*.json`（早停同源，无竞态）
- 通用 daemon（tmux eval_supervisor）递归扫描消费
- `tools/eval/collect_v89_series.py` 已加 `v9c_skel_joint` 系列 →
  `build_master_results.py` 来源 7 自动并入 master 汇总

## 7. 判读标准（跑完后）

| 观察点 | 通过 | 警示 |
|---|---|---|
| ctrl.ssim vs v8b 峰值 0.7641（同 eval 协议） | >0.77 即架构假设成立 | ≤v8b 则外挂路线保留 |
| base 臂（纯主干） | ≥0.50 缓降可接受 | 像 v8e 那样崩向 0.15 → 降 main_lr 或去 REPA 复跑 |
| skel_iou vs v8b 的 0.35 | 显著↑ 说明结构容量进了主干 | 持平 → 瓶颈在 VAE latent 带宽 |
| 特征探针（跑 feature_analysis.py 于 v9c best） | 骨架 R² 转正 | — |
