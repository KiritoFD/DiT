# 注入改造 · 下一阶段实验计划（2026-09-23）

> 目标：验证 **`docs/922/80_injection_redesign.md`** 的改动 1 / 改动 2
> 能否把"风格进不了像素"（D3 严格下界全负）治好。
>
> 本文件只讲**怎么跑**：跑什么、跑多久、要什么数、什么条件下停。

---

## 0. 硬件约束（决定了整个计划形态）

```
GPU = 单张 RTX 4090 (24 GB)   ← 只有一张，没有第二张
```

**→ 必须串行。** 三个实验排队跑，不能并行，不能"各占一张卡"。

⚠ 注意 `global_batch_size=360` 在单卡下是 **每个 micro-step 真读 360 个样本**
（`world_size=1`，无梯度累积拆分），显存 20.2 GB / 24 GB，余量约 3.8 GB。
这也是不能同时跑两个进程的另一个原因。

---

## 1. 三个实验（串行）

| # | 配置 | 变量 | 参数量 | 目的 |
|---|---|---|---|---|
| **E0** | `v17_s2_s2z_baseline.json` | 无（纯基线） | 36,445,457 | **对照锚点** |
| **E1** | `v17_s2z_ada1_ln.json` | 改动 1（LN + 自动标定 gain） | 36,495,762 | 单变量：只换"风格通路" |
| **E2** | `v17_s2z_ada12_ln_rank64.json` | 改动 1 + 改动 2（+ 低秩 adaLN 支路） | 39,316,242 | 加独立调制支路 |

三份配置**逐字段一致**，唯一差别就是 `style_ln` / `style_gain_init` /
`style_y_over_t_init` / `style_ada_rank` 这四个字段。

### E0 是必须的，不是可选的

`v17_s2_s2z_baseline` 的 `max_steps=40000 / lr=5e-4`，**这个组合历史上从没跑过**
（历史基线都是 `250k / 1e-4`）。所以：

- E0 的作用 = 给出"**在当前这个短程高 LR 配方下**，风格本来是什么水平"的锚点；
- 没有 E0，E1/E2 的任何数字都无法判断是"改造有效"还是"配方变了"。

---

## 2. 步数与 ETA

### 2.1 ★ 实测速度（2026-09-23 01:07，真实配置）

**必须用真实配置测速**（`preload=True` / `num_workers=8` / `gbs=360` /
`in_mem_eval=True`），从 `v17_s2z_ada12_ln_rank64.json` 复制后**只改 `max_steps`**：

```
preload=True  num_workers=8  gbs=360  in_mem_eval=True
[style_branch] gain 自动标定: ‖t_emb‖=0.928 ‖style_out‖=19.197 -> style_gain=0.0484
Trainable Parameters: 39,316,242
step=5   Steps/Sec: 0.22   ← 首步含 JIT 编译，不算数
step=10  Steps/Sec: 4.14
step=15  Steps/Sec: 4.12
step=20  Steps/Sec: 4.15
step=25  Steps/Sec: 4.15
step=30  Steps/Sec: 4.16   ← 稳态
[in-mem-eval] step 30 done in 145s: seen ssim=0.3635 | strict ssim=0.3582
总用时: 238 秒
```

| 阶段 | 实测 |
|---|---|
| JIT 编译 + preload（一次性） | ~73 s |
| **纯训练稳态** | **4.15 steps/s**（0.24 s/步） |
| in-mem-eval（seen + strict） | 89 s / 次 |
| 30 步 + 1 次 eval 总计 | 238 s |

GPU 状态健康：**利用率 59%、功率 130 W**（对比错误基准里的 0% / 61 W）。

> ⚠ **教训（本节的价值所在）**：第一次测速时我擅自把 `preload=False`，
> 结果 8 个 dataloader worker **100% CPU 空转 5 分钟、一步都没产出**
> （`time == elapsed`，磁盘 IO = 0，数据只有 274 MB）→ GPU 显存占满但
> **0% 利用率、61 W 掉功率**。
> **测真实速度就必须用真实配置，一个字段都不能动。**

### 2.2 ETA

| 场景 | 纯训练 | +eval | **合计** |
|---|---|---|---|
| **单 run @ 100k** | 6.7 h | 0.5 h | **≈ 7.2 h** |
| **单 run @ 250k** | 16.7 h | 1.2 h | **≈ 18.0 h** |
| **三 run 串行 @ 100k** | — | — | **≈ 21.6 h（0.9 天）** |
| **三 run 串行 @ 250k** | — | — | **≈ 54.0 h（2.2 天）** |

（eval 成本：每 5k 步一次 × 89 s → 100k 共 20 次 ≈ 0.5 h；250k 共 50 次 ≈ 1.2 h）

### 2.3 ★ 裁定：100k（用户裁定）

**三 run 各跑 100,000 步，每 5k 步一次 in-mem-eval，串行 ≈ 21.6 小时。**

`ckpt_every=5000` 同时定义了 ckpt 存盘点与 eval 点
（`train.py:2145` 传 `is_eval_step=_save_ckpt`，而
`_save_ckpt = train_steps % ckpt_every == 0`）→ 每 5k 步自动 eval，无需额外配置。

历史步数参考：

| 实验 | 步数 | 备注 |
|---|---|---|
| v11_pretrain_M432 | 250k–400k | strict 在 **152.5k** 才见顶 |
| v12_pretrain_S_cat | 400k | — |
| v13_base_50k | 250k | 主线基线 |
| v17_s2_s2*（7 个 S2 消融） | 250k | — |
| v17_pretrain_S_cat | 400k | — |
| **v17_s2_s2z_baseline** | **40k** | ← 冒烟值，不是实验值 |

100k 足以看到 `dmod` / D3 下界的**单调趋势**。若 100k 处趋势"在涨但没到位"，
再 `--resume-full` 续跑到 250k（额外 ~11 h/run）。

> **裁定：E0/E1/E2 各跑 100,000 步。**

### 2.4 ckpt 磁盘管理（★ 必须设，否则吃满磁盘）

**根分区只剩 ~331 GB（已用 91%），单 ckpt = 148 MB。**

```
100k 步 / 5k 间隔 = 20 个 ckpt/run × 3 run × 148 MB ≈ 8.9 GB   （可接受）
```

但 `ckpt_keep` 默认是 **0 = 不轮转**。脚本里已设
**`ckpt_keep=8`**（保留最近 8 个 = 40k 步回滚窗口，约 1.2 GB/run），
防止后续 `--resume-full` 续跑到 250k 时累积到 22 GB。

---

## 3. 判据：每一步看什么

### 3.1 早期信号（20k 步内）

| 指标 | 合格线 | 含义 |
|---|---|---|
| `ratio_y_over_t` | **0.7 – 1.3** | 风格项幅度与时间项同量级（改造 1 直接产物） |
| `dmod` | **≥ 0.30** | 风格真的改变了 adaLN 的调制量（改动 2 直接产物） |
| `Diff` 无 NaN 且与 E0 同量级 | — | 改造没打乱优化 |

`ratio_y_over_t` 由自动标定**在 step 0 就打印**：

```
[style_branch] gain 自动标定: ‖t_emb‖=0.928 ‖style_out‖=19.197
                             target_y/t=1.00 -> style_gain=0.0484
```

→ **这一行是第一个检查点，训练一启动就能看。**

### 3.2 中期判据（50k–100k 步）

| 指标 | 合格线 | 含义 |
|---|---|---|
| **D3 严格下界** | **> 0**（三个 run 对照） | ★ **核心判据**：风格真的进了像素 |
| D6 `ratio_null`（**≥16 字**） | **> 1.5** | 模型能区分**具体书家**（不只是"有/无条件"） |
| `strict ink_ssim` | 不低于 E0 超过 0.02 | 风格注入**没有**破坏字形 |
| `strict ssim` | 同 E0 量级 | 同上 |

⚠ **D6 的样本量铁律**：必须 ≥16 字。9 字时 `ratio_null=2.68` 是虚高
（分母不稳），16 字才降到 1.46。**别用少样本的 D6 下结论。**

⚠ **D3 的三个坑**（沿用 `docs/922` 的既有结论）：
噪声地板、方差、in/bg 区域无意义 —— 读下界时不能只看绝对值，
要看 **E1/E2 相对 E0 的位移方向**。

---

## 4. 停止 / 推进规则（预先定死，避免事后找理由）

```
E1 (改动 1) 跑满 100k
    │
    ├─ D3 下界 > 0 且 ratio_null > 1.5?
    │      YES → 改造 1 够了。E2 仍要跑（确认改动 2 是否额外加分），
    │             然后进"三选一"的最终配方确认。
    │      NO  ↓
    │
E2 (改动 1+2) 跑满 100k
    │
    ├─ D3 下界 > 0?
    │      YES → 改动 2 是必需的（支路带来了增益）。
    │      NO  → **上改动 3**（x 的直接风格加法，绕开 glyph_scale=0.4 的二次衰减）
    │             实现清单见 80_injection_redesign.md §2 改动 3
    │
    └─ 若改动 3 也无效 → **上改动 4**（t_emb 的 LN 驯服）
```

**硬规则**：

1. **每 5k 步存 ckpt**（`ckpt_every=5000` 已在配置里）→ 随时可中止、可续跑；
2. **20k 步看一次早期信号**，`dmod` 若 < 0.10 且 `ratio_y_over_t` 落在
   0.7–1.3 之外 → **不必跑完 100k**，直接判定"改动 1 没生效"并查接线；
3. **不许中途改配置**：任何超参变化都得新起一个 run，否则 100k 的对照白跑；
4. **E0/E1/E2 必须同一份 `data_csv` / `eval csv` / `seed`**（已保证）。

---

## 5. 串行脚本

见 `_sync_work/run_stage1_serial.sh`（跑 E0 → E1 → E2）。

用法：

```bash
ssh 4090
cd /root/Workspace/xy/DiT
tmux new -s s1
bash _sync_work/run_stage1_serial.sh 2>&1 | tee _sync_work/stage1.log
# Ctrl-B D 脱离
```

脚本特性：

- **串行**，前一个跑完（或崩了）才起下一个；
- 每个 run 独立 `results_dir` + 独立日志；
- `CUDA_VISIBLE_DEVICES=0` 显式绑定唯一那张卡；
- 失败**不静默继续**（记录 return code，崩了就停下来让人看）；
- 每个 run 起跑时打印参数量与 `style_gain` 标定值（§3.1 的第一个检查点）。

---

## 6. 时间线（实测 4.15 steps/s，100k 档）

| 时段 | 内容 |
|---|---|
| **T+0（01:13 已起跑）** | E0（纯基线，100k，36,445,457 参数） |
| T+~7.2h | E0 完 → 自动起 E1（改动 1，100k） |
| T+~14.5h | E1 完 → 自动起 E2（改动 1+2，100k） |
| T+~21.6h | E2 完 → **三个 run 一起做 D3 / D6 对照分析** |
| T+~1d | 按 §4 规则决定：收工 / 上改动 3 / 上改动 4 |

**≈ 0.9 天出趋势结论。** 若要标准档：`STEPS=250000 bash ...`（≈ 2.2 天）。

---

## 7. 起跑状态与检查清单

**已起跑**：`2026-09-23 01:13:32`，tmux 会话 `s1`

```
CPU 侧: tmux s1 -> bash _sync_work/run_stage1_serial.sh | tee _sync_work/stage1.log
GPU    : 82% 利用率 / 21.2 GB / 368 W
E0 实测: Trainable Parameters: 36,445,457   ← 正好是纯基线
         Diff 1.40(step100) -> 0.52(step600)  单调下降
         Steps/Sec 4.09-4.11                  ← 与基准 4.15 一致
```

### 日志在哪

| 文件 | 内容 |
|---|---|
| `_sync_work/stage1_runs/<run>.log` | **★ 每个 run 的完整训练日志（实时）** |
| `_sync_work/stage1_runs/<run>.run.json` | 该 run 实际用的配置（已合并覆盖项） |
| `_sync_work/stage1.log` | 队列级输出（**注意 `tee` 块缓冲，有延迟**） |
| `assets/results/<run>/<时间戳>-<exp>/` | ckpt / poster / eval 产物 |

⚠ **看进度一律读 `stage1_runs/<run>.log`**，不要读 `stage1.log`
（后者经管道 `tee`，输出被块缓冲，会显得"没进展"）。

### 进度看板

```bash
ssh 4090 "cd /root/Workspace/xy/DiT && bash _sync_work/stage1_status.sh"
```

一次性打印：GPU 状态 / 三 run 的 step 与进度 / 参数量 / `style_gain` 标定值 /
最近一次 in-mem-eval 的 strict 指标。

### 从队列中恢复 / 重跑单个 run

```bash
# 只重跑一个（不动队列）
ssh 4090 "cd /root/Workspace/xy/DiT && \
  CUDA_VISIBLE_DEVICES=0 /opt/conda/envs/cu121/bin/python -m src.train.train \
  --config _sync_work/stage1_runs/v17_s2z_ada1_ln.run.json"
```

### 检查清单

- [x] `src/model/dit.py` / `modules.py` 已同步远端，md5 一致
- [x] 三套验收测试全绿（`test_style_{branch,ada,wiring}.py`）
- [x] 4 种 `condition_fusion` 梯度可达（含 `xl_highdim` 懒适配修复）
- [x] 默认配置零影响（新增参数全 `None`，参数量与改动前一致）
- [x] 两份新配置 JSON 合法、已同步远端
- [x] 改动 1 / 改动 1+2 的**真实训练路径**冒烟均通过（20 步 / 40 步无 NaN）
- [x] GPU 确认空闲（15 MiB / 24564 MiB，0%）
- [x] **★ 速度基准实测：4.15 steps/s（真实配置，preload=True）**
- [x] **磁盘管理：`ckpt_keep=8`（根分区仅剩 ~331 GB，已用 91%）**
- [x] **串行队列已起跑（tmux `s1`）**
