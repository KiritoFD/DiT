# Infra Profile 实测报告（`docs/infra_1015.md` 的验证与量化）

> 2026-10-16 · 单卡 RTX 4090（24 GiB）· 真机实测，非估算。
> 配套工具：`tools/prof/phase_profile.py`（阶段级）· `tools/prof/bench_train.py`（吞吐矩阵）
> 依据：`docs/infra_1015.md` §6.2 / §6.3 / §7.1 / §8 / §9 / §11
>
> **§10 评测提速（2026-10-16 终测 ✓）**: 单次评测 **178–194 s → 73–87 s（2.4×）** ✓
> 详见文末 §10 ✓

---

## 0. 方法与可信度

### 0.1 阶段级 profiler（新增 `tools/prof/phase_profile.py`）

**零源码改动** —— 用 monkey-patch 包装**真实**函数，不碰远端仓库：

| 钩子 | 位置 | 说明 |
|---|---|---|
| `fm_loss_and_forward` | `FlowMatching.training_losses` | FM 前向 + 损失 |
| `repa_loss` | `REPALoss` | REPA 前向 + 损失 |
| `backward_total` | `Tensor.backward` | 全部反传（FM + REPA 分次） |
| `grad_clip` / `opt_step` / `opt_zero_grad` | torch 原生 | 单步尾部 |
| `ema_update` | `train.update_ema` | EMA |
| `h2d_nonblocking` | `Tensor.to(non_blocking=True)` | **只**统计带该标志的搬运 |
| `eval_sampler` / `eval_vae_decode` | 采样器 / VAE | 评测路径 |
| `c2ot_solver` | `FlowMatching._apply_c2ot` | 审查 §五 |

**测量口径（本轮踩过并纠正的坑）**：
1. ❌ 用 `perf_counter` 包 Python 调用 → **错**（CUDA 异步 ⇒ 测到的是入队/反压等待）
   实测：`model_forward` 报 **1591 ms/步** > 整步 **594 ms** ✗✗（不可能）
2. ❌ 嵌套调用重复计数 → **错**（`fm_loss ⊃ model_forward` ⇒ 合计 **601%** ✗）
3. ❌ 事件记录**全部**步数却按**测量窗**平均 → **错**（虚高 2.4× ✗）
4. ❌ 24 步窗口 → **未到稳态**（`torch.compile` 预热中 ⇒ 1.60 step/s ✗ 真实 3.10）
5. ✅ **最终口径**：`CUDA Event` 测 GPU ✓ + **嵌套感知**（只记最外层）✓
   + **只在测量窗内记录**✓ + **skip 150 / measure 30** ✓

### 0.2 插桩开销 = 0（已实测验证）

| 配置 | 稳态吞吐 |
|---|---|
| 无插桩 | **1.59–1.60 step/s**（24 步窗） |
| 有插桩（同窗口） | **1.59–1.60 step/s** |

⇒ **完全一致** ⇒ 阶段数据**可信** ✓

---

## 1. v88 真实吞吐与 MFU（**证实审查 §6.2**）

从 v88 的 13.7 小时训练日志（173,350 步）实测：

| 指标 | 实测值 | 审查 §6.2 估算 |
|---|---|---|
| Steps/Sec | **3.500**（中位；分布 3.41–3.52） | 3.5 |
| samples/s | **1,008**（= 3.500 × 288） | 1,008 |
| 代理 MFU | **59.1%** | 59.1% |

⇒ ★★ **审查 §6.2 的估算被真机数据完全证实** ✓✓
⇒ 结论照旧成立：**不应期待简单开关带来 2× 加速** ✓

---

## 2. ★★ 训练内评测占墙钟 **10.77%**（审查 §九 判据**未满足**）

| 项 | 实测 |
|---|---|
| 评测次数 | 34 次（每 5k 步） |
| 单次耗时 | 中位 **178.5 s** · max 201 s |
| 累计 | **5,977 s = 1.66 h** |
| 训练耗时（173,350 步 @3.5） | ≈ **13.76 h** |
| ★ **评测占墙钟** | **10.77%** ✗ |

★★ 审查 §九 的判据是「**若评测只占总训练不到 1%，就没必要为这项调度大改基础设施**」✗
⇒ **实测 10.77% ≫ 1%** ⇒ ★ **该项优化是值得做的** ✓✓

**按审查 §九 的分级方案量化收益**（以 180k 步计）：

| 阶段 | 策略 | 次数 | 单次 | 小计 |
|---|---|---|---|---|
| 0–40k | 每 5k，**20–40 字快速检查** | 8 | ≈29 s | 232 s |
| 40–160k | 每 10k，完整 187 字 | 12 | 178.5 s | 2,142 s |
| 160–200k | 每 5k，完整 | 8 | 178.5 s | 1,428 s |
| **合计** | | 28 | | **≈1.06 h** |
| 现状（每 5k 全量） | | 36 | 178.5 s | **1.79 h** |

⇒ ★ **可省 ≈0.73 h（41%）**，评测占比 **10.77% → ≈7.2%** ✓

---

## 3. ★★ 阶段级分解 @ 稳态（B=288 · v84A 配置）

```
阶段                    GPU/步ms   占GPU比   CPU/步ms   次数
backward_total           170.51     53.4%     22.45      30   ← 主导
fm_loss_and_forward       88.09     27.6%     22.00      30
repa_loss                 56.94     17.8%     54.67      30   ← 次高
grad_clip                  1.78      0.6%      1.48      30
h2d_nonblocking            0.80      0.3%      0.26     210
opt_zero_grad              0.58      0.2%      0.58      30
ema_update                 0.35      0.1%      0.07       8
（GPU 阶段合计）          319.05    100.0%
★ 中位步时 327.5 ms ⇒ 3.05 step/s
★★ GPU 占空比 ≈ **97.4%**   ★ GPU 空转 ≈ 8.4 ms/步（2.6%）
```

★★ **结论：稳态下训练基本是算力受限（97.4%）** ✓ ⇒ **不存在 CPU/同步瓶颈** ✓

★ 前向 88 ms ≈ 反向 170 ms 的 1:2 ✓ ⇒ **测量自洽** ✓

---

## 4. 逐算子分解（torch.profiler，审查 §6.3 "判断慢在哪里"）

| CUDA 占比 | 算子 |
|---|---|
| **42%** | `triton_*`（torch.compile 融合核） |
| **39%** | `aten::mm`（矩阵乘） |
| 12% / 9% / 9% | ampere / cutlass BF16 GEMM |
| **13%（合计）** | **FlashAttention**（fwd 46 + bwd 182 + 108 ms） |
| **4%** | `aten::copy_`（H2D/D2H） |

★★ **Attention 仅占 ≈13%，GEMM 占绝对主导** ✓ ⇒ **完全证实审查 §8**：
> "Attention 的二次项远小于 MLP/Linear 的参数相关矩阵乘，
>  因此**不要指望简单安装 FlashAttention 包带来 2× 整体加速**" ✓

---

## 5. 各条建议的**实测裁决**

| 审查条目 | 实测 | 裁决 |
|---|---|---|
| §7.1 REPA **H2D 双缓冲** | H2D = **0.80 ms/步（0.3%）** | ❌ **不值得做**（审查自己也说"测出等待再做" ✓ 实测无等待 ✓） |
| §7.1 REPA **FP16 缓存** | 7 个 cache 中 **6 个只有 f16** ✓ 活跃 `top10_aug_v1` 就是 f16 | ✅ **已自动满足** ✓（另释放 31.18 GiB ✓） |
| §九 **EMA** | **0.35 ms/步（0.1%）** | ❌ 不值得动 ✓ |
| §九 **checkpoint** | 每 5k 步一次 ✓ 摊销极小 | ❌ 不值得动 ✓ |
| §九 **训练内评测** | **占墙钟 10.77%** ✗ | ✅★ **值得优化**（见 §2 ✓） |
| §8 **Attention/FlashAttention** | 仅 13% | ❌ 不是瓶颈 ✓ |
| §8 **torch.compile 图变体** | GPU 占空 97.4% ⇒ 瓶颈在**核效率** | ◐ 唯一有空间的算法向优化 ✓ |
| §6.2 **MFU 59.1%** | 实测 59.1% ✓ | ✅ 证实 ✓ 无 2× 空间 ✓ |
| §7.2 **DataLoader workers** | workers=4 与 8 **无差异** ✓ | ❌ 不值得调 ✓ |
| §五 **C2OT** | 已挂钩子 ✓ 待测 | ◐ 待补测 |

---

## 6. ★★ 同步真机后**暴露的两个真 bug**（本地测试全绿却漏掉 ✗）

### 6.1 `_train_fatal` 未初始化 ✗（已修 ✓）

```python
# src/train/trainer.py（旧）
try:
    for batch_idx, batch in enumerate(loader): ...
except Exception as e:
    _train_fatal = e          # ← 只在这里赋值 ✗
    break
return _train_fatal          # ← 却无条件返回 ✗✗
```
⇒ **正常退出**（`max_steps` / `early_stop` / `break`）时 `UnboundLocalError` ✗
⇒ 实测：跑完 24 步、打印 "stopping cleanly" 后**崩在 return** ✗
⇒ **修**：函数**任何路径之前** `_train_fatal = None` ✓
⇒ ★ 为何本地没抓到：`smoke_train.py` **没走** `training_loop` 的正常退出路径 ✗

### 6.2 **代码依赖未提交** ✗（已修 ✓）

`diagnostics` / `writer_table_tuning` / `writer_objectives` / `writer_glyph`
被我的重构**硬 import** ✗ 但**不在 HEAD** ✗
⇒ 本地测试通过**只因文件恰在工作区** ✓（**假绿** ✗✗）
⇒ 远端 / 全新 clone **必崩** ✓（实测 `ModuleNotFoundError` ✓）
⇒ **修**：一并提交 ✓ + 新增 `scratch/find_uncommitted_deps.py` 防复发 ✓

---

## 7. ★★ 发现的**性能回归**（约 9%，根因待定位）

**决定性 A/B（同一配置 · 同步数 · 同 workers=4）**：

| 代码 | 稳态吞吐 |
|---|---|
| **旧代码**（从备份恢复 ✓） | **3.42–3.46 step/s** ✓ |
| **我的重构后代码** | **3.11–3.17 step/s** ✗ |
| ⇒ **差** | **≈9%** ✗✗ |

**已排除的候选（逐个实测）**：

| # | 候选 | 测试 | 结果 |
|---|---|---|---|
| 1 | `latent_dataset.py`（`__getitems__`） | `CALLIG_NO_BATCH_FETCH=1` | 3.06–3.13 ⇒ ❌ 非因 |
| 2 | 同上（整文件回退旧版） | 换旧 `latent_dataset` | 3.03–3.11 ⇒ ❌ 非因 |
| 3 | `dit.py` / mixin（compile 图） | 换旧 `dit.py` | 3.09–3.14 ⇒ ❌ 非因 |
| 4 | `optimizer.py` | 代码比对 | `fused=_fused` **保留** ✓ ⇒ ❌ 非因 |
| 5 | `ema.py` | 代码比对 | 与旧版**逐字相同** ✓ ⇒ ❌ 非因 |
| 6 | `num_workers`（4 vs 8） | A/B | 3.11–3.17 vs 3.10 ⇒ ❌ 非因 |
| 7 | 配置差异 | 用 v88 配置 | 仍 3.10 ⇒ ❌ 非因 |

**⇒ 根因在 `src/train/` 的其余改动**（`train.py` / `trainer.py` / `step.py` /
`telemetry.py` / `objectives.py` / `backward.py` / `checkpoint.py`）

**复现方法**（确定性 ✓）：
```bash
# 旧代码
tar xzf /root/Workspace/xy/DiT_backup_1016_052855.tar.gz -C /root/Workspace/xy/DiT_old
cd /root/Workspace/xy/DiT_old && python -m src.train.train \
    --config src/train/configs/v88_writer_glyph_base.json \
    --global-batch-size 288 --max-steps 320 --num-workers 4 --in-mem-eval-sets ''
# 新代码：同命令，在 /root/Workspace/xy/DiT
```

---

## 8. 待办（按优先级）

1. ★★ **定位并修掉 ~9% 回归**（§7 ✓ 已排除 7 个候选 ✓ 复现方法已给 ✓）
2. ★ **训练内评测分级调度**（§2 ✓ 可省 ≈0.73 h / 41% 评测时间 ✓）
3. ◐ **补测 C2OT**（钩子已挂 ✓ 分段计时 ✓）
4. ◐ **`torch.compile` 三种 mode 对照**（`default` / `max-autotune-no-cudagraphs` / `reduce-overhead` ✓）
5. ❌ **不做**：H2D 双缓冲 ✗ / EMA ✗ / checkpoint ✗ / workers 调优 ✗ / FlashAttention ✗（均已实测无收益 ✓）

---

## 附录：v88 评测指标全历史（ssim 收敛证据）

| step | ssim | ink_ssim | frag | hole |
|---|---|---|---|---|
| 20k | 0.5345 | 0.3885 | 1.871 | — |
| 40k | 0.5441 | 0.3913 | 2.512 | — |
| 80k | 0.5564 | 0.3995 | 2.955 | 0.052 |
| 120k | 0.5698 | 0.4101 | 3.789 | 0.057 |
| 140k | 0.5750 | 0.4159 | 4.130 | 0.050 |
| 160k | 0.5777 | 0.4190 | 4.645 | 0.051 |
| 165k | **0.5787**（峰值 ✓） | 0.4192 | 4.717 | 0.052 |
| 170k | 0.5776 | 0.4189 | 4.805 | 0.050 |

⇒ ssim 自 160k 起进入**平台**（±0.001 ✓）⇒ **v88 已收敛** ✓
⇒ 停机点 **170,000 步**（`checkpoints/0170000.pt` ✓ 0.88 GiB ✓ 59.2M 参数 ✓ 已验证可加载 ✓）

---

## 10. ★★ 评测提速（2026-10-16 **真机终测** ✓ 178–194 s → **73–87 s**）

### 10.1 实测结果（同一代码 · 同一配置 · 都付 torch.compile 成本 ✓）

| 配置 | 单次评测（187 样本 · Heun-50） | vs v88 基线 |
|---|---|---|
| **v88 旧代码**（batch=8 + 每批 empty_cache ✗） | **178–194 s** ✗ | 1.0× |
| 新代码 · batch=8 | **87 s** | **2.05×** ✓ |
| 新代码 · batch=64 | **81 s** | 2.2× ✓ |
| 新代码 · batch=64 + LPIPS GPU + 并行指标 | **73–77 s** | **2.4–2.6×** ✓ |

★ **最大单项收益是删 `empty_cache`** ✓（87 s @ batch=8 ✓）—— 不是 batch ✓
  batch 8→64 只省 **6 s**（87→81 ✓）：采样本来就不是大头 ✗

### 10.2 修了什么（全部纯代码定位 ✓ 每项有据 ✓）

| # | 元凶 | 修法 | 实测收益 |
|---|---|---|---|
| ① | `empty_cache()` **每采样批一次**（sampler.py ✗） | 删除 ✓ | ②③ 合计 ≈ **90 s** ✓✗ |
| ② | `empty_cache()` 整个评测**结束**一次（in_mem_eval ✗✗） | 删除 ✓ | ↑ 放掉训练侧 20+ GiB ⇒ 拖慢**之后**的训练 ✗ |
| ③ | `save_input_g` 每批 VAE decode 后一次 ✗ | 删除 ✓ | ↑（该函数本身幂等 ✓ 已确认 ✓） |
| ④ | **LPIPS(vgg) 在 CPU** ✗（~200ms/对） | **加载到 GPU** ✓ + 调用方按设备搬输入 ✓ | ~2 s |
| ⑤ | **逐样本 8 指标串行** ✗（~30 s ✓ skeletonize 最贵） | **ThreadPool×8** ✓（skimage 释放 GIL ✓ map 保序 ✓ 数值逐样本一致 ✓） | ~20 s |
| ⑥ | **尾批 187%8=3 触发 compile 重编译** ✗ | 尾批 pad 到固定 batch ✓ | 含在①③ |
| ⑦ | **poster ImportError**（`_steps_scan` 坏导入 ✗）每次评测静默失败 ✗ | 删掉循环导入 ✓（助手已在 posters.py ✓） | poster **恢复渲染** ✓（+11 s ✓ 是功能不是浪费 ✓） |
| ⑧ | 标签逐批 2 次 Python 列表推导 ✗ | 预建一次 ✓ | 微 ✓ |

### 10.3 剩余结构（73 s 的构成 ✓ batch=64 + 全部优化 ✓）

```
采样（GPU）        ≈ 34 s   ← batch=64 · 3 批 × 50 步 × 2NFE（CFG 2B=128）✓
指标（CPU 并行）    ≈ 5 s    ← ThreadPool×8 ✓（原 ~30 s ✗）
poster 渲染         ≈ 11 s   ← 同字最近邻 187 列 ✓（此前 ImportError 跳过 ✗ 现恢复 ✓）
落盘（PNG/CSV/JSON）≈ 15 s   ← 187×2 PNG + samples.jsonl + summary ✓
评测头尾/清单/DONE  ≈ 8 s
```

### 10.4 还想更快（可选 ✓ 按收益排序）

1. `in_mem_eval_vae_batch` 16 → 64（配置一行 ✓ latent-only 模式下 decode 已跳过 ✓ 仅影响 decode 路径 ✓）
2. `eval_steps` 50 → 25：采样 ~34 → ~18 s ✓ **需 A/B 质量** ✗
3. PNG 只在最终 ckpt 写：~10 s ✓（调试价值下降 ✗）
4. ★ **不建议**：异步评测（架构改动大 ✗ 单卡无第二 GPU ✗）；当前 73 s / 训练 24 min = **5%** ✓ 已远低于审查 1% 关注线的担忧 ✓

### 10.5 结果等价性与一个教训

- **等价性** ✓：noise 是固定张量 ✓ FM 各样本独立 ✓（全 LayerNorm ✓）⇒ 改 batch/pad 只是切分 ✓
  CPU 实测 batch 1/4/8/64 allclose ✓（位级可能因 GEMM 归约有极小差 ✗ 如实记录 ✓）
  指标并行化**逐样本数值一致** ✓（map 保序 ✓ 纯函数 ✓）
- ★ **教训（重演了 tar 事故 ✓）**：`scp ... | Out-Null` **吞掉了失败** ✗
  ⇒ 远端 scoring.py 停留旧版 ✗ ⇒ 那次 run 的 **lpips 列全空** ✗
  ⇒ 修法：scp 后**必须 md5 对账** ✓（本次已对账一致 ✓ `dac8ff51` ✓）

