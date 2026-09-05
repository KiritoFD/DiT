# 34 — CPU/RAM 硬件档案与 eval 瓶颈细粒度测量（2026-09-05 深夜）

> 为优化方案准备的完整数据集。所有数字均为实测，脚本在 `tools/diag/`
> （hw_gemm / hw_mem / profile_cpu2 / profile_cpu3 / profile_cpu4 / bench_cpu2），可复跑。
> 结论先行见 §8；§1-7 是原始数据。

---

## 1. 硬件档案（实测 dump，非规格书）

| 项 | 值 |
|---|---|
| CPU | 2× AMD EPYC 7542（Zen2/Rome，family 23 model 49），32C/64T ×2 = 64C/128T |
| 频率 | governor=**schedutil**，实测 2.4GHz，max 2.9GHz（AVX 重载下通常 2.6-2.9） |
| 单核 fp32 峰值 | 2×256b FMA = 32 FLOP/clk → 2.9GHz ≈ **93 GFLOPS**；全 socket ≈ 2.95 TFLOPS |
| 缓存 | L1d/L1i 32K，L2 512K/核，**L3 16MB/CCX（每 socket 8 CCX，共 128MB）** ← 关键拓扑 |
| 指令集 | **AVX2 仅此**（无 AVX512、无 AVX512-BF16、无 AMX）→ bf16 无硬件加速 |
| NUMA | node0 = CPU 0-31,64-95；node1 = CPU 32-63,96-127（物理核 0-31 / 32-63） |
| 内存 | 251G（node0 131G/node1 131G），node0 free 42G，全局 available 221G |
| THP | **`always [madvise] never` —— madvise 模式**（glibc malloc 不主动 madvise → 大分配默认无巨页）；AnonHugePages 仅 11.5G |
| glibc | **2.27**（Ubuntu 18.04，较老；动态 mmap 阈值上限 32M） |
| 内核 | 6.8.0-111-generic |
| torch | 2.1.2+cu121，cap=AVX2，mkldnn=True |
| 实测负载 | v9c 训练主进程 ~7.3 核 + daemon ~3.5 核 + 3×dataloader 0.12 核；load 2-6（评测时段会短时抬升） |

## 2. GEMM 线程扩展性（`hw_gemm.py`，socket1 物理核逐级绑核）

TFLOPS（每个格子 = 30-40 次平均）：

| 线程 | square 4096³ | qkv (8192,384)@(384,1152) | mlpup (8192,384)@(384,1536) | mlpdown (8192,1536)@(1536,384) |
|---|---|---|---|---|
| 1 | 0.076 | 0.061 | 0.062 | 0.075 |
| 2 | 0.148 | 0.113 | 0.116 | 0.149 |
| 4 | 0.217 | 0.150 | 0.160 | 0.193 |
| 8 | 0.509 | 0.270 | 0.325 | 0.448 |
| 16 | 0.841 | 0.417 | 0.442 | 0.790 |
| 32 | **1.300** | **0.545** | **0.554** | **0.861** |
| 56（跨双 socket） | 1.674 | 0.653 | 0.644 | 1.263 |

读法：
- **单核 76 GFLOPS = 99% 单核峰值** —— BLAS 内核对 Zen2 没有病态劣化。
- 32 线程 square = 1.30 TFLOPS = 44% socket 峰值；1→32 核并行效率 53%，**拐点在 ~8-16 线程**（CCX 边界后每核效率 41-53 GFLOPS 持续下滑）。
- **K=384（qkv/mlpup，模型主力形状）封顶 0.545-0.554**，只有 square 的 42% —— 小 K 的算术强度天花板，换 BLAS 收益有限。
- K=1536（mlp down）明显更好（0.861）——K 越大越接近 compute-bound。
- **跨 socket 56 线程只 +29%**（32→56 线程本应 +75%）—— NUMA 惩罚。⇒ 多 socket 要用**多进程各绑一 socket**，不要单进程跨 socket 加线程。

## 3. 内存带宽与分配开销（`hw_mem.py`）

| 测量 | 32t（socket1） | 56t（双 socket） |
|---|---|---|
| triad 元素操作带宽 | **59.8 GB/s** | 75.5 GB/s |
| **新分配 64MB + touch** | **12.5 ms/次** | 13.1 ms/次 |
| 复用缓冲 fill 64MB | **0.3 ms/次** | 0.2 ms/次 |

- triad 60GB/s = 理论（DDR4-2933 8ch ≈ 190GB/s/socket）的 **~32%**（torch eager 元素操作非带宽优化 + schedutil 降频）。
- **`新分配+touch` 比 `复用` 慢 42×（12.5 vs 0.3 ms）** —— mmap + 4KB 页错误的代价，
  THP=madvise 下 glibc 不启用巨页。**这是全部后续分析的基石数字：≈0.20 ms/MB 首触代价。**

## 4. 单次 forward 分解（`profile_cpu2.py`，32 行 = 16 样本 CFG 拼批，main+ctrl）

| 测量 | 值 |
|---|---|
| eager CFG forward | **2253 ms**（多轮复测 1.34~3.3s：**±50% 方差来自 v9c 训练/eval 的负载波动**，比值稳定、绝对值要打折看） |
| eager main-only（16 行） | 251 ms |
| GEMM 合计（mm 1153 + bmm 421 + addmm 165） | ≈1740 ms |
| elementwise 合计（mul/add/copy/cat/pow/silu…） | ≈1430 ms |
| **模型内 GEMM 有效速率** | 77 GFLOPs / ~1740ms ≈ **0.044 TFLOPS —— 比独立同形状基准（0.545）慢 ~10×** |

## 5. 分配 churn 实锤（`profile_cpu3/4.py`）

- **T3 同一个 mm：fresh-alloc 13.4 ms vs `out=` 预分配复用 5.8 ms —— 分配开销占 56%**（36MB 输出，0.21ms/MB，与 §3 的 12.5ms/64MB 互相印证）。
- **每 forward 新分配总量推算**（24 个块 × qkv/attn/gate/up/silu/down/mod 中间张量）≈ **7.5 GB**；
  按 0.20 ms/MB 首触 → **≈1.5 s/forward，占 1.34s 基线 forward 的 50-70%**。
  （这同时解释了"模型内 GEMM 慢 10×"：op 计时里混入了输出页错误，以及权重 tile 冷读。）
- THP=madvise + glibc 2.27 不 madvise → 这些分配全部走 4KB 页。**系统没有一处启用巨页来吸收它。**

## 6. attention 路径（`profile_cpu4.py` A + `profile_cpu3.py` T2）

| 路径 | attention 微基准 (32,6,256,64) | 整 forward 对照 |
|---|---|---|
| **sdpa（flash CPU）** | **7.8 ms（0.413 TFLOPS）** | forward 3316 ms |
| eager bmm+softmax | 32.1 ms（4× 更慢） | forward 3826 ms |

⇒ **"换 eager attention"这个候选方向被证伪**：sdpa flash-CPU 在 CPU 上反而是最优路径，
保留。in-model attention 偏慢（~22ms/次 vs 独立 7.8ms）同样归因于分配/负载波动。
（合头 mm 重构方案在数学上需要分块掩码且计算量 ×32，废弃。）

## 7. 采样器选择与端到端（`cpu_sampler.py` + `bench_cpu2.py`）

- **Heun 逐 stage vs heun_batch**：0.821 vs 1.441 s/NFE（多轮一致 ~39%）——
  heun_batch 的 6B 行/步是给 GPU launch 摊销设计的，CPU 纯亏。`cpu_sampler.py` 的
  逐 stage + CFG 拼批（forward_with_cfg 内 cat 成 2B 行）是对的。
- 端到端（32t，batch16，默认 malloc，v9c 共存负载）：ctrl 臂 133.5s/16 样本、
  base 臂 52.0s/16、VAE decode 8.8s/16 → **顺序全量 ≈ 22.1 min**（窗口 14.9 min）✗

## 8. 瓶颈综合（贡献拆分，基线 forward ≈1.34-2.25s）

| 成分 | 估算耗时 | 占比 | 性质 | 可优化性 |
|---|---|---|---|---|
| **新分配 + 首触页错误**（7.5GB/forward） | **~1.0-1.6 s** | **~50-70%** | 系统配置问题，非算法问题 | **极大（THP / malloc env / 缓冲复用）** |
| GEMM 计算（77 GFLOPs @ K384 形状天花板 0.545） | ~140-300 ms | ~10-15% | 硬件/形状天花板 | 小（换 BLAS 或改形状不现实） |
| attention（sdpa flash，已是最优路径） | ~200-540 ms | ~15-25% | 含分配开销 | 中（随分配修复连带下降） |
| eager elementwise（未融合） | ~400-700 ms | ~20-30% | 软件融合缺口 | 中（torch.compile 可消） |

**一句话：不是算力不够，是每一步前向都在用 4KB 页"冷启动" 7.5GB 内存。**

## 9. 优化杠杆（数据支持的，按预期收益排序，供你出方案）

| # | 杠杆 | 机制 | 预期 | 代价/风险 |
|---|---|---|---|---|
| 1 | **THP 切 `always`**（root，系统级，重启免） | 巨页把首触页错误降 ~512×，7.5GB/forward 的分配代价基本消失 | forward 1.34s→~0.4-0.6s，全 eval **22→~6-8 min** | 全机生效（含训练），内存碎片/膨胀风险；251G 富余 |
| 2 | glibc env：`MALLOC_MMAP_THRESHOLD_=2G MALLOC_TRIM_THRESHOLD_=2G`（进程级，免 root） | 大分配改走 brk+free-list，跨步复用已热内存 | 未干净验证（首测被训练 eval 负载污染，见 §10），理论接近 #1 | 无 |
| 3 | torch.compile CPU（干净重测） | inductor 融合 elementwise + 区内 buffer 复用（也吃掉一部分分配开销） | 首测 0.97× 无效（污染数据），需干净重测 | 编译一次 ~2-5 min，常驻进程可摊 |
| 4 | **双进程臂并行**（ctrl@socket1 + base@socket0，taskset 绑物理核） | socket 间 NUMA 惩罚（GEMM 56t 只 +29%）→ 进程级隔离 | wall = max(ctrl, base) ≈ 1.6× | 两进程模型各一份 RAM（~0.7G），工程简单 |
| 5 | CPU governor → performance（root） | schedutil 实测 2.4GHz vs 2.9 峰值 | 全部 +10-20% | 功耗/发热 |
| 6 | 换 BLIS/OpenBLAS | K384 形状天花板 0.545 | 小（天花板是形状不是 BLAS） | 环境工程 |
| 7 | ~~换 eager attention~~ | **已证伪**（sdpa 快 4×） | — | — |
| 8 | ~~int8 oneDNN~~ | 破坏 GPU eval 可比性 | — | 不推荐 |

端到端预算推演（#1+#4 叠加）：ctrl 臂 ~60-90s + base ~40-60s + decode ~60-120s（分配修复连带受益）+ 指标 ~60s → **全量 ~4-6 min**，对 14.9 min 窗口余量充足；即使 #1 不可行，#2+#3+#4 保守也有 22→10-13 min，勉强进窗。

## 10. 数据可信度备注

- forward 绝对耗时多轮 1.34/2.25/3.32s：**±50% 方差来自 v9c 训练与其 GPU eval 阶段的负载波动**；
  本文结论均基于比值与多次一致的测量（分配 microbench、GEMM 扫描、attention 对照不受影响）。
- S4（malloc env A/B）首测**无效**（S3 默认 env 与 S4 带 env 的两次运行撞上 v9c 不同负载相），需在训练 eval 间隙重跑。
- 首轮 profile 曾与我自己的并发 bench 撞车（13.3s/forward），已废弃并清场重测。
- GEMM 扫描的 taskset 绑定只含物理核（无 HT 兄弟线程），不构成瓶颈。

## 11. 协议约束（方案设计时不可动）

- eval 集 clean_v8 n=100 双臂、Heun 50 步（100 NFE）、cfg 0.7 —— 与 GPU eval 数字可比性
- 输出物：`eval_samples_ctrl/stepN/{ctrl,base}/*.png` + `eval_auto_ctrl_{step}.json`
  （早停 / universal daemon / collect_v89_series 三方零改动兼容）
- CPU eval 与 GPU eval 的 ssim 对齐验收：同 ckpt 差 ≤±0.002（基准：v8b 0035000 = 0.7641）

---

## 12. 实施修订（2026-09-06，以实测推翻/修正本文前述结论）

1. **FLOP 会计修正（重大）**：§4 的"77 GFLOPs/forward"漏乘 24 块——单次 forward 实际
   **~1.4 TFLOP**。带形状 profiler（profile_cpu7）证明各 GEMM 已运行在 0.6-0.9 TFLOPS，
   **贴着 K=384/K=1024 形状的实测天花板（§2 表）**——"模型内比独立基准慢 10×"系
   分母错误，不是事实。eager fp32 前向 ~1.34s 已接近该硬件的可行极限。
2. **分配开销修正**：T3 的 56% 是冷启动值；glibc 动态 mmap 阈值在稳态采样时自适应
   复用大块内存，**稳态分配开销 ~10-15%**，不是 50-70%。
3. **优化 A/B 全部实测完毕**：jemalloc（conda jemalloc-local）**中性**（16 样本
   132.9s vs 基线 133.5s）；MALLOC_MMAP/TRIM 阈值**更差**；torch.compile **1.01×**；
   eager attention **更慢 4×**（sdpa 保留，§6 维持）。THP 与 performance governor
   因 /sys 只读（容器）**不可用**。
4. **NUMA 双 socket 分工是唯一有效杠杆**：ctrl_pair 双 worker 实测 wall 16.8 min
   （22.1 串行 ÷ 2 socket × 争抢系数 1.5）；随训练共存负载波动 ±50%。
5. **窗口结论**：ctrl_pair 双臂模式下 14.9 min 窗口**只能勉强贴住**；
   `pretrain_g` 单臂模式（v10a 起，无 ctrl encoder、无第二臂）工作量减半，
   **~9-11 min，舒适进窗**。多段 `idx_offset` 全局下标 + 逐样本列表合并已实现
   并修复一处错位 bug（decode_and_save 局部下标 → 全局）。
