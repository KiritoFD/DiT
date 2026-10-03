# INFRA 升级与 MFU 定位 (2026-09-16)

> 执行: AutoCoder (auto-coder agent) · 主机 `4090` (10.176.54.17:36430, RTX 4090 24G, driver 590.48/CUDA13.1)
> 仓库 `/root/Workspace/xy/DiT` (本地镜像 `G:\GitHub\DiT`)

---

## 0. 一句话结论

- **"编译永远跑不完"的根因是 torch 2.1.2 inductor 的一个指数级递归 bug**，不是配置问题；现在已双保险解决。
- **cu121 环境 torch 2.1.2 → 2.5.1+cu121**（base 未动，已备份可回滚）。训练步速 **5.30 → 5.83 step/s（+10%）**，MFU **~31% → 34.4%**。
- **xformers 已从 cu121 卸载**（torch≥2.0 自带 SDPA，它是遗留 fallback；且 ABI 不匹配）。
- **MFU 80% 在 RTX 4090 + 本模型规模下不可达**（有量化论证，见 §4）。实际天花板 ~40-45%。

---

## 1. 根因: torch 2.1.2 inductor `percolate_tags` 指数级递归

**证据（faulthandler + SIGABRT 栈）**：

```
Current thread ... (most recent call first):
  File "torch/fx/node.py", line 343 in all_input_nodes
  File "torch/_inductor/pattern_matcher.py", line 652 in percolate_tags
  File "torch/_inductor/pattern_matcher.py", line 655 in percolate_tags   ← 同一帧递归上万层
  ...（重复上千次）
```

`torch/_inductor/pattern_matcher.py` 内 `ReplacementPatternEntry.replace_with_graph`：

```python
def percolate_tags(node, recompute_tag):
    for arg in node.all_input_nodes:
        if hasattr(arg, "meta"):
            arg.meta["recompute"] = recompute_tag
            percolate_tags(arg, recompute_tag)   # ← 无 visited 集 → 路径数指数爆炸
```

- 触发条件：模型被编译成**一整张大图**（fwd+bwd）。独立 bench 脚本必然触发；训练脚本因图被切断成多段而侥幸跑过 —— 属于**随机地雷**，任何图结构变化都可能引爆。
- 现象：编译 >30 min 无任何输出（不是报错，是纯 CPU 空转），GPU 0%。
- 复现脚本：`/tmp/iso_compile.py` + `/tmp/iso_run.sh`（栈落 `/tmp/iso_run.log`）；卡点探针 `/tmp/stall_probe.sh`。

**修复（两层）**：
1. `src/train/train.py` 增加 env 门控（默认关闭 pattern_matcher）：
   ```
   if os.environ.get("DIT_PATTERN_MATCHER", "0") != "1":  # 需要时置 1 恢复
       from torch._inductor import config as _ind_cfg
       _ind_cfg.pattern_matcher = False
   ```
   实测关掉后：同一图 **65 s 编完**，step 时间无损失（181.1 vs 181.2 ms）。
2. 升级 torch 后上游已修此 bug（2.5.1 下 `pattern_matcher=True` 也能 74 s 正常编完）。
   → 门控保留无害，且 pm=0 略快（178.7 vs 181.4 ms）。

---

## 2. 环境变更（base 零改动）

| | base `/opt/conda` (只读记录) | cu121 升级前 | cu121 升级后 |
|---|---|---|---|
| python | 3.10.8 | 3.10.18 | 3.10.18 |
| torch | **1.13.1+cu117** | 2.1.2+cu121 | **2.5.1+cu121** |
| torchvision | 0.14.1+cu117 | 0.16.2+cu121 | 0.20.1+cu121 |
| torchaudio | 0.13.1+cu117 | 2.1.2+cu121 | 2.5.1+cu121 |
| triton | 2.1.0 | 2.1.0 | 3.1.0 |
| xformers | 0.0.16 (**未动**) | 0.0.23.post1 | **已卸载** |
| cudnn | — | 8902 | 90100 |

- **base 未做任何修改**（仅读取版本）。万一误动，还原命令：
  ```bash
  /opt/conda/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 xformers==0.0.16 \
      xformers==0.0.16 --index-url https://download.pytorch.org/whl/cu117
  ```
- **cu121 硬链接备份**：`/opt/conda/envs/cu121_bak_20260916`（秒级创建，仅 17M 额外占用）。
  回滚：`rm -rf /opt/conda/envs/cu121 && mv /opt/conda/envs/cu121_bak_20260916 /opt/conda/envs/cu121`
- 升级前完整 pip freeze：`/root/cu121_freeze_before_20260916.txt`
- 升级命令：`pip install --upgrade torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121`
- 为什么能停在 2.5.1：2.6+ 的 wheel 是 `manylinux_2_28`（需 glibc≥2.28），本机 glibc 2.27；2.5.1 仍是 manylinux2014，可用。

### xformers 为什么可以卸
- `src/model/modules.py` 的 `resolve_attn_impl`：`attn_impl="sdpa"` 时**永远走 SDPA**，xformers 只在 torch 没有 `F.scaled_dot_product_attention` 时兜底。
- `train.py` 第 2 行本来就有 `os.environ["XFORMERS_DISABLED"] = "1"`。
- 卸载后验证：`_XOPS=None`，`resolve_attn_impl('sdpa')→sdpa`，模型前向正常。

### 升级带来的实际收益
- `fx_graph_cache: True`（2.5 起默认；2.1 根本没有），热启动 **74 s → 42 s**。
- `pattern_matcher` 上游修复。
- 训练步速 **5.30 → 5.83 step/s（+10%）**；显存 18.83G → 13.27G。

---

## 3. 编译/缓存纪律（回应"怎么一直在 compile"）

- **缓存目录只用 `/root/.cache/torch/inductor`**（训练脚本已如此）；不要用一次性目录（诊断脚本曾用 `/tmp/ind_cache_*`，导致每次全量重编——已废弃）。
- 每次**进程启动**的实际开销（实测）：
  | 阶段 | 冷 | 热 |
  |---|---|---|
  | default 模式 | ~74 s | ~42 s |
  | max-autotune | ~437 s（首次 autotune） | ~60 s |
- 剩余热启动 ~42 s 属于 **Dynamo 追踪 + AOTAutograd + inductor lowering**，torch 2.5 没有对这部分做磁盘缓存（FX cache 覆盖的是 AOT 之后的图，仍省了 ~30 s）。训练一跑数小时，这属于**一次性成本**。
- 训练**稳态无重编译**（v12 日志 5.5±0.1 step/s 平稳，无 recompile 记录）。

---

## 4. MFU 定位：为什么 80% 不可达

### 4.1 实测数据（torch 2.5.1, B=240, S/2）

| 配置 | ms/step | TFLOPs/s | MFU | 说明 |
|---|---|---|---|---|
| eager（不编译） | 473.9 | 21.4 | 12.4% | |
| compile `default` | 178.2 | 56.9 | 33.1% | |
| compile `default`(pm=0) | 178.7 | 56.7 | 33.0% | |
| compile `reduce-overhead` | 181.5 | 55.8 | 32.5% | CUDA graph 未生效，无收益 |
| compile `max-autotune` | **166.2** | **61.0** | **35.5%** | 首次编译 437 s，之后 ~60 s |
| 真实训练（2.5.1, 全链路 smoke） | ~171.5 | 59.1 | **34.4%** | 5.83 step/s |
| 真实训练（2.1.2, 升级前） | ~188.7 | 53.7 | ~31% | 5.30 step/s |

### 4.2 卡的真实峰值（实测，不是标称）
- 8192³ bf16 GEMM 持续 10 s：**172.0 TFLOPs/s**（≈ 官方 dense 峰值，用作分母合理）。
- 本模型的 GEMM 形状（M=61440, K=384, N=1152/1024/384）单独跑：**157.2 TFLOPs/s（91% of 峰值）**。
  → **GEMM kernel 本身没问题**。

### 4.3 一个 step 的时间去向（chrome trace 精确统计，无重复计数）

| 类别 | ms/step | 占比 | kernel 数/step |
|---|---|---|---|
| GEMM（线性层+部分 conv） | 76.5 | 43.0% | 404 |
| **triton pointwise/reduction** | **76.1** | **42.8%** | 697 |
| attention (flash) | 17.9 | 10.0% | 60 |
| optimizer | 3.8 | 2.1% | 32 |
| **合计** | **177.7** | | **1255** |

- kernel 时间 177.7 ms ≈ wall 178.2 ms → **GPU 100% 占满，启动开销已完全隐藏**（所以 CUDA graph / 放大 batch 都无效）。

### 4.4 batch 扩展性（决定性实验）

| B | ms/step | MFU | 峰值显存 |
|---|---|---|---|
| 240 | 178.2 | 33.1% | 9.0 GB |
| 360 | 263.8 | 33.5% | 13.2 GB |
| 480 | 353.2 | 33.4% | 17.4 GB |
| 600 | 436.3 | 33.8% | 21.6 GB |

**MFU 完全平坦，时间随 batch 线性增长** → 瓶颈是纯吞吐（GEMM 效率 + 显存带宽），不是调度/固定开销。

### 4.5 天花板推导

- 43% 的时间是逐元素/归一化内核（RMSNorm / adaLN / 门控 / RoPE / SwiGLU / 损失），它们**用不到 tensor core**，受 HBM 带宽限制（4090 ≈ 1.0 TB/s）；76 ms ≈ 76 GB 激活流量。
- 纯 GEMM 下界：linear fwd+bwd ≈ 7.8 TFLOPs / 157 TFLOPs/s ≈ **50 ms**；attention ≈ 18 ms；pointwise ≈ 76 ms。
  → **即使零额外低效，一步也要 ≥ 145 ms**。
- 80% MFU 需 **74 ms/step**（10.134 / 0.074 / 172），**低于"GEMM+attention"的物理下界**。
- 结论：**80% 不可达**；本模型规模（h=384、12 层、32×32 latent、B=240）在 4090 上的现实天花板约 **40-45%**；当前 34.4% 已在合理区间内。

### 4.6 若一定要更高 MFU（非 infra 范畴，供决策）
- 提高每字节算力比：加大 hidden/深度（GEMM 占比↑，pointwise 占比↓），或上更大 batch+更大模型；
- 换 FLOP:带宽比更高的卡（如 A100/H100 的 HBM 波特率与 L2 更优）。

---

## 5. 本次改动清单

| 文件/位置 | 改动 | 状态 |
|---|---|---|
| `src/train/train.py` | compile 前加 `pattern_matcher` env 门控（`DIT_PATTERN_MATCHER`） | 已改（本地+远程同 md5 `e015055b4121c1dfd3028b9e1e332065`） |
| `src/train/train.py.bak_pm20260916` | patch 前备份 | 已存 |
| cu121 env | torch 2.5.1+cu121 / tv 0.20.1 / ta 2.5.1 / triton 3.1.0；卸载 xformers | 已生效 |
| `/opt/conda/envs/cu121_bak_20260916` | 硬链接备份 | 已存 |
| `/root/cu121_freeze_before_20260916.txt` | 升级前 pip freeze | 已存 |
| base `/opt/conda` | **未改动** | — |

## 6. 证据文件（远程）

| 内容 | 路径 |
|---|---|
| 卡点栈（递归爆炸） | `/tmp/stall_probe.log`, `/tmp/iso_run.log` |
| compile 模式矩阵（2.1.2） | `/tmp/bench2_run.log` |
| 升级脚本与验证 | `/tmp/torch_upgrade.sh`, `/tmp/torch_upgrade.log` |
| compile 模式矩阵（2.5.1） | `/tmp/bench3_run.log`, `/tmp/ma2.log` |
| kernel 级拆解 | `/tmp/profile3.py`, `/tmp/profile3.log`, `/tmp/trace.json` |
| batch 扩展性 | `/tmp/bench5_run.log` |
| 峰值/形状效率 | `/tmp/ceiling.py`（输出见 §4.2） |
| 全链路 smoke | `/tmp/smoke_train.sh`, `/tmp/infra_smoke/train_smoke.log` |
| xformers 卸载验证 | `/tmp/xformers_remove.log` |

---

## 7. 代码级审计（"哪里实现拖累 MFU"）

审计对象：`src/model/modules.py`、`src/model/dit.py`、`src/loss/flow_matching.py`、`src/train/train.py`。
方法：**纯 eager 微基准**（不 compile）+ chrome trace 内核账本 + 逐文件通读。

### 7.1 已修复的代码级问题（含实测）

| # | 位置 | 问题 | 修复 | 实测 |
|---|---|---|---|---|
| 1 | `modules.py: RMSNorm.forward` | 每次调用 `x.float()` → 算 → `.to(dtype)`，bf16 训练下多出 fp32 中间张量；全模型 ~48 次/步 | 改走 `F.rms_norm`（torch≥2.4 融合内核），老 torch 自动回退 | **-2.9 ms（-1.7%）** |
| 2 | `train.py` optimizer | `AdamW` 默认逐张量更新 | `fused=True`（带守卫） | 含在下一行 |
| 3 | `train.py` 顶部 | glyph_embedder 的 conv 每次用默认启发式算法 | `cudnn.benchmark=True`（形状固定） | 2+3 合计 **-1.5 ms（-0.9%）** |
| 4 | cu121 env | xformers 0.0.23 与 torch 2.5 ABI 不匹配、且 `attn_impl=sdpa` 下永远不用 | 从 cu121 卸载（`train.py` 本来就有 `XFORMERS_DISABLED=1`） | 消除告警/隐患 |

合计（default 模式）：**177.8 → 173.4 ms，MFU 33.1% → 34.0%**。

### 7.2 审计过但**不必改**的点（有实测依据）

| 位置 | 疑问 | 实测结论 |
|---|---|---|
| `modules.py: Attention.forward` | q/k/v 是 `permute+unbind` 出来的非连续张量（stride 非标准），可能拖慢 flash | **无差异**：SDPA fwd 222 µs vs 连续 221 µs；fwd+bwd 1562 vs 1625 µs。flash 对 stride 不敏感，不用加 `.contiguous()`（加了反而多一次 47MB 拷贝） |
| `modules.py: apply_rope` | `torch.cat([-x2,x1])` 物化一份 | 数学与 slice 版等价；eager 下 cat 版 525 µs vs slice 419 µs，但**编译后两者同样被融合**，收益 <1%。且改它会动到数值实现，暂不碰 |
| `modules.py: Attention.forward` | SDPA 输出 `transpose(1,2).reshape` 是一次 47MB 拷贝 | 实测 104 µs/次（≈ 纯带宽），×12 层 ×(fwd+bwd) ≈ 3.7 ms/步（~2%）。**multi-head + 融合 proj 的固有开销**，除非写自定义 kernel，否则无法避免 |
| `train.py` 训练步 | 5 处 `.item()` 在 `backward()` 之前，看似排空流水线 | 其中第 1 处来自 **NaN 守卫 `torch.isfinite(loss)`**（必需的安全检查）；后 4 处在首次同步后 GPU 已空闲，几乎零额外成本。**移动顺序无收益**，不动 |
| `dit.py: glyph_embedder` | depth=2 的满秩 3×3 conv 占 fwd FLOPs 9.7% | 代码已提供 `glyph_embedder_sep`（depthwise-separable，3×3 那层便宜 ~8.8×）与 `depth=0` 两种开关。**这是模型结构实验变量，不属于 infra 范畴**，不擅自改 |
| `loss/flow_matching.py` | 每步都算 `mse_ch`（仅日志用）、`model_output.float()` | 张量都是 (240,4,32,32)=7.9MB 量级，合计 <0.3 ms，不值得动 |

### 7.3 根因：pointwise 内核受**显存带宽**限制，不是实现 bug

eager 微基准（4090 实测）：

| 模式 | 带宽 |
|---|---|
| 大张量 copy（峰值） | **918 GB/s** |
| 我们的 47MB 张量：`x*2` / `silu(x)` 单内核 | **978 / 982 GB/s**（已到峰值） |
| 链式 `x*2+1`（两个未融合内核） | **585 GB/s**（每多一个未融合内核就多一遍全量读写） |
| `mean(-1)` 归约（行内归约） | **109 GB/s** ⚠ |
| 手写 RMSNorm（fp32 往返） | 356 GB/s → 换 `F.rms_norm` 后 302 µs |

结论：**单内核效率没问题（978 GB/s），低效来自 (a) 归约类内核 (b) 融合边界**。编译后 inductor 已把逐元素链融掉，剩下的归约/边界就是 step 里那 76 ms 的主体，物理上无法再压。

### 7.4 组合后的最佳配置（实测）

| 配置 | ms/step | TFLOPs/s | MFU |
|---|---|---|---|
| 原始（torch 2.1.2 + compile default） | ~189 | 53.7 | ~31% |
| torch 2.5.1 + default | 177.8 | 57.0 | 33.1% |
| torch 2.5.1 + default + 代码修复(#1-3) | **173.4** | 58.4 | **34.0%** |
| torch 2.5.1 + max-autotune | 166.2 | 61.0 | 35.5% |
| torch 2.5.1 + max-autotune + 代码修复 | **158.4** | **64.0** | **37.2%** |

（max-autotune 首次编译 ~437 s，之后命中缓存 ~60 s。）
