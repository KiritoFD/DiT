# 71. Infra 深挖：显存、同步、H2D、数据 id —— 到底是怎么回事

> 日期：2026-09-17
> 相关：[70](70_code_review_refactor.md)（代码审查）、[59](59_12ch_and_white_zero_retrospective.md)
>
> **本文的纪律**：只写**实测过**的结论。推测一律标注，并给出验证方法。

---

## 0. 结论速览

| # | 结论 | 证据强度 |
|---|---|---|
| 1 | **"显存空洞"是误称** —— 那是每步激活的工作集被 allocator 缓存复用，是**正常且有益**的 | 实测 |
| 2 | 三种配置**稳态**的每样本显存几乎一样（56–60 MB/样本） | 实测 |
| 3 | xattn @batch360 的 OOM 来自 **compile-warmup 的瞬时峰值**（85 MB/样本，1.5×），不是稳态 | 推算 + 实测 |
| 4 | `expandable_segments` **不是** batch360 能跑的原因；它只让 `Mem` 每步波动 | 实测（关掉后照跑） |
| 5 | 新数据集的 id 与**所有旧资产**不兼容 → 三处静默错位风险 | 实测 |
| 6 | `--resume-full` 的 `strict=False` 会**静默跳过**形状不匹配的张量 | 代码 + 实测 |

---

## 1. 显存：先分清三个量

```
memory_allocated()      活跃   —— 当前真正被张量占用的
memory_reserved()       保留   —— allocator 向驱动要的总量（含缓存的空闲块）
max_memory_reserved()   高水位 —— reserved 的历史峰值（不会回落）
```

训练日志里的 `Mem: X/Y` 报的是 `reserved / max_reserved`。

### 实测（v13 base @batch360，`--alloc-debug`）

```
[alloc] 循环前基线: reserved 0.29G | 活跃 0.28G | 高水位 0.29G
[alloc] step 1  : reserved 0.93G -> 0.87G | 活跃 0.84G | 高水位 18.70G
[alloc] step 10 : reserved 19.13G -> 1.04G | 活跃 0.84G | 高水位 19.21G
[alloc] step 50 : reserved 19.17G -> 1.05G | 活跃 0.84G | 高水位 19.21G
```

**怎么读**：
- **循环前只有 0.29G** —— 只有参数 + 优化器状态
- **step 1 内高水位就冲到 18.70G** —— 一次前反向的激活峰值
- **日志的 `活跃 0.84G` 很小，是因为日志打在 `zero_grad` 之后**（激活已释放，
  只剩参数+优化器+EMA ≈ 36.4M×4B×4份 ≈ 0.6G）。**这不代表训练只用 0.84G。**
- 步间 `reserved` 保持 ~19G = allocator **缓存着激活块等下一步复用** —— 正常且有益
  （否则每步都要向驱动重新申请 19G）

### ⚠ 教训
**别把 `reserved - allocated` 当成"泄漏"。** 在 `zero_grad` 之后量它，得到的
永远是"参数+优化器"那点量，差值全部是缓存的激活块。

---

## 2. 各配置的真实显存画像

| 配置 | batch | 稳态 reserved | **每样本** | warmup 瞬时峰值 | **每样本** |
|---|---|---|---|---|---|
| v13 base（adaLN4, 4ch） | 360 | **20.14 G** | 56 MB | 18.70 G | 52 MB |
| v12（adaLN4, 4ch） | 360 | 20.42 G | 57 MB | 22.76 G | 63 MB |
| 12ch（adaLN4） | 360 | 20.42 G | 57 MB | 22.76 G | 63 MB |
| xattn | 240 | 14.42 G | 60 MB | **20.51 G** | **85 MB** |

**两条结论**：
1. **稳态下三种配置的每样本显存几乎一样（56–60 MB）** —— cross-attention 与
   aux 通道对**稳态**显存的影响很小（≤7%）
2. **但 xattn 的 warmup 瞬时峰值是 85 MB/样本（1.5×）** →
   batch360 时 `85 × 360 ≈ 30.6 G > 24.5 G` → **OOM**

**→ xattn @360 的 OOM 是编译期瞬时峰值造成的，不是稳态需求装不下。**

---

## 3. 我犯过的错（记录在案）

| 错误说法 | 真相 | 怎么发现的 |
|---|---|---|
| 空洞来自 `torch.compile` 的 **Triton autotune** | `mode="default"` **不做** autotune（那是 `max-autotune`） | 查 torch 文档 |
| `20.51→14.42` 是"eval 释放的缓存" | 方向对、**定性错**：那是**编译期瞬时峰值**留下的高水位，不是稳态需求 | `--alloc-debug` 打出 step1 高水位 18.70G |
| 用 `expandable_segments` **消灭**了空洞，从而让 batch360 可跑 | 它只是让 allocator **更激进地归还空闲段** → 让 `Mem` 每步波动；**且不是 batch360 能跑的原因**（v13 关掉它照样跑在 20.14G） | 关掉后实测 `Mem: 20.14G/20.14G` 零波动 |
| 用 `train.py --global-batch-size 16` 起小探针复现启动 | **卡死在 preload** —— REPA 的 DINO 缓存是 `mode=pinned`（29.6GiB **独占**的 page-locked 主机内存），训练已占一份，第二份申请会阻塞 | 探针卡死 7 分钟无进展 |
| 拿 `--resume-full` 加载 4ch→12ch | `strict=False` 会**静默跳过**形状不匹配的 3 个张量 → 停在随机初始化，不报错、loss 正常下降 | 读代码 |

### 附带的一个坑
**在 Windows 侧写 `.sh` 会变成 CRLF** → bash 报
`set: -: invalid option` / `python\r: No such file or directory`，
tmux 起来又立刻退出 → **重启静默失败**（看起来"已启动"但实际没跑）。
**写完必须 `bash -n` 校验 + 确认 LF。**

---

## 4. 数据 id：三处静默错位风险

新 50k 数据集的文件名是 **6 位补零**（`data/50k/imgs/000000.png`），
`extract_img_id` 解析出 `0, 1, 2, ...`；而**所有旧资产都按旧 id 建**：

| 资产 | 旧 id 空间 | 直接复用的后果 |
|---|---|---|
| DINO 缓存 `data/dino_cache/base_sym_v1` | 77 … 7,154,891 | 新 id 命中旧 id 的**另一张图** → REPA teacher 静默错位 |
| `data/aux/aux_canny_latents_base` | 2,259 … 266,225 | aux 查表大量落空 / 命中错行 |
| `data/aux/inst_skel_latents_px60` | 60,008 … 60,938 | 同上 |

**v13 必须全部重建**（DINO 缓存已重建为 `data/dino_cache/50k_v1`；aux 待做）。

⚠ 而且新数据经过描满/去噪，**图本身也变了** —— 旧特征即便 id 对得上也不可用。

---

## 5. 能跑对的关键设计（已落地）

| 机制 | 位置 | 作用 |
|---|---|---|
| `--expand-from-4ch` | `cli.py` + `train.py` + `src/utils/channel_expand.py` | 显式做 4→12 通道扩展，**不靠 `strict=False`** |
| `--alloc-debug` | `cli.py` + `train.py` | 转储按尺寸分桶的分配统计 + `memory_summary()`，定位显存问题的**唯一有效工具** |
| `empty_cache_after_warmup`（默认 **0**） | `train.py` | 关掉。开着会让 `reserved` 每步波动 |
| `PYTORCH_CUDA_ALLOC_CONF` | **不设置** | 设 `expandable_segments` 会让 `Mem` 每步波动；需要时用 env 显式开 |

---

## 6. 还没搞清的

| 问题 | 现状 | 怎么查 |
|---|---|---|
| xattn 的 warmup 瞬时峰值为什么是 v13 的 1.6× | 只知道现象（85 vs 52 MB/样本），不知机制 | 用 `--alloc-debug` 在 xattn 上跑一次，看尺寸分桶 |
| 能否让 xattn 也上 batch360 | 峰值 30.6G 装不下 | 试 `compile_mode: reduce-overhead`（cudagraph 固定池）或 warmup 用更小 batch |
| 12ch 从头训为什么全面落后 4ch | 有数据（80k 时 strict −0.056 / seen −0.144），无机制解释 | aux 占 52% final-layer 梯度是候选，但未验证 |
