# 919 / 50 — 基础设施、性能与运维手册

> 快照日期 2026-09-19。远端 `ssh 4090`（不写主机/账号/端口），仓库 `/root/Workspace/xy/DiT`。

---

## 1. 硬件与环境

| 项 | 值 |
|---|---|
| GPU | **RTX 4090 24G**（单卡） |
| 系统 | Ubuntu 18.04，**glibc 2.27** |
| driver | 590.48 / CUDA 13.1 |
| 仓库 | `/root/Workspace/xy/DiT`（本地镜像 `G:\GitHub\DiT`） |

### 1.1 两个 conda 环境（**Golden Rule：base 永不动**）

| | base `/opt/conda` | **cu121 `/opt/conda/envs/cu121`** |
|---|---|---|
| python | 3.10.8 | 3.10.18 |
| torch | **1.13.1+cu117** ← 别动 | **2.5.1+cu121** |
| torchvision | 0.14.1+cu117 | 0.20.1+cu121 |
| torchaudio | 0.13.1+cu117 | 2.5.1+cu121 |
| triton | 2.1.0 | 3.1.0 |
| cudnn | — | 90100 |
| xformers | 0.0.16（未动） | **已卸载** |
| 用途 | 旧脚本 / CPU metrics daemon / 数据工具 | **一切训练**（需 `--compile true`） |

⚠ **base 是系统组件** —— 它被大量旧脚本依赖，升级会连带炸掉整个工具链。

### 1.2 glibc 2.27 硬墙

```
torch 2.6+ 的 wheel 是 manylinux_2_28 → 需 glibc ≥ 2.28 → 本机 2.27 → 无法加载
→ 2.5.1 是可用最高版
```

xformers 可卸载的理由：`src/model/modules.py:resolve_attn_impl` 在 `attn_impl="sdpa"` 时
**永远走 SDPA**，xformers 只在 torch 无 `F.scaled_dot_product_attention` 时兜底；
且 `train.py` 第 2 行本来就有 `os.environ["XFORMERS_DISABLED"]="1"`。

### 1.3 备份与回滚

| 项 | 路径 |
|---|---|
| cu121 硬链接备份 | `/opt/conda/envs/cu121_bak_20260916`（仅 17M） |
| 升级前 freeze | `/root/cu121_freeze_before_20260916.txt` |

```bash
# 回滚 cu121
rm -rf /opt/conda/envs/cu121 && mv /opt/conda/envs/cu121_bak_20260916 /opt/conda/envs/cu121
# base 误动时的还原
/opt/conda/bin/pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    xformers==0.0.16 --index-url https://download.pytorch.org/whl/cu117
```

### 1.4 训练必备环境变量

```bash
export PYTHONPATH=/root/Workspace/xy/DiT:$PYTHONPATH      # 无 setup.py / pyproject
export TORCHINDUCTOR_CACHE_DIR=/root/.cache/torch/inductor # compile 持久缓存
PY=/opt/conda/envs/cu121/bin/python
```

---

## 2. 编译（torch.compile）—— 项目最大的 infra 事故与修复

### 2.1 根因：torch 2.1.2 inductor 的**指数级递归 bug**

现象：编译 **>30 min 无任何输出**（不是报错，是纯 CPU 空转，**GPU 0%**）。

```
File "torch/_inductor/pattern_matcher.py", line 655 in percolate_tags   ← 同一帧递归上万层
```

```python
def percolate_tags(node, recompute_tag):
    for arg in node.all_input_nodes:
        if hasattr(arg, "meta"):
            arg.meta["recompute"] = recompute_tag
            percolate_tags(arg, recompute_tag)   # ← 无 visited 集 → 路径数指数爆炸
```

**触发条件**：模型被编译成**一整张大图**（fwd+bwd）。
独立 bench 脚本**必然触发**；训练脚本因图被切断成多段而侥幸跑过 ——
属于**随机地雷**，任何图结构变化都可能引爆。

### 2.2 两层修复

1. `src/train/train.py` 加 env 门控（默认关 pattern_matcher）：
   ```python
   if os.environ.get("DIT_PATTERN_MATCHER", "0") != "1":
       from torch._inductor import config as _ind_cfg
       _ind_cfg.pattern_matcher = False
   ```
   实测关掉后**同一图 65 s 编完**，step 时间**无损失**（181.1 vs 181.2 ms）。
2. torch 2.5.1 上游已修（`pattern_matcher=True` 也能 74 s 编完）。
   **门控保留无害**，且 pm=0 略快（178.7 vs 181.4 ms）。

### 2.3 编译缓存纪律

| 项 | 值 |
|---|---|
| 缓存目录 | **只用 `/root/.cache/torch/inductor`**（一次性目录会每次全量重编） |
| 冷启动（default） | ~74 s（2.5.1 后 ~42 s，`fx_graph_cache` 生效） |
| 热启动 | **~42 s**（Dynamo 追踪 + AOTAutograd + lowering，torch 2.5 无磁盘缓存） |
| max-autotune 首次 | 437 s，之后 ~60 s |
| 稳态重编译 | **无**（v12 日志 5.5±0.1 step/s 平稳） |

---

## 3. 性能实测

### 3.1 环境升级收益

| 配置 | batch | Steps/Sec | Mem | 备注 |
|---|---|---|---|---|
| base (torch 1.13.1) | 192 | ~3.3 | 20.21G | 旧运行态 |
| cu121 (torch 2.1.2) | 192 | 3.60 | 19.89G | 纯迁移 |
| cu121 + compile | 192 | **8.50** | **9.73G** | **速度 ×2.6，显存 −52%** |
| cu121 + compile | 384 | 4.22 | 18.88G | 旧拍板 batch |
| torch 2.5.1 + compile | 240 | **5.83** | 13.27G | **+10%** vs 2.1.2 的 5.30 |

**显存大幅节省来自 SDPA flash**（不再物化完整 attention 矩阵）+ inductor 融合。

### 3.2 MFU 定位（为什么 80% 不可达）

| 配置 | ms/step | TFLOPs/s | MFU |
|---|---|---|---|
| eager（不编译） | 473.9 | 21.4 | 12.4% |
| compile `default` | 178.2 | 56.9 | 33.1% |
| compile `reduce-overhead` | 181.5 | 55.8 | 32.5%（CUDA graph 未生效） |
| compile `max-autotune` | **166.2** | **61.0** | **35.5%** |
| 真实训练（2.5.1 全链路） | ~171.5 | 59.1 | **34.4%** |

**实测峰值**（分母）：8192³ bf16 GEMM 持续 10 s → **172.0 TFLOPs/s**；
本模型 GEMM 形状（M=61440, K=384, N=1152/1024/384）→ **157.2 TFLOPs/s（91% 峰值）**。

→ **RTX 4090 + 本模型规模的实际天花板 ~40–45%**，不是 80%。

### 3.3 FLOPs 模型（预测步速）

**FLOPs ∝ d · h²** —— 已验证：S/2 vs M/2 比值 0.790 → 预期 1.266×，
实测 5.28/4.15 = **1.272×** ✓

| variant | d | h | 参数量 | 相对 M/2 |
|---|---|---|---|---|
| XS6/2 | 6 | 384 | 20.60M | 0.395× |
| XS/2 | 8 | 384 | 25.92M | 0.527× |
| S320/2 | 12 | 320 | 25.93M | 0.549× |
| **S/2** | 12 | 384 | **36.55M** | 0.790× |
| M/2 | 12 | 432 | 47.05M | 1.000× |
| Sp/2 | 12 | 512 | 66.70M | 1.405× |

### 3.4 VRAM 实测 batch（目标 22G / 24G 卡）

用**同一份 train.py 本体**扫描（REPA + EMA + compile + preload 全在），
只改 `--global-batch-size`，从日志读 `Mem: X/Y`。

| 配置 | batch | 显存 | 说明 |
|---|---|---|---|
| **S/2 @ 4ch（base，v12/v13/v14 用）** | **360** | 20.08G / 22.42G | 稳态 |
| XS/2 @ 4ch（v12_d8） | **576** | 21.14G | d8 更省显存 |
| S320/2 @ 4ch（v12_w320） | **440** | 21.40G | |
| S/2 @ **12ch**（v12_12ch） | **360** | **22.78G** | 沿用 base 的 batch（非最优） |
| S/2 @ xattn（v12_xattn） | **260** | 22.60G | xattn **+33% 算力** |

⚠ **batch 按实验分别定档**，不要跨配置复用 batch。

---

## 4. 长 epoch 重构（2026-09-18/19）

**问题**：`DataLoader` 的 epoch 由 `len(dataset)/batch` 决定（50,786/360 ≈ **141 步**），
而 ckpt/eval 每 **5000** 步一次 → **loader reset 与 ckpt 点不对齐** →
每 141 步一次 re-shuffle 抖动，实测 **~11% 的步进时间锯齿**。

**解**：`src/utils/samplers.py` 新增 `LongEpochDistributedSampler`，
令 **epoch = 固定步数 = `ckpt_every`**（`epoch_steps=5000`）→ loader reset 与 ckpt/eval 完全对齐。

**配套修复**：
- `--resume-full` 现在**读 ckpt 顶层 `train_steps`**（原先解析文件名 → 非 5000 整数倍会错）
- `--fresh-scheduler` 现在**也归零步骤计数器**，且发生在调度器构建**之前**
  → ⚠ **不要与 `--resume-full` 同用**（若 `max_steps` 是绝对步数）
- 已修复 config 中 `max_steps` 40000 → 200000 的问题（用绝对步数而非依赖代码路径）

**验证**：`_review/test_longepoch.py`、`_review/_dbg_sampler.py`、`_review/epoch_boundary_audit.py`。

---

## 5. 运维手册（remote.md 的三条铁律）

### 5.1 铁律

1. **绝不在 PowerShell 内联写多行 python/bash**（`python3 -c "..."`、内嵌 heredoc）
   → 一律**本地 write 成 `.py`/`.sh` → `scp` → 远程执行**。
   （本项目实测：多层嵌套引号会导致 `bash: unexpected EOF` / PowerShell 转义吞字符）
2. **远程文件保持 LF；绝不在远程跑 `sed -i "s/\r//g"`**
   → 经 Windows→ssh 传参时 `\r` 变 `r`，实际执行 `s/r//g`，**删光所有字母 r**
   （`root→oot`、`export→expot`，脚本瞬间报废）。
3. **长任务挂 tmux，必须 `TERM=xterm` 前缀**（非交互 ssh 下 `TERM=dumb` 会让 tmux 起不来）。

### 5.2 命令写法

```powershell
# 简单命令
ssh 4090 'tail -5 /tmp/x.log'
# 内层双引号要转义
ssh 4090 'grep \"step=\" /tmp/x.log | tail -3'
# 长任务（tmux）
ssh 4090 'TERM=xterm tmux new-session -d -s v14 \"bash _sync_work/_launch_v14.sh\"'
# 备选：setsid nohup ... > /tmp/x.log 2>&1 < /dev/null &
```

### 5.3 路径策略（不依赖时间戳）

训练器会按时间戳建子目录（`<results_dir>/20260918-234236-.../checkpoints/`）。
**禁止 `ls -dt` + glob 猜目录**（glob 少一层就断链）。
→ **所有跨阶段 ckpt 先 copy 到固定无时间戳路径**再向下传。

### 5.4 监控

```bash
# GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
# 判断在跑哪一段
pgrep -f 'src/train/train.py'
# 训练日志
tail -3 /tmp/v14_fullft.log
```

⚠ **已知显示怪象**：`nvidia-smi` 显示 20097MiB 但 `--query-compute-apps` 无进程 ——
该容器环境的已知行为，**以训练日志的 `Mem:` 为准**。

⚠ **不要在训练机上跑与训练争 GPU/CPU 的基准**（先杀基准进程再测）。

### 5.5 常见坑速查

| 症状 | 根因 | 修复 |
|---|---|---|
| `cd: /oot/Wokspace/...` | 远程 `sed s/r//g` 删了 r | 重传干净文件，不再 sed |
| tmux `no server running` / `lost server` | `TERM=dumb` | `TERM=xterm tmux new-session -d ...` |
| `python -c` syntax error | PowerShell 内联多行坏了 | write 脚本 + scp |
| ssh 命令超时挂死 | 命令含 `$(...)` 或内层引号被吞 | 拆成单引号简单命令或写脚本 |
| 时间戳目录选错/选空 | `ls -dt` 猜目录少一层 | 固定路径 copy |
| config 键改了不生效 | **键未注册 argparse** → 静默丢弃 | 注册 + 断言（见 `30_training.md` §4） |

---

## 6. 目录与产物约定

| 内容 | 路径 |
|---|---|
| 训练主日志 | `<results_dir>/<ts>-<exp>/log.txt` |
| 训练 stdout | `/tmp/<exp>.log`（如 `/tmp/v14_fullft.log`） |
| ckpt | `<results_dir>/<ts>-<exp>/checkpoints/00XXXXX.pt`（含 ema / optimizer / scheduler / `train_steps`） |
| 评测汇总 | `<results_dir>/eval_stdskel_summary.csv` |
| poster | `<results_dir>/posters/*.png` |
| 多样性 | `assets/diversity_*.csv` / `_summary.json` |
| master 表 | `assets/master_results.csv` / `.json` |

---

## 7. 数据体量（磁盘）

| 项 | 大小 |
|---|---|
| REPA DINO 缓存（px60, base_sym_v1） | 29.6 GiB（161,766 patches） |
| REPA DINO 缓存（50k_v1） | 9.3 GiB（51,036 patches） |
| DINO CLS 特征（dino_cls_50k.npz） | 76 MB |
| latent shards（img + std，50k） | 见 `10_data.md` §3.5 |

---

## 8. 一次完整训练的时间预算

| 阶段 | 时长 |
|---|---|
| 冷启动 compile | ~74 s（热 ~42 s） |
| S/2 @ batch 360 跑 100k 步 | **≈ 6.7 h**（4.14 step/s） |
| v14 full-ft（60k → 160k） | ≈ 6.8 h |
| v14 三阶段全程 | ≈ 20 h |
| 数据 encode（px60 shards） | 多小时（VAE encode 主导） |
| 实例骨架 latent 构建 | 224 s |