# 922 / 96 — SkelNet：骨架形变网（DeformSkel）完整说明

> 一句话：**用书家风格把"标准骨架"形变成"这个书家写的这个字的骨架"**，
> 再把形变后的骨架喂给主网络当条件。
>
> 它是本项目**第一个被证明真的在用书家条件**的模块。

---

## 1. 为什么需要它（数据证据，不是猜测）

### 1.1 关键发现：`g` 是**跨书家共享的规范字形**

`_sync_work/probe_g_source.py` 实测（按 md5 去重）：

| 骨架目录 | 只有 1 张的字占比 | std 数/样本数 | 前景占比（宽度） |
|---|---|---|---|
| **`std`（当前的条件 g）** | **43.5%** | **0.481** | 0.0642（≈4px）|
| `inst_skel1` | 20.2% | **0.996** | 0.0096（1px）|
| **`aux_skel3`** | 20.2% | **0.996** | 0.0365（≈3px）|
| `inst_skel20` | 20.2% | 0.996 | 0.2377（20px，太粗）|

**具体例子**：字「出」有 22 个样本、来自 **11 个书家**，但只有 **3 张不同的 std 骨架**
——**李邕和褚遂良共用同一张**（md5 都是 `1334bfa8cf`）。

→ 结论：**`g` 只携带"这是什么字"，不携带"谁写的"**。
   通道其实已经分开了；**风格只能来自 `y_callig`**。

### 1.2 因此主网络可以"照着 g 描"，永远不需要学书家

`g` 给的是**精确的字形结构**，而重建 loss 被结构主导 → 主网络只要描 g 就能把 loss 压下去。
这解释了此前所有现象：

| 观测 | 数值 |
|---|---|
| 书家条件的实测强度（correct − shuffled）| **只有 +0.011** |
| 生成结果在同字池里的"跟随书家"命中率 | **≈ 随机** |
| `own_gt − 同字均值`（目标特异性）| +0.012，**只有"认字"能力的 1/3** |

### 1.3 SkelNet 的立足点：**它自带梯度**

前面所有风格模块（`style_ln` / `spatial_film` / `87-pair` / `style-rank`）都失败了，
根因是同一个：**扩散 loss 不奖励风格**。它们必须靠 loss 之外的信号，而那信号量不到风格。

SkelNet 不一样：**g 是规范字形，而目标是该书家写的那个字** →
"把 g 形变到更接近目标"**直接降低重建 loss**。这是唯一一个不需要额外监督就能拿到梯度的风格干预。
再加上出口的**稠密中间监督**（见 §4），它有两个信号。

---

## 2. 架构

```
                    ┌──────────────── 书家风格 e (128d, 预训练表, 冻结) ────────────────┐
                    │                                                                  │
                    │   ┌─ FiLM 逐层调制 ─────────────────────────┐                     │
                    │   │                                          │                     │
 g_std (4,32,32) ───┼──→│ 小 U-Net (width=96, 3.58M) ──→ 偏移场 (2,32,32) ──┐          │
                    │   └──────────────────────────────────────────┘              │          │
                    │                                                              ├─(+)─→ off
                    │   风格专属偏移底图 Linear(e)→(2,32,32) ─────────────────────┘          │
                    │                                                                  │
                    │                            tanh 限幅 ±max_off=6                       │
                    │                                    │                              │
                    │                          grid_sample(g_std, base+off)                │
                    │                                    │                              │
                    │                                    ├─────────(+)──────────→ g'       │
                    │   内容残差 Conv → (4,32,32) ────────┘         ▲                     │
                    │   风格专属残差 Linear(e)→(4,32,32) ───────────┘                     │
                    └──────────────────────────────────────────────────────────────────┘
                                                                    │
                              中间监督: MSE(g', aux_skel3) ←────────┘
                              主网络条件: concat / adaLN 注入 / local_ca 都用这个 g'
```

### 五条路径，各自解决一个问题

| # | 路径 | 作用 | 证据 |
|---|---|---|---|
| 1 | **FiLM 逐层调制** | 让风格**进得去** | 纯"广播拼接"会被卷积层直接忽略：v1 的 `correct ≈ shuffled`、follow ≈ 50% |
| 2 | **风格专属全分辨率偏移场** | 表达"这个书家的整体间架倾向" | 风格部分占 offset 的 0.31/0.40 |
| 3 | **内容自适应偏移残差**（U-Net）| 逐字修正"这一笔该拉长" | — |
| 4 | **加性残差** | 补 **2D 形变做不到的**笔画粗细/墨色 | 闭合率 36.4% → 52.7%，follow 62.9% → 97.2% |
| 5 | **风格专属全分辨率残差** | 每个书家的"笔法底图" | — |

**为什么必须有第 4/5 条**：2D 形变**只能移动像素位置，改不了笔画粗细/压力/墨色**，
而书家风格里很大一部分正是这些。只做形变时闭合率卡在 **36.4%**（见 §6 演进表）。

### 为什么用"小 U-Net 输出偏移场"而不是直接输出图像

- 输出偏移场 → **拓扑由 g 保证**，不可能凭空画出别的字（这正是"g 给拓扑、风格给排布"的解耦语义）
- 若直接用 U-Net 输出 g′，它可能绕过 g 自己生成 → 退化成又一个独立生成器
- 用 flow 模型是杀鸡用牛刀：目标是**确定性映射**，不是从噪声生成

---

## 3. 参数

### 3.1 网络结构参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `deform_grid` | **32** | **骨架 latent 的空间尺寸**。⚠ 不是 token 网格（模型里的 `_grid`=16）|
| `deform_width` | 96 | U-Net 首层通道数（容量旋钮）|
| `deform_coarse` | 8 | 偏移先在 8×8 上预测再上采样 → 天然平滑 |
| `deform_max_off` | **6.0** | 偏移限幅（latent 像素）。32 网格下 1px ≈ 8 图像px → ±6 ≈ ±48px |
| `deform_style_ch` | 32 | 风格广播图的通道数 |
| `residual` | 1 | 是否加性残差（**必需**，见 §2）|
| `res_cap` | 2.0 | 残差限幅 |
| `cond_dim` | 128 | 风格向量维度（= `callig_emb_pretrained_50k.pt` 的列数）|

参数量：**3,578,214**（其中风格专属底图 264,192 + 风格专属残差 262,144）。

### 3.2 训练参数（离线预训练）

| 参数 | 值 | 说明 |
|---|---|---|
| `--steps` | 8000 | 约 38 分钟 |
| `--batch` | **16384** | 显存 21GB（4096 时只有 6.6GB，白浪费）|
| `--group` | 96 | **同字分组采样**：每批由 96 个"字"组成，每字取多个书家 |
| `--lr` | 1e-3 | AdamW + cosine，`weight_decay=0.01` |
| `--style-emb` | `callig_emb_pretrained_50k.pt` | **冻结**，必须与线上 `_e_callig()` 同源 |
| `--std-dir` | `data/50k/shards_std_fixed` | 输入骨架 |
| `--gt-dir` | `data/50k/shards_aux_skel3` | **监督目标** |

**为什么必须同字分组采样**：5750 个字 / batch 512 时，同一批里几乎不会出现同一个字
→ 同一个 `g_std` 不会对应多个 `g_gt` → **风格对降低 loss 没有帮助，模型直接忽略它**。
v1 用随机采样时 `correct ≈ shuffled`，改成分组后 follow 从 50% → 62.9%。

### 3.3 接进扩散训练的参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `deform_skel` | 0 | 1 = 启用 |
| `deform_ckpt` | "" | 离线训好的权重（`assets/deform_skel_v5.pt`）|
| `deform_trainable` | 1 | 1 = 解冻；0 = 冻结（冻结时更该直接预算成 shards）|
| `deform_lr_scale` | **0.1** | 形变头 lr = 主 lr × 此系数，**单独一个 param group** |
| `w_deform_skel` | 1.0 | 中间监督权重 |
| `inst_skel_shards_dir` | `data/50k/shards_aux_skel3` | 中间监督目标 |

**为什么单独一个 lr 组**：头已在**另一个目标**上收敛，扩散 loss 对它是"扰动信号"而非有用信号
→ 用主 lr 会在头几百步破坏它。`LambdaLR` 按各组**初始 lr** 等比缩放，所以单独设初值即可自动跟随 cosine。

---

## 4. 中间监督（这是它区别于其它风格模块的关键）

**目标**：`aux_skel3` —— 从 **GT 图像**派生的骨架，宽度 ≈3px（最接近 `std` 的 ≈4px），
实测**逐样本**（比值 0.996，不像 `std` 那样跨书家共享）。

```
L_skel = ‖ g' − aux_skel3 ‖²         (只在"真的做了形变"的样本上平均)
```

**为什么这个监督有效**：step0 时 g′ = g_std ≠ g_gt → **立刻就有稠密梯度**，
不像其它风格模块那样"loss 不奖励风格所以学不动"。

**必须按掩码平均**：条件被 drop 的样本（CFG uncond 分支）拿到的是 **null 风格向量**，
形变头会产出垃圾骨架 → 不该进监督。

---

## 5. 怎么用（三种模式）

### 模式 A：离线预训练（推荐起点）
```bash
python tools/train_deform_standalone.py \
    --steps 8000 --batch 16384 --group 96 \
    --residual 1 --width 96 --max-off 6 --res-cap 2 \
    --out assets/deform_skel_v5.pt
```
约 38 分钟，产出 `assets/deform_skel_v5.pt`。

### 模式 B：冻结接进扩散训练 —— **更好的做法是直接预算成 shards**
冻结时 `g' = deform(g_std, 书家)` 是**确定**的，而每个训练样本的书家是固定的
→ **g′ 也是固定的** → 没必要在训练时反复算：
```bash
python tools/cache_deform_shards.py --ckpt assets/deform_skel_v5.pt \
    --out data/50k/shards_deform_v5
```
然后把 `skel_latent_shards_dir` 指向它即可 —— **零额外算力、模型里不用挂头、也没有那个 loss**。

### 模式 C：解冻联合训练（当前 v18-skelnet）
```bash
python -m src.train.train --config src/train/configs/v18_skelnet_100k.json
```
配置 = 严格 E0 + 可训练 SkelNet（`gbs=320`，`deform_lr_scale=0.1`）。

---

## 6. 效果

### 6.1 三个判据

| 判据 | 定义 | 含义 |
|---|---|---|
| **闭合率** | `1 − MSE(g′,g_gt) / MSE(g_std,g_gt)` | 形变消掉了多少"规范骨架 → 该书家骨架"的距离 |
| **style-follow** | 同一个字，`g′(c,k)` 是否比 `g′(c,k′)` 更接近 **k 自己的** `g_gt(c,k)` | 直接回答"同一个 skel 是否对不同书家产出对应 skel" |
| **残差/下界** | `MSE(g′,g_gt) / 不可约下界` | 是否已到信息论极限 |

### 6.2 ★ 不可约下界（判"够不够"的关键）

头的输入是 `(g_std, 书家)`。但 **`g_std` 按字共享**，而同一个 (字, 书家) 常有**多个样本**、
它们的 `g_gt` 各不相同 → 头只能输出**同一个** g′ → 最多只能预测**组内均值**
→ 残差有不可约下界 = **组内方差**。

`tools/diag_skelnet_floor.py` 实测：
```
同(字,书家) 分组: 35379 组, 其中 10315 组有多样本（共 25722 条）
下界·同字同书家 = 0.06505      <- 可达
下界·仅同字     = 0.22582
基线 MSE(g_std,g_gt) = 0.50494
-> 闭合率天花板 = 1 − 0.06505/0.50494 = 87.1%
```

### 6.3 演进表（每一步都有明确收益）

| 版本 | 配置 | 闭合率 | style-follow |
|---|---|---|---|
| v1 | 纯形变，风格只广播拼接，batch 512 | **−0.4%**（=恒等）| **50%**（=随机）|
| v2 | + FiLM + 风格底图 + **同字分组采样** | 35.0% | 62.9% |
| v3 | + **加性残差** | **52.7%** | **97.2%** |
| v4 | 风格改全分辨率偏移场+残差（激进）| — | — |
| **v5** | **batch 16384 + width 96 + max_off 6** | **66.9%** | **99.8%** |

**v5 最终明细**：
```
MSE(g_std,g_gt)=0.50494  ->  MSE(g',g_gt)=0.16700   闭合率 66.9%
correct 0.16700  vs  shuffled 0.35773   （差 2.14×）
输出变化 0.2568 | offset 0.4729（风格部分 0.1724）
style-follow 99.8%
残差/下界 = 2.57×   已达到可达上限的 77%
```

### 6.4 怎么读这些数

- **99.8% 的 style-follow** = 你要的那个性质**完全成立**：同一个 skel、用书家表、对不同书家产出对应 skel
- **correct 与 shuffled 差 2.14×** = 风格是真的在起决定作用（对比扩散模型：correct−shuffled 只有 +0.011）
- **残差/下界 = 2.57×** = **还没饱和**，闭合率能从 67% 推向 87%

---

## 7. 踩过的坑（都修了，都值得记住）

| # | 坑 | 症状 | 修法 |
|---|---|---|---|
| 1 | **base grid 用错约定** | `linspace(-1,1,N)` 是 `align_corners=True` 的约定，而 `grid_sample` 用 False → 差半像素 → "零偏移"时 `\|out−g\|=2.5` 而非 0，**"step0 恒等"这个性质没了** | 改成 `(2*(i+0.5)/N)−1` |
| 2 | **风格只广播拼接** | 卷积层直接忽略它 → `correct ≈ shuffled`，follow ≈ 50% | FiLM 逐层调制 + 风格专属底图 + **同字分组采样** |
| 3 | **以为 2D 形变够了** | 闭合率卡在 36.4% —— 形变改不了笔画粗细/墨色 | 加**加性残差** |
| 4 | **接线缺口** | dit.py 加了形参但 train.py 构造模型时没传 → 冒烟测试的 `[deform] 模型没有 deform_skel` 告警抓到 | 三处透传（dit/train/model_io）|
| 5 | **`inst_skel` 没加载** | 它只在 `w_latent_skel>0 or w_std_mid>0` 时加载 → 开了 `w_deform_skel` 但两者为 0 时 `batch['inst_skel']` 不存在 → 中间监督**静默失效** | 把 `w_deform_skel` 也加进加载条件 |
| 6 | **`deform_grid` 用了 token 网格** | 模型里 `_grid`=16（32÷patch2），而形变作用在骨架 latent (4,32,32) 上 → `style_off` 建成 2×16×16 → 载入离线权重报 **size mismatch** | 加 `deform_grid` 形参（默认 32）|
| 7 | **`num_calligraphers` 属性不存在** | 它只是构造形参，`getattr(self,...,0)` 返回 0 → cond-drop 跳过形变的修复**变成空操作** | 用 `y_callig_embedder.num_classes` |
| 8 | **只改了 forward 没改 loss** | loss 用 `last_out`（全批都形变过的）→ 实测 loss 一模一样 0.3687，修复**完全无效** | 把掩码存到 `last_mask`，loss 按掩码平均 |
| 9 | **离线自学风格 embedding** | 模型 `_e_callig()` 给的是 `callig_emb_pretrained_50k.pt` 的行 → **分布不符，训好的头接进去会失效** | `nn.Embedding.from_pretrained(那张表, freeze=True)` |
| 10 | **多文件 scp 放错目录** | config 被丢进 `src/train/` 而不是 `src/train/configs/` → 训练用**默认配置**启动（ddpm/2021/legacy）| 目录结构不同的文件分开传 |
| 11 | **诊断脚本的 `.sum()` vs `.mean()`** | 组内方差用了 `.sum()` → 下界算出 **438.9**（比 MSE 尺度大三个量级）→ 据此打印了"已贴近下界、加训练没用"的**相反结论** | 改 `.mean()` |

---

## 8. 文件与命令

| 文件 | 作用 |
|---|---|
| `src/model/deform_skel.py` | 模块本体（五条路径）|
| `tools/train_deform_standalone.py` | **离线训练器**（不需要扩散模型，38 分钟出结果）|
| `tools/diag_skelnet_floor.py` | 算不可约下界，判"够不够" |
| `tools/cache_deform_shards.py` | 冻结场景下把 g′ 预算成 shards |
| `tools/verify_deform_integration.py` | 端到端集成验证（四项检查）|
| `src/train/configs/v18_skelnet_100k.json` | 解冻联合训练配置 |

```bash
# 离线训练
python tools/train_deform_standalone.py --steps 8000 --batch 16384 --group 96 \
    --residual 1 --width 96 --max-off 6 --res-cap 2 --out assets/deform_skel_v5.pt
# 判够不够
python tools/diag_skelnet_floor.py --ckpt assets/deform_skel_v5.pt
# 集成验证
python tools/verify_deform_integration.py
# 解冻联合训练
python -m src.train.train --config src/train/configs/v18_skelnet_100k.json
```

---

## 9. 当前状态与下一步

- **v5 头**：闭合 66.9% / follow 99.8%，`assets/deform_skel_v5.pt`
- **v6（续训中）**：从 v5 续到 20k 步（12000 步），看是否收敛到更接近下界
  - 关键观察量：`残差/下界`（现在 2.57×）是否下降
  - 若 20k 后仍在 2.5× 附近 → 说明**容量不够**，该加宽而不是加步数
- **v18-skelnet**：解冻联合训练，已跑到 20k（Deform loss 0.4078 → 0.2362，头在被改善而非破坏）

---

## 10. ★ 显存实测（2026-09-25，纠正此前"4096 是上限"的错误结论）

RTX 4090 24G（可用 23.52 GiB），**无梯度检查点**，含 TV/Jacobian 反向，每配置**独立进程**测峰值：

| batch | width | 结果 | 峰值 |
|---|---|---|---|
| 4096 | 96 | **OOM** | 22.8 GiB |
| 4096 | 80 | **OOM** | 22.6 GiB |
| 4096 | 64 | **OOM** | 22.2 GiB |
| 4096 | 48 | OK | 18.6 GiB |
| 3072 | 96 | **OOM** | 22.5 GiB |
| 2048 | 96 | OK | 18.2 GiB |

**根因**：`d1` 在 **32×32 全分辨率 × 96ch** 下，**单个激活张量 = batch × 96 × 1024 × 4 B**，
batch 4096 时就是 **1.61 GB/张**，而反向要保存 6–8 张
（conv1 / gn1 / gelu1 / conv2 / gn2 / gelu2 + FiLM 的乘和加）→ **单 d1 就 ~10 GB**。
前向 `no_grad` 单独就占 **12.3 GB**（batch 4096 / width 96）。

**关键点：显存几乎只随 batch 线性、对 width 很不敏感**（96→48 只省 ~5 GB），
因为大项是"保存下来的 32×32 激活张量数量 × batch"，width 只影响每张的大小。

**两种"不换架构"的省显存手段均无效（已实测）**：
- `nn.GELU(inplace=True)` → 峰值**一模一样**。原因：GELU 反向本来就能用**输出**算，省不掉保存量。
- `bf16 autocast` → 无效。原因：GroupNorm / `torch.cat` 仍走 fp32，瓶颈不在 conv 的算力精度。

**结论**：width 96 下 batch 4096 单次前向在 24G 上物理放不下。
要 batch 4096 只有三条路：① 降 width（会丢 v5 的 U-Net 暖启动，只留 `style_off` 底图和全局仿射）；
② 梯度累积 2×2048（**数学等价**，因为模型只用 GroupNorm，逐样本归一化）；
③ 梯度检查点（**用户已明确禁止**）。

**v9 最终采用**：单次前向 **batch 2048 / width 96 / ckpt 0**，实测 22.4 GiB 驻留、~0.50 s/step。
> 注意 22.4 GiB 已贴到 23.5 GiB 上限，eval 时还要把 VAE 搬上 GPU，**有余量风险**。
> 启动脚本里加了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 降低碎片导致的假 OOM（纯分配器设置，不改训练语义）。

### 顺带修掉的两个真 bug（`tools/train_deform_standalone.py`）

1. **`img_loss()` 定义了但从未被调用** —— 训练循环里只有一行孤儿注释"★ 图像空间监督（关键项）"，
   `--w-img` 是静默空操作。
2. **`_to_gray` 被 `@torch.no_grad()` 包住** —— 即使调用，图像 loss 也 `requires_grad=False`，
   **梯度根本回传不到模型**。已拆成 `_decode_gray(lat, grad=False)`：训练路 `grad=True`（走 `torch.enable_grad()`），
   诊断/评估路仍 `no_grad`；并在 `--w-img>0` 时让 VAE **常驻 GPU**（否则每步来回搬 160MB 太慢）。

---

## 11. ★★ 训练速度排查：0.496 → 0.358 s/step（1.39×）

### 症状
v9 首跑 batch 2048 / width 96 只有 **0.496 s/step（2 步/s）**、功耗仅 **272 W**。
对照：30M 的主网络能跑 4 steps/s —— 1.7M 的小 U-Net 反而更慢，明显不合理。

### 定位（先测，不猜）
1. **吞吐对 batch 完全线性**：batch 256/512/1024/2048 分别 4637/4231/4110/4140 samples/s。
   → 排除了"kernel launch 开销"和"显存分配器压力"两个假设（若成立，小 batch 的 samples/s 会更低）。
2. **算子级 profiler**（`torch.profiler`，10 步）：

| 算子 | 占 CUDA | 备注 |
|---|---|---|
| `convolution_backward` | 33.7% | 真实计算 |
| **`upsample_bilinear2d`** | **30.2%** | **40 次调用，平均 37.3 ms/次** |
| `nchwToNhwcKernel` | 9.4% | cuDNN 为用 TF32 张量核做的布局转换，**700 次/10 步** |
| conv forward (tf32 gemm) | ~10% | |
| `NativeGroupNormBackward` | 5.3% | |

### 根因：torch 2.5 的 **NCHW `upsample_bilinear2d` CUDA 核病态地慢**
batch 2048 实测：

| 形状 | bilinear(NCHW) | nearest | bilinear+channels_last | 固定核 convT |
|---|---|---|---|---|
| C=192, 8×8→16×16 | **97.03 ms** | 0.57 ms | 0.94 ms | 0.31 ms |
| C=96, 16×16→32×32 | **49.62 ms** | 1.14 ms | 1.91 ms | 0.27 ms |
| C=2, 8×8 | 0.61 ms | 0.01 ms | 0.43 ms | 0.02 ms |

两次大通道上采样 = 97 + 50 = **147 ms/step**，与 profiler 的 149 ms/step 完全对上。
换算：10 步输出 ~3G 个元素只用 1.49 s ≈ **2 G 元素/s**，连显存带宽的 2% 都不到。

### 修复：`_up()` —— 转 channels_last 再插值（**语义完全不变**）
```python
def _up(x, size):
    return F.interpolate(x.contiguous(memory_format=torch.channels_last),
                         size=size, mode='bilinear', align_corners=False) \
            .contiguous(memory_format=torch.contiguous_format)
```
同一个操作换布局会走另一条向量化核：**42× / 10× 更快**；数值一致到 **2.4e-07**（纯 float32 舍入，逐形状验证过）。
`off_u` 的 32→8→32 那两次（C=2）本来就便宜，一并走 `_up` 保持一致。

**结果**：494 → **353.5 ms/step**；`forward(no_grad)` 255 → **114.9 ms**；`upsample_bilinear2d` **从 profile 里彻底消失**。

### 试过但**无效/更差**的（都已实测，别再走一遍）
- **固定核 `conv_transpose2d` 替双线性**：快 78×，但**数值不等价**。
  `align_corners=False` 的双线性权重是**位置相关**的（偶数位 0.75/0.25、奇数位 0.25/0.75），
  均匀 stride-2 核复现不了；边界（首行/列 clamp）也不一致。相对误差 ~0.88，不可用。
- **整模型 channels_last**：**420 ms，反而慢 0.84×**。因为风格是 `style_proj(st).expand(...)` 的
  **跨步视图**，`torch.cat` 后布局退回 NCHW → 白付转换成本。只有 `_up` 这种**局部**转换才划算。
- **`cudnn.allow_tf32=False`**：353.5 → 354.0 ms，**无变化**（`nchwToNhwc` 那 13% 并没省掉）。
- **`GELU(inplace=True)`**：峰值与速度均无变化（GELU 反向本就能用输出算）。
- **bf16 autocast**：无变化（GroupNorm / `cat` 仍走 fp32）。

### 真实运行验证
`step 500`：**248 s → 179 s**（0.496 → 0.358 s/step），功耗 **272 W → 400 W**，驻留 23.0 GiB。

### 修复后的剩余瓶颈（下一步的线索）
conv backward 48.6% / conv forward 16.3% / **nchwToNhwc 13.1%** / groupnorm 12% / gelu bwd 4.3%。
> ★ **结构性浪费**：`coarse=8` 意味着 `off_u` 在 32×32 算完后立刻被降到 8×8 再升回 32×32，
> 所以 U-Net 在 32×32 上的计算**信息上被丢掉了**。按 FLOPs 算，32×32 那几个卷积占
> **54%**（u1 conv1 24.6% + d1 conv2 12.3% + u1 conv2 12.3% + d1 conv1 4.7%）。
> 把 U-Net 直接做在低分辨率（输出 8×8 控制点网格）理论上能再快 ~2×，但会改动架构。

---

## 12. ★★ 主模型是否有同样的 upsample 问题？—— 结论：**没有**

这个问题很关键：如果主模型也中招，整个训练吞吐都被拖累。逐条排查主训练路径：

| 位置 | 调用 | 中招？ | 依据 |
|---|---|---|---|
| **DiT 主干** | 无任何 interpolate | **否** | 全库 grep：`src/model/` 下只有 `legacy/controlnet.py` 和 `std_dino_embedder.py` 有，主干干净 |
| **VAE 解码** | `Upsample2D(interpolate=True)` ×3 | **否** | diffusers 实现用的是 **`mode="nearest"`**（`diffusers/models/upsampling.py:67/172/174`），nearest 走快核 |
| **REPA (DINO teacher)** | `losses.py:224` bicubic 256→224 | **否** | ① C=3 小通道，实测 B=32 只 **0.13 ms**；② v18 配了 `repa_cache_dir="data/dino_cache/50k_v1"`，**缓存命中时根本不调用 teacher 前向** |
| **MidStructureLoss** | `structure_mid.py:36` bilinear lowpass | **否（且默认关）** | C=4，实测 **0.09 ms**；`w_std_mid` 默认 0，只有 2 个配置开 |
| **LatentSkelStructureLoss** | `latent_structure.py:162` | **否** | v18 的 `aux_loss_weight=0.0` |
| `std_dino_embedder` | 1D `mode="linear"` | 否 | 1D、通道极少 |
| `legacy/*`、`eval/*` | `auto_eval_ctrl*.py` 等 | 不在训练路径 | — |

### 为什么主模型不中招
这个病态核的触发条件是 **大通道数 × 小空间尺寸 × NCHW 上采样**。
v9 正好是 **C=192/96 在 8×8/16×16 上放大**；主模型那几处要么**通道极少（3/4）**，
要么是**大空间下采样**，要么直接是 **nearest**。

### 实测（B=32；逐元素开销可线性外推）

| 调用 | NCHW | channels_last |
|---|---|---|
| REPA bicubic 256→224 | **0.13 ms** | 0.23 ms（**反而慢 0.6×**）|
| REPA bilinear 256→224 | 0.10 ms | 0.13 ms |
| MidStructure lowpass (4,16,16)→(32,32) | 0.09 ms | — |

外推到 B=320：REPA ≈ **1.3 ms**、MidStructure ≈ **0.9 ms**，相对主模型单步（~250 ms）可忽略。

> ⚠ **不要把 `_up()` 套到主模型上**：主模型这些调用本来就不慢，换 channels_last
> **反而更慢**（和"整模型 channels_last 慢 0.84×"同一个原因）。
> `_up()` 只适用于 **大通道 × 小空间 × NCHW 放大** 这一种形状。

---

## 13. ★★ 这个模块要达到什么水平，下游才可用？

### 三条硬指标（按重要性排序）

| 指标 | 无用 | **可用门槛** | 理想 | v5 | **v9 @2000** |
|---|---|---|---|---|---|
| **① style-follow** | 50%（=随机） | **≥75%** | ≥90% | 99.8%* | **51.7%** ❌ |
| **② 闭合率** | 0%（=直接用 g_std） | **≥60%** | ≥75% | 66.9% | **34.4%** ❌ |
| **③ 墨量比 pred/target** | <0.5 或 >2 | **0.85–1.15** | 0.95–1.05 | 0.40 ❌ | **0.04** ❌ |
| ④ 连通分量（越少越好，目标 ~1.7） | >10 | **≤3** | ≤2 | 28.4 ❌ | 未测（墨量太少） |

\* v5 的 99.8% 是**假的**，见下。

### ① 是"有没有用"的开关
`style-follow = 50%` 意味着 `g'(c,k)` 并不比 `g'(c,k')` 更接近 k 自己的 GT ——
即模型产出的是**"该字的平均风格"**，DiT 从条件里拿不到任何书家信息，**模块等于白做**。
75% 是"开始有信号"，90% 才算稳。

### ★ 顺带查出一个关键事实：v5 的 99.8% 是残差在"作弊"
v9 在 **step0**（刚载入 v5 的 61/66 个键，含 `style_off` / FiLM / `style_proj`）
测到的 style-follow 只有 **51.4%**。也就是说：
**v5 里那套"几何形变"路径（偏移场 + FiLM）本身并不带风格判别力**，
v5 的 99.8% 主要来自 `style_res` —— 它是**每个书家一张 4×32×32 的加性墨迹图**，
不移动任何笔画，只是"按书家往画布上补墨"。这天然让 `g'(c,k)` 更贴近 `g_gt(c,k)`，
指标爆表，但**几何根本没动**。这正好印证了"加性残差会凭空产生虚假线条、破坏拓扑"。

### ②③④ 的关系：为什么闭合率和墨量都上不去
去掉残差后只剩**纯几何形变**（`grid_sample` 只移动像素，**改不了笔画粗细/墨色**），
所以墨量比只有 0.04 —— 这是"没墨"而不是"碎"，是几何形变的**能力上限**，
不是训练问题。同理闭合率也只有 34.4%。

### ★ 为什么 style-follow 卡在 50% 不动（真正的病根）
目标函数是 `MSE(g', g_gt)`，而这个 MSE **被"该字的平均形变"主导**：
风格特异的那部分只是总方差里的一小块。
于是模型**走了捷径** —— 学到"字 → 平均形变"就能把 loss 降掉大部分，
风格特异的部分梯度太弱，直接躺平。这就是 style-follow 从 step0 的 51.4% 到
step2000 的 51.7% **完全没动**的原因（而偏移幅值 `off` 从 0.42 涨到 1.72，
说明它在动，只是动的方向不是"朝该风格自己的 GT"）。

**要修的方向**（不是加步数能解决的）：
- 目标函数**显式奖励风格判别**：例如按**字内中心化**（减去该字在所有书家上的均值）
  后再算 MSE，让 loss 只看见"风格特异"那一块；或加对比/排序项
  （`g'(c,k)` 应比 `g'(c,k')` 更接近 `g_gt(c,k)`）。
- 否则再加宽、再加步数都只会让"平均形变"更准，style-follow 仍然 ~50%。

### 结论
**当前不可用**：① 51.7%（随机）、② 34.4%（天花板的 39%）、③ 0.04（严重偏少）。
即使 v9 跑满 12000 步，只要 ① 不动，就不能接下游。

---

## 14. ★★ 监督对比学习（治 style-follow 卡 50%）

### 为什么
见第 13 节病根：`MSE(g', g_gt)` 被"该字的**平均形变**"主导，风格特异那部分只是总方差里
一小块 → 模型走捷径学到"字 → 平均形变"就够降 loss，风格梯度太弱 → 躺平。
实测 `style-follow` 从 step0 的 51.4% 到 step2000 的 51.7% **完全没动**。
所以要给"风格判别"一个**显式目标**。

### 形式（InfoNCE / 监督对比）
同一个字 c 内，对每个样本：
```
anchor    = g'(c,k)       模型对「字 c + 书家 k」的形变输出
positive  = g_gt(c,k)     k 自己写的那个字的骨架
negatives = { g_gt(c,k') }  同字、但别的书家写的骨架
loss = -log  exp(s(a,p+)/τ) / Σ_{j ∈ {p+} ∪ neg} exp(s(a,j)/τ)
```

### 三个关键设计决定
1. **相似度用余弦，不用负 MSE。**
   余弦是 O(1) 量级，与温度解耦、好调。若用负 MSE 当 logits，`d≈0.33` 会让 logits
   整体偏到 −33 附近、而正负样本之间只差 ~0.002 → softmax 极平、梯度极弱。
2. **负样本只取"同字且不同书家"。**
   同一个 `(字,书家)` 在组内可能有多张样本（组内重复），它们**不能**当负样本，
   否则等于把 positive 当 negative。用 `same_char & diff_style` 掩码。
3. **按行屏蔽 + 跳过无负样本的行。**
   没有负样本的行（如 B=1、或全同书家）整行屏蔽并跳过，避免 softmax 全 −inf 出 NaN。
   用 `-1e4` 而不是 `-inf` 填充。

### 单元测试（用 AST 抽出真实函数体跑的，不是重写）
| 用例 | 结果 | 期望 |
|---|---|---|
| 输出 == 自己的 GT | **0.0000** | 0 ✓ |
| 全用错书家的 GT（最坏） | **14.2755** | 大 ✓ |
| 随机输出 | **2.0534** | ≈ log(7)=1.946 ✓ |
| 边界 B=1 | 0.0000 | 不 NaN ✓ |
| 边界 全同书家（无负样本） | 0.0000 | 不 NaN ✓ |
| 梯度可回传 | True | ✓ |

### 接口
```bash
--w-contr 0.1        # 权重(0=关)
--contr-tau 0.07     # 温度(余弦 logits, CLIP 默认)
```
实现：`tools/train_deform_standalone.py::contrastive_loss(gp, gt, gid, sid, tau)`

### 实验设置：v9c（v9 的**单变量** A/B）
`_sync_work/run_v9c.sh` = v9 配置**逐项相同**，只多 `--w-contr 0.1 --contr-tau 0.07`：
```
--init assets/deform_skel_v5.pt --steps 12000 --batch 2048 --group 32
--residual 0 --width 96 --max-off 6 --dt-ch 1
--w-tv 0.01 --w-fold 0.1 --w-tv-out 0.01 --w-tv-res 0.01
--w-contr 0.1 --contr-tau 0.07 --eval-every 1000 --ckpt 0
--out assets/deform_skel_v9c.pt
```
`--eval-every` 从 2000 收到 **1000** —— 因为要尽快看到 ① 到底动不动。
判读：**只看 style-follow 是否显著离开 50%**；若仍不动，说明病根不在目标函数而在别处。

### 15. ★ 预先备好的多种实现（可一键切换 A/B）

`--contr-mode {cos,mse,margin}` × `--contr-hard N`，实现都在
`tools/train_deform_standalone.py::contrastive_loss`。

| mode | 形式 | 与评估口径 | 要调的超参 | 风险 |
|---|---|---|---|---|
| `cos` | InfoNCE, 相似度 = 余弦/tau | **不一致**（余弦≠MSE）| tau=0.07 | 余弦降了但 style-follow 不涨 |
| `mse` | InfoNCE, 相似度 = **−(d−d_self)**/tau | **一致** | tau≈0.005 | 温度需按"超出量"量级猜 |
| `margin` | 排序 `mean relu(margin + d_self − d_neg)` | **完全一致** | margin≈0.005 | 间隔太大会推坏闭合率 |

**`mse` 为什么要先减 `d_self`**：直接 `−(d/tau)` 时 d≈0.33 会让 logits 整体落在 −33 附近 ——
softmax 平移不变，所以**绝对值本身无害**；真正的问题是正负样本只差 ~0.002，
按绝对尺度猜温度会要么全平要么全尖。先减 d_self 让对角线恒为 0，温度只需覆盖"超出量"。

**`--contr-hard N`（难负样本挖掘）**：组内负样本最多 ~63 个，但真正有区分度的只是那几个
形近书家，全用会被稀释。`_hard_neg()` 用 `masked_fill(-inf)+topk+scatter_` 向量化实现。

**三种 mode 的单元测试**（AST 抽真实函数体，负样本 7 个，log(7)=1.946）：

| mode | 完美 | 全错书家 | 随机 | hard=3 |
|---|---|---|---|---|
| `cos` | 0.0000 | 14.2755 | 2.0534 | 1.5300 |
| `mse` | 0.0000 | 399.85 | 11.7455 | 10.6322 |
| `margin` | 0.0000 | 0.3023 | 0.0272 | 0.0405 |

边界（B=1 / 无负样本）= 0 且不 NaN；梯度有限；非法 mode 抛 `ValueError`。

### 配套工具（都在 `_sync_work/`，已同步到远端根目录）
```bash
# 一键起一个变体（单变量: 只改 --contr-*）
bash run_contr.sh <cos|mse|margin> <w> [hard] [tag] [tau]
bash run_contr.sh cos    0.1          # 余弦
bash run_contr.sh mse    0.1          # 自归一 MSE
bash run_contr.sh margin 0.1          # MSE 排序
bash run_contr.sh cos    0.1 8        # + 难负样本挖掘 top-8
bash run_contr.sh margin 0.5 0 m5     # 加大权重

# 横向汇总所有实验(含 gap = shuffled−correct, 比二值 style-follow 敏感)
bash contr_status.sh
```
`gap = shuffled − correct`（正 = 好）是**比二值 style-follow 敏感得多**的指标 ——
n≈1600 时二值率 1σ≈1.25pp，而 gap 能看见幅度。两个都看。

### 16. 结果

#### v9c（`cos`, w=0.1）—— **负结果**

| 步 | style-follow | gap(shuffled−correct) | 闭合率 | contr | off |
|---|---|---|---|---|---|
| 0 | 51.5% | +0.00047 | 32.3% | — | 0.42 |
| 1000 | 52.7% | +0.00239 | 33.8% | 2.032 | 1.70 |
| 2000 | **49.6%** | **+0.00033** | 35.2% | **2.226** | 3.48 |

**读法**：step1000 那 +1.2pp 是**噪声**（1σ≈1.25pp），到 step2000 就掉回 49.6%、
gap 从 0.00239 **塌回 0.00033**。同时 `contr` 从 2.03 **涨到 2.23**（对比目标本身在变差）。
而**闭合率一路稳定上升**（32.3→33.8→35.2）—— 正是"模型在把平均形变学得更准、
但完全不理会风格"的典型表现。

**结论**：余弦对比在 w=0.1 下**不起作用**。两个可能：① 权重太弱；② 余弦与评估口径
（MSE）不一致 —— 余弦降了不代表 MSE 的 style-follow 会涨。

#### 两个必须记住的坑

**① 各 mode 的 loss 量级差几个数量级，`--w-contr` 在 mode 之间**不可比**：**

| mode | loss 典型量级 | w=0.1 时的贡献 |
|---|---|---|
| `cos` | ~2.0 | ~0.2（占 MSE 的 60%，有实际作用）|
| `margin` | ~0.003 | ~0.0003（**完全可忽略**）|

所以 margin 模式必须给大得多的 w。**但注意：margin 的梯度量级和 MSE 是同一数量级**
（都是 `2(g2−gt)/D`），所以**判读要看 gap，不要看 loss 值**。
另外 `margin` 会**给 gap 设上限**：一旦 gap ≥ margin，梯度就归零 —— margin 设多
就等于"最多把 gap 推到多少"。

**② 进程会静默退出 —— 已查明：是并行会话的 `pkill`，不是 bug。**

现象：v9c 在 step2000 后、margin 在 step0 后无声消失 —— **无 Python traceback、
无 OOM-killer 记录、系统内存充裕（251G 用 7G）、GPU 干净**，`dmesg` 又被内核权限挡住。

**真正原因**：**另一个并行会话在同一台机器上跑实验**，它需要 GPU 时会
`pkill -f train_deform_standalone` 清场 —— 这个模式**同时命中我们的进程**，
于是我们的训练"无声消失"。证据：23:59:12 出现了一个**不是我们启动的**进程，
命令行里带着我们没写过的选项（`--stroke-mod / --stroke-cap / --gate-radius /
--w-tv-stroke`）、输出到 `assets/deform_skel_v10.pt`；而我们的 margin 进程
在同一时刻前后消失。**没有 traceback / 没有 OOM，正是被 SIGKILL 的特征。**

> 教训：**在多会话共用的机器上，"进程无声消失"优先怀疑被外部 kill，
> 而不是先怀疑框架 bug。** 排查顺序应该是：① 看是否有别人的进程占着 GPU；
> ② 看命令行是不是自己的；③ 再怀疑自己。

我最初怀疑的 `expandable_segments` **是错的**（那只是我自己加的唯一非标准设置，
但它是无辜的）。不过摘掉它也没坏处 —— 变量少一个是一个。

**应对**（已落地）：
- 训练器新增 **`--save-every N`** —— 之前只在最后存一次，被 kill 时**整轮白跑、产出为零**。
  这是真正的教训：**长跑必须周期性落盘**，因为你控制不了别人什么时候来抢 GPU。
- 启动前先确认 GPU 上没有别人的训练；启动器里不要再盲目 `pkill`。

#### 并行会话的 v10（别人在跑，不是我们起的）

`logs/deform_v10.log`，`assets/deform_skel_v10.pt`：

```
--residual 0 --stroke-mod 1 --stroke-cap 1.0 --gate-radius 0.25 --w-tv-stroke 0.01
--width 96 --max-off 6 --dt-ch 1 --w-tv 0.01 --w-fold 0.1 --w-tv-out 0.01 --w-tv-res 0.01
--contr-mode margin --contr-margin 0.008 --contr-hard 4 --w-contr 0.5
--save-every 2000
```

**注意它是基于我们这套代码继续做的**：`--contr-mode/--contr-margin/--contr-hard`
都是本轮的产物，它在此基础上又加了 **stroke 调制分支**
（`src/model/deform_skel.py` 里新增 `stroke_conv` + `stroke_style`，zero-init），
方向正好对应第 13 节诊断出的"纯几何形变改不了笔画粗细/墨色"。

所以：**v10 = margin 对比(w=0.5, hard=4) + stroke 调制 + 全局仿射 + 去残差**，
是比我们那轮 `cos@0.1` 更完整的一个组合。评估里多了一项 `stroke_mod`。

> ⚠ 两个会话在**同一份本地文件**上编辑（`tools/train_deform_standalone.py`
> 的 mtime 被对方改成 23:57:07，md5 与远端一致），改动是叠加的、没互相覆盖，
> 但这很危险 —— 后续要串行，或者明确分工。
