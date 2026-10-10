# VAE Latent 契约与使用规程（Latent Contract）

> 2026-10-10 立。**本文档是 VAE 与 DiT 之间接口的规约**：VAE 应当给出什么样的 latent、
> 训练/冻结/换版本时每一步必须做什么、以及违反契约会造成什么后果。
> 起因：本项目**反复**在"latent 约定与尺度"上静默出错（见 §6 失败案例目录 E1–E8），
> 其中 E5（存盘尺度只有应值的 1/5）刚刚被实测确认，代价是多轮实验。
>
> **规则**：任何 VAE 相关的改动（重训/微调/换版本/换 shards），**必须**走 §4 的 runbook 并通过 §1 的全部不变量。

---

## 1. 契约：latent 的定义与 5 条硬性不变量

### 1.1 定义（三段式，`shift` 默认 0）

```
编码:  z_raw    = posterior.sample()        # 或 .mode()，见 I3
存盘:  z_stored = (z_raw − shift) × sf      # shards 里存的就是它
解码:  x̂       = decode(z_stored / sf + shift)
```

- `sf`（`vae_scaling_factor`）**不是超参，是"由实测拟合出来的归一化常数"**——它存在的唯一目的是
  让 `z_stored` 的分布接近 `N(0, I)`，因为**采样起点就是 `N(0, I)`**（`make_eval_cache` 的 noise）。
- `shift` 仅用于 FLUX 家族（其 config 给出 `shift_factor=0.1159`）；**用了 shift 就必须在
  PROVENANCE 与 DiT config 里成对出现**，漏掉一侧就是静默错图。

### 1.2 五条不变量（全部硬性，缺一不可）

| # | 不变量 | 判据（阈值） | 怎么量 |
|---|---|---|---|
| **I1** | **往返保真** | `L1(rec, gt) ≤ 0.005`；PSNR ≥ 30 dB（我们的字帖）；**顶带高频份额 ≥ 0.85× GT** | `tools/vae/hf_roundtrip_ab.py`、`tools/hf_decompose.py` |
| **I2** | **★ 存盘分布 ≈ N(0,1)**（尺度**与**均值都要） | `std ∈ [0.8,1.25]` **且** `mean` 必须通过 `shift` 归零（未用 shift 时 `|mean| ≤ 0.05`）。**`sf = 1/std(z_raw)`、`shift = mean(z_raw)` 必须由实测拟合** | `tools/scale_audit`（见 §4 步骤 2） |
| **I3** | **后验宽度已记录** | 记录相对宽度 `σ/|z_raw|`；若 **> 0.1** 则必须固定用 `sample()` 并写进 PROVENANCE（此时 `mode` 与 `sample` 不等价） | `tools/vae_judge.py` |
| **I4** | **解码约定自检** | `L1(decode(z_stored/sf))` 必须比 `L1(decode(z_stored))` **小一个数量级** | 编码脚本内置自检 |
| **I5** | **DiT 侧三处一致** | `shards.std` ↔ `config.vae_scaling_factor` ↔ `config.vae_path`（**编码与解码必须是同一个 VAE**） | 启动脚本的 preflight（见 §4 步骤 6） |

### 1.3 基准参考（2026-10-10 实测，187 张字帖）

| VAE | raw std | 存盘 std | **数据/噪声 幅度比** | 往返 顶带份额 | 后验相对宽度 |
|---|---|---|---|---|---|
| sd-vae-ft-ema | 5.87 | **1.070** | **1.070** ✅ | 0.816 | 4e-5（点质量） |
| Calli 冻结（v72 用） | 1.085 | **0.198** | **0.198** ❌ | 0.783 | 0.372 |
| Calli topband（v76/77 用） | — | **0.204** | **0.203** ❌ | 0.828 | ~0.43 |
| flux-vae 16ch | — | — | （shift 0.1159） | **0.97** | 2e-4（点质量） |

> **一眼可见的异常**：SD 的数据/噪声比是 1.07（设计正确），**我们的 Calli 系只有 0.20**。

---

## 2. 为什么 I2（单位方差）是硬性要求

**采样起点是 `N(0,I)`（std=1）**，而流匹配在"噪声端 ↔ 数据端"之间插值。数据端尺度错了会同时破坏四件事：

1. **路径 SNR 被扭曲**：数据 std 0.20 vs 噪声 std 1.0 → 数据是比噪声小 5 倍的信号，
   速度目标被噪声项主导 → 模型容量主要花在表示噪声上，**数据分量只是残差 → 相对误差被放大**。
   （SD 的设计正是两端相当：1.07。）
2. **主损失量级被缩放**：L2 随方差缩放 → **5.3² ≈ 28 倍**。我们日志里 `Diff ≈ 0.15`，
   在 SD 尺度下应是 ~4 量级。
3. **所有辅助损失的相对权重被隐性地乘以 ~28**：
   辅助项占比 = `w_aux·L_aux / L_main`，而 `L_main` 随数据方差变。
   → **`w_repa=0.03` 这个"在 SD 尺度下调好的值"，在 Calli 潜码上相对强度被放大了近 30 倍。**
   这解释了为什么"REPA 可能压过纹理"是可检验的机制（v77 正在消融）。
4. **绝对超参被隐性调偏**：`clip_grad_norm(max_norm=1.0)`（train.py:2482）是**绝对**阈值；
   梯度幅度随数据尺度线性缩放 → 尺度改 5.3× 后梯度也放大 5.3×（v72 实际 grad-norm ~0.2）
   → **会开始触发裁剪**。AdamW 本身对全局梯度缩放不敏感（更新按 RMS 归一），
   **但绝对裁剪不是**，必须同步放大 `max_norm`。

**反例（同一批 187 字，cfg1.0/γ0）**：

| | 存盘 std | HF/GT | 用掉自 VAE 天花板 |
|---|---|---|---|
| v68（SD-VAE） | 1.070 ✅ | 0.811 | 0.994 |
| v72（Calli 冻结） | 0.198 ❌ | 0.622 | 0.794 |

> 注：这两行**不是**单变量对照（VAE 全换），仅作"尺度异常与异常的软/高频缺口同时出现"的观察。

---

## 3. VAE 训练时怎么保证 I2

**关键事实：我们的 VAE 损失对 `z` 的全局尺度完全不敏感。**

- `L1(rec, gt)`：解码器可以吸收任意常数缩放 → 对 `z·k` 不变；
- `DINO struct/style`：同样不变；
- `KL`：只约束后验相对 `N(0,1)` 的形状，**不约束 μ 的全局幅度**（且我们只给 1e-6～1e-4）。

→ **`z_raw` 的幅度是一个"自由漂移"的自由度。** 实测漂移量：sd-vae raw std **5.87** →
Calli 微调后 **1.085**（缩 5.4 倍）。**这不是 bug，是没有约束的自由度必然发生的漂移。**

**因此，二选一（推荐两者都做）：**

1. **事后重拟合 `sf`（必须做）**：训练结束 → 用 ≥256 张代表图测 `std(z_raw)` → 令
   **`sf = 1/std(z_raw)`**（不要再沿用 SD 的 0.18215）。SD 官方正是这么做的（0.18215 是拟合值）。
2. **训练时加尺度正则（可选，更稳）**：在 VAE 损失里加 `w_zscale · |std(z_raw) − 1|`
   （或对 `μ` 做 batch-norm 式的尺度归一）。这样"存盘尺度"在训练中就自然落在 1 附近。

> ⚠ 注意 `sf` 是**每个 VAE 版本各自的常数**；换 VAE = 换 `sf` = **新数据变体**（见 §5.3）。

---

## 4. 冻结 / 部署 runbook（7 步，每步都必须留证据）

```
1. 训练 VAE
   python tools/vae/train_dino_calli_vae.py --vae-base <上一版> --output experiments/<run> ...

2. ★ 量 latent 尺度**与均值**并【重拟合 sf + shift】
   测 raw std / raw mean（≥256 张）→ **`sf = 1/std(z_raw)`、`shift = mean(z_raw)`**；
   写进 PROVENANCE 的 `scaling_factor` / `shift_factor`
   （本步缺失 = E5。**只做 scale 不做 shift 是半个修复**：2026-10-10 实测我们 raw 的
    mean = −0.115、std = 1.120 —— raw 本身就是 N(0,1) 级，说明原实现**只抄了 SD 的 sf**；
   FLUX 用 `shift_factor = 0.1159` 修的正是这个均值，而 SD 自己没修，留下 ±0.22 偏移。）

3. 五项不变量验收（I1–I4）
   python tools/vae/hf_roundtrip_ab.py        # I1 往返（含顶带高频份额）
   python tools/vae/judge_vae.py --spec ...   # I3 后验宽度 + 往返
   （I4 由编码脚本内置自检输出）

4. 冻结制品（只读 + 可溯源）
   experiments/vae_frozen/<vaename>/
     ├── diffusion_pytorch_model.safetensors
     ├── config.json
     ├── MD5SUMS.txt            # md5sum -c 必须通过
     └── PROVENANCE.md|json     # 必填: 训练数据/步数/损失配方/encode_mode/
                                #       scaling_factor(拟合值) / raw_std / posterior_std
                                #       / 往返 L1 / 顶带高频份额

5. 用【记录在案的 sf】重编码 shards
   python tools/encode_aug_latents.py --csv ... --vae <frozen> --encode-mode sample \
       --out exp-std/data/shards_<vaename>            # 落盘 z_stored = z_raw × sf
   → shards 目录同时写 PROVENANCE.json（含 scaling_factor / latent_std_measured / 自检 L1）

6. ★ DiT 启动 preflight（三处一致性，必须写进启动脚本）
   - config.latent_shards_dir == 预期的 shards 目录
   - config.vae_path / eval_vae_path == 【编码该 shards 的那个 VAE】（防"潜码来自 A、解码用 B"）
   - config.vae_scaling_factor == shards PROVENANCE 里的 scaling_factor
   - assert abs(shards.std − 1.0) <= 0.25      # I2

7. 登记
   state.md 记录: 新 VAE 版本 / sf / 往返指标 / 后验宽度 / 下游 DiT 臂与对照口径
```

---

## 5. DiT 侧怎么用（六条纪律）

**5.1 `sf` 只出现在解码边界。**
DiT 全程只看 `z_stored`；`sf` 仅用于 `decode(z_stored/sf + shift)`。
→ **改尺度必须"shards ×k" 与 "sf ×k" 同时改**，两侧同时变则解码输入恒等（本修法的安全性来源）。

**5.2 永远不要对已训练模型"事后缩放 latent"。**
实测反例：对训练好的 v72 把 latent ×1.15 / ×1.30 → SSIM 0.6129 → 0.6098 / 0.6042、LPIPS 变差。
原因：破坏了模型学到的解码流形。
→ **尺度问题只能在"编码/数据准备"阶段修，然后重训。**（这两件事过去被我们混为一谈，见 E5。）

**5.3 换 VAE（哪怕只是微调）= 新数据变体 = DiT 必须从头训。**
不能 warm-start：`z` 的尺度变了，而**采样起点 `N(0,I)` 没有跟着变** → 路径形状变了 →
速度目标变了 → 这是一个**不同的学习问题**，不是同一问题的重参数化。

**5.4 辅助损失（REPA 等）的相对权重随尺度变化。**
`w_repa` 应理解为"相对于主损失的占比"。**换了 VAE 尺度，`w_repa` 的等效强度就变了**
（本次：Calli 潜码使 REPA 相对强度放大约 28 倍）。跨尺度比较 `w_repa` 是无意义的。

**5.5 历史数值不可跨尺度比较。**
`Diff`/loss 绝对值、以及一切绝对阈值型超参（grad clip）在改尺度后都不可直接沿用。
改尺度时必须：**放大 `clip_grad_norm` 的 `max_norm`**（按尺度比）、并在文档里标注"此前的 Diff 数值不可比"。

**5.6 评测/落盘复用 `results_dir` 会静默复用旧结果**（见 E6）：
`run_in_mem_eval` 有"`(step,set)` 已完成则跳过"的逻辑 → **每轮用唯一 `results_dir` 并先清空目标目录**。

---

## 6. 失败案例目录（都是本项目真实发生过的静默错误）

| # | 现象 | 根因 | 检出方法 | 纪律 |
|---|---|---|---|---|
| **E1** | 生成图整体发灰、mode 出灰图 | 旧 Calli-VAE 用了非标准约定 `decode(z/sf)`，逼解码器适配被放大 5.5× 的输入 | 解码约定自检（I4） | 只用标准约定 `decode(z_stored/sf)` |
| **E2** | SSIM 崩到 0.158、frag 16.8 | 把为**旧** VAE（σ≈0.865）写的 `calli_decode_noise` 用在新 VAE（σ≈0.39）上 | 换 VAE 时重跑注噪开关 | 注噪补丁与 VAE 版本强绑定，换版本必须重验 |
| **E3** | 图发灰（mean 110 vs 正常 225），**指标却照出** | `load_eval_vae` 只按 `device` 缓存 → 同进程内第二次换 VAE 被静默复用 | 查生成图 mean/std；看是否同一进程评了两个 VAE | 缓存键必须含 VAE 路径；跨 VAE 对比**一进程一模型** |
| **E4** | 16ch 端到端崩（0.478），曾被怀疑 shift | 实测**漏 `+shift` 只掉 2.3 dB** → **不是**主因 | 成对测"带 shift / 不带 shift" | shift 必须成对出现并写进 PROVENANCE |
| **E5** | **存盘 latent std 只有 0.198（应 ~1.0）；FM 损失小 28×、REPA 相对强度放大约 28×** | **微调后未重拟合 `sf`**（沿用 SD 的 0.18215） | §4 步骤 2 的 `std(z_stored)` 检查 | **每次 VAE 变更后必须重拟合 `sf`**；改尺度要同步改 grad clip、且不可跨尺度比 loss |
| **E6** | 海报某几行图**贴错标签**且不报错 | `run_in_mem_eval` 的"(step,set) 已完成则跳过" + 复用 `results_dir` → 把上一轮残留 `g{k}.png` 当本轮结果拷出 | 检查图数与预期是否一致；每轮清空 | **唯一 results_dir + 先清空目标目录** |
| **E7** | 误判"高频已修好"（日志 1.03，离线 0.783） | 训练内指标 `band_fraction` 多除一次 `r.max()` → 与离线普查差 41% | 训练指标必须与离线判据做数值对齐校验 | **训练内指标必须先对齐到 0.00% 误差才允许采信** |
| **E8** | 排名结论随口径翻转 | LPIPS `vgg`（训练日志）与 `alex`（海报/指标表）混用 | 打印时标注 metric 全名 | 指标必须写明实现（`LPIPS(vgg)` / `LPIPS(alex)`） |
| **E9** | 换 shift 后解码图**完全错**（新旧路径差 **227 灰阶**） | 给**已被缓存共享**的 VAE 对象**就地打 `decode` 补丁** → 第二次以不同 shift 调用时拿回**仍带旧补丁**的对象（`shift=0` 为假又不会重包）→ 静默污染 | 同一路径、不同 shift 各取一次, 断言**不是同一对象**; 并对比两条路径的解码图 | **禁止就地改写缓存共享对象**；要改行为就用**不侵入的包装对象**（`_ShiftedVAE`）。缓存键必须含**所有**影响行为的参数 |

---

## 7. 一页速查

```bash
# 量 shards 里 DiT 真正看到的尺度（I2 / I5）
python - <<'EOF'
import glob, numpy as np
z = np.load(sorted(glob.glob("exp-std/data/shards_<name>/shard_000*.npz"))[0])
k = "latents" if "latents" in z else list(z.keys())[0]
a = z[k].astype(np.float32); print("std=%.4f |mean|=%.4f" % (a.std(), abs(a.mean())))
EOF
# 期望: std ∈ [0.8, 1.25]；若 ~0.2 → 命中 E5

# 往返保真 + 顶带高频（I1）
python tools/vae/hf_roundtrip_ab.py
# 后验宽度 + 往返（I3）
python tools/vae/judge_vae.py --spec "name=experiments/vae_frozen/<v>:0.18215:0:sample"

# 尺度修正（不改 VAE、不重编码）：
#   1) shards × k        (k = 1.0 / 当前 std)
#   2) config.vae_scaling_factor × k
#   3) clip_grad_norm max_norm × k      ← 容易漏！
#   4) 从头训 DiT（不可 warm-start）
```

**一句话**：VAE 必须给出**"往返保真合格 + 存盘尺度≈单位方差 + 后验宽度已记录"**的 latent；
`sf` 是**拟合值不是超参**；尺度只能在数据准备阶段修、且必须与 grad clip 同步；
**换 VAE 就是换问题，DiT 从头训。**

---

## 8. 相关文件

- `tools/vae/hf_losses.py` — 高频损失（`topband` 与离线普查同口径）+ 后验 std 惩罚
- `tools/vae/judge_vae.py` — 后验宽度 + 往返判定
- `tools/vae/prep_mode_shards.sh` — shards 制备 + 自检 + 溯源（模板）
- `tools/encode_aug_latents.py` — 编码器（`--encode-mode mode|sample`）
- `tools/hf_decompose.py` / `tools/hf_roundtrip_ab.py` / `tools/check_band_metric.py` — 高频与口径校验
- `state.md` — 每个 VAE 版本与下游 DiT 臂的登记处

---

## 9. 配套文档

- **[`vae_training_recipe.md`](vae_training_recipe.md)** —— **VAE 该用什么 loss、为什么、怎么训、怎么验收**（含上一轮 DINO 版的结果与结论、PatchGAN 对抗的四条稳定性机制、实测定权重的量级表）。本文管接口/尺度约定，那份管训练配方。

