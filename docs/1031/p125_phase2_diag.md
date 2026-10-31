# P125 第二阶段机制诊断（纯 CPU，4090 只读）

日期：2026-10-31 23:0x（Asia/Shanghai）
目标 checkpoint：`assets/p125_20261030/execution/runs/P3A/checkpoints/0020000.pt`（step 20000，EMA）
工具：`tools/diag/`（`common.py` / `t1_writer_timeline.py` / `t2_inject_ablation.py` / `t3_writer_swap.py` / `t4_vae_recon.py`）
产物：`assets/diag_20261031/{t1,t2,t3,t4}/*.json`（4090）

## 0. 一句话结论

**书家条件在速度场里基本是死的**：换错书家几乎不改变 FM 目标（ΔL ≈ +0.002），
空书家条件反而更低（ΔL ≈ −0.027）；三个 style 注入点里两个在**帮倒忙**
（关掉后 loss 下降），只有一个深层 site 在高噪声段有正贡献；
VAE 往返重建几乎无损（SSIM 0.975 / Ink-IoU 0.986），**不是瓶颈**。

因此第三阶段应当先修**条件通路**（writer 表/注入的有效性），
而不是继续加长训练或换 VAE。

## 1. 执行方式与成本（回答“能不能直接用 CPU 完成诊断”）

可以，全部四个诊断在 CPU 上完成，未触碰 4090 的 GPU（`abc` 链仍在训练）。

| 诊断 | 规模 | CPU 用时 | 说明 |
|---|---|---|---|
| T1 writer timeline | 74 行 × 2 噪声 × 6 t × 3 条件 = 2664 前向 | ≈ 35 min（28 线程） | 纯前向，无采样 |
| T2 注入消融 | 16 行 × 6 t × 2 条件 × 4 变体 | ≈ 45 min（24 线程） | 推理期 gain 置零 |
| T3 writer swap | 30 字 × 10 书家 × 3 噪声 | 见 §4 | 含 50 步 Heun 采样 + VAE 解码 + OCR |
| T4 VAE 往返 | 300 张真迹 | 见 §5 | encoder→decoder + 指标 + OCR |

工程约定（写进 `tools/diag/common.py`）：`torch.set_num_threads(--threads)`、
`set_num_interop_threads(2)`、`os.nice(10)`，产物写小 JSON（原子替换），
训练目录与权重只读。

## 2. T1 —— 书家条件时间线（核心证据）

协议：固定真迹 latent `z0`、STD 掩码、书体、噪声；在
`z_t=(1-t)z0+tε`，`t∈{0.10,0.25,0.40,0.60,0.80,0.95}` 上分别用
**正确书家 / 错误书家 / 空书家（id=47）** 计算 FM 目标。

| t | L_correct | ΔL_wrong | ΔL_null |
|---|---|---|---|
| 0.10 | 0.5956 | +0.0006 | −0.0141 |
| 0.25 | 0.8321 | +0.0021 | −0.0299 |
| 0.40 | 0.6158 | −0.0000 | −0.0244 |
| 0.60 | 0.5987 | +0.0026 | −0.0209 |
| 0.80 | 0.9262 | +0.0038 | −0.0174 |
| 0.95 | 1.5609 | −0.0077 | −0.0564 |

分组（全部同向）：

- 书体：楷（script 0）ΔL_wrong +0.0012、行（3）+0.0021、隶（4）−0.0051
- 字频：常见 +0.0045、中频 −0.0002、罕见 −0.0052
- **书家桶：全部落在 generic 槽（10–46）**——P125 训练 CSV 里没有 Top10 槽（0–9）
  的样本，`init_p1_table.py` 克隆出来的 0–9 行**从未被数据训练过**

判读：ΔL_wrong 与噪声量级同阶（±0.008），ΔL_null 稳定为负。
即“告诉模型正确的书家”不比“告诉它一个错的书家”更有用，
而且比“不给书家”更差。**书家条件当前是负贡献。**

## 3. T2 —— 注入尺度消融

方法：不改权重文件，把 `style_injections[i].gain` 置零（`injections.py` 里
整个 delta 乘以 gain，置零即精确关闭该 site），在相同 `z_t` 上重算 FM 目标。

site 几何（forward hook 实测，不是按命名猜的）：

| site | 分辨率 | 通道 |
|---|---|---|
| 0 | 32×32 | 192 |
| 1 | 16×16 | 384 |
| 2 | 8×8 | 576 |

| 变体 | mean ΔL_site | 逐 t 特征 |
|---|---|---|
| off_0（32×32） | **−0.0161** | 所有 t 都是负（−0.009 ~ −0.024）：关掉更好 |
| off_1（16×16） | **−0.0063** | 全 t 轻微负（−0.000 ~ −0.015）：关掉更好 |
| off_2（8×8） | +0.0012 | 仅高噪声段为正：t=0.8 **+0.0143**、t=0.95 +0.0054 |

所有变体下 `dL_wrong` 仍 ≈ 0（±0.01），**没有任何 site 承载可测的书家区分度**
（冒烟里出现的 “retention 17×” 是分母≈0 的比值假象，已改为分母 <5e-3 时记 None）。

判读：32×32 与 16×16 的注入在往速度场里加误差；只有 8×8 深层注入在
高噪声（结构成型）阶段有正贡献。书家信息既没进来，注入本身还在拖后腿。

## 4. T3 —— Writer-Swap 生成诊断

协议：固定字/STD/书体/噪声，只换 writer id；方向 = 生成 latent 差向量与
真实差向量的余弦（同字、同书家对）；幅度 = ‖d_gen‖/‖d_true‖；
内容稳定 = OCR Top-1。

（数字待 T3 全量完成后填入。）

## 5. T4 —— VAE 往返保真

协议：真迹 → encoder → latent → decoder → 重建，与真迹用**同一套** OCR 与
墨迹指标对比（不与生成面板混算）。n=300 张 trusted 真迹，CPU 约 20 分钟。

| 指标 | 真迹 → 重建 | 参考：生成侧（P3A@20k, val100） |
|---|---|---|
| OCR Top-1 | **99.66% → 99.66%（drop 0.0）** | 生成侧内容正确率远低于此 |
| SSIM | **0.9855** | 0.5857 |
| Ink-IoU | **0.9906** | 0.2725 |
| Skeleton-IoU | 0.8662 | 0.0228 |
| frag（断裂） | 1.007 | 2.97 |
| hole（空洞） | 0.0329 | 0.0449 |

判读：**VAE 往返几乎无损**，重建的 OCR 与真迹完全一致、墨迹域 IoU 0.99。
生成图与真迹之间那道巨大差距（SSIM 0.586 / Ink-IoU 0.27 / Skeleton 0.023）
**不可能来自 VAE**，只能来自生成器与条件通路。
→ 第三阶段不必重训 VAE，D 方向（改 decoder / 重训 VAE）优先级下调。

## 6. 对第三阶段的建议（待 T3/T4 收口后定稿）

- 若 T3 方向≈随机、幅度<1 → 与 T1/T2 一致，问题在**条件通路本身**：
  优先做“条件有效性”方向（writer 表训练不足 + 注入尺度失配），
  而不是继续延长训练或换 VAE。
- VAE 已排除：重建几乎无损，D 方向（重训 VAE / 改 decoder）优先级下调。
- 数据层需要单独确认：Top10 槽（0–9）无训练样本，若后续要按 Top10 书家出图，
  必须先补数据或显式声明该行为空表。

## 7. 复现命令（4090，CPU，只读）

```bash
cd /root/Workspace/xy/DiT
P=/opt/conda/envs/cu121/bin/python
nice -n 10 $P -u -m tools.diag.t1_writer_timeline --n 96 --threads 28 --out assets/diag_20261031
nice -n 10 $P -u -m tools.diag.t2_inject_ablation --rows 16 --threads 24 --out assets/diag_20261031
nice -n 10 $P -u -m tools.diag.t3_writer_swap --chars 30 --writers 10 --seeds 3 --threads 20 --out assets/diag_20261031
nice -n 10 $P -u -m tools.diag.t4_vae_recon --n 300 --threads 8 --out assets/diag_20261031
```
