# FIG13 素材：同一 latent 的多 decoder 交叉解码

对应全周期报告 FIG13 左侧面板「K 原 Decoder / K+D1dec / K+GAR / GT」。
本目录补齐了该项此前缺失的**同 latent 实际生成图**。

## 目录内容

| 目录 | 内容 | 来源（4090 侧） |
|---|---|---|
| `K_A/` | K 原装 sd-vae decoder 输出 | `assets/f_20261101/eval/K_A/` |
| `K_plus_D1dec/` | K 生成器 + **D1 训出的 decoder**（真正换装） | `assets/d_20261101/eval/K_plus_D1dec/` |
| `D1b_2500/` | D1b@2500 评测输出（**见下方警告：实为 no-op**） | `assets/d_20261101/eval/D1b_2500/` |
| `GAR_s000672/` | K 生成器 + GAR decoder | `assets/d_20261101/eval/GAR_s000672/` |
| `manifest.csv` | 逐字元数据（400 行 = 4 臂 × 100 字） | 本目录 |

每个子目录仅保留 `eval_samples_ctrl/<step>/eval200fix/` 下的 `g*.png`（生成）、
`gt*.png`（真值）、`metrics.json`、`protocol.json`、`DONE.json`。
`t3_*` writer-swap 网格与 `posters/` 未取（体积 13M/臂，非 FIG13 所需）。

**STD 条件输入只保留一份**：`eval200fix_input_g/`（100 张）在六条链中**逐字节相同**
（合并哈希均为 `a126292989fc874b`），它是共享的条件输入而非 decoder 输出，
因此只在 `K_A/` 下保留一份，其余臂删除，避免把同一份数据提交六遍。

## 可比性核验（作图脚本会自动重跑这些检查，不通过就拒绝作图）

四条链的 `protocol.json` 逐字段比对：

| 字段 | 值 | 含义 |
|---|---|---|
| `checkpoint_sha256` | `60a21973…`（**四者相同**）| 生成器确实是未改动的 K |
| `seed` | 0（相同） | 同噪声 |
| `sampler / num_steps / cfg / gamma` | heun / 50 / 1.0 / 0.0（相同）| 同采样协议 |
| `vae_sha256` | `b29b4139…`（相同）| 同 VAE |
| `dataset_sha256` | `41d2ad62…`（相同）| 同面板 |
| 100 张 `gt*.png` 的合并哈希 | **四者相同** | 行 i 在四列中是同一个字 |

因此四列**只差 decoder**，差异可直接归因于解码器 —— 这正是 FIG13 想证明的事。

## ⚠️ 关键发现：D1b_2500 是一个空操作

取回素材时核对发现（见 `manifest.csv` 与 `docs/1031/GAR_DECODER.md` 的更正小节）：

1. `D1b_2500/eval.json` 的 `decoder_override` 是 **`null`** —— 评测从未换装 D1b 训好的 decoder；
2. 它的 200 张 `g*.png` 与 `K_A` **逐字节完全相同**（manifest 中 100/100 输出哈希相同）；
3. 它的 `metrics.json` 与 K **完全相同**（SSIM 0.594575 / LPIPS 0.346214）。

即 D1b 的训练没问题（checkpoint 里确有 138 张 decoder 张量），
**坏的是它的评测调用**。因此：

- 此前把 D1b 当「decoder 起作用的干净对照」是**错的**，它没提供任何换装信息；
- 但 GAR 的改进幅度**依然成立**，因为它实际对照的就是 K（冻结基线）；
- 真正的「换 decoder」证据是 **`K_plus_D1dec`**（`decoder_override` 指向
  `runs/D1/checkpoints/0002500.pt`）：SSIM **0.598173**、LPIPS **0.341561**，
  图像与 K 有实际差异（mean|Δ|=0.0081）。

**教训：** `--use-ckpt-decoder` 与 `decoder_override` 是两个不同的开关。前者未被正确
触发时，评测会**静默**地用原装 decoder 出图、并照常打印一整套指标。凡换装类实验，
必须核对 `eval.json` 的 `decoder_override` 字段，并抽查输出 PNG 与基线是否逐字节相同
—— 只看指标表格发现不了这个故障。

## 复现

```bash
# 图（自动跑可比性核验，失败即退出，不会画出误导性对比）
python assets/metrics_index/plot_fig13_crossdecode.py
# -> assets/metrics_index/fig13_crossdecode.png

# 逐像素差异统计
#   K vs D1b = 0.00000（逐字节相同）
#   K vs K+D1dec = 0.00809；K vs GAR@672 = 0.00465
```

**缺失字段显式标 NA**：`decoder_ckpt` 列对 K_A、D1b_2500 记
`NA(stock sd-vae)`（D1b 行另有说明），未猜测。
